# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python m5_grid_execution_audit.py
Módulo 5 Completo:
1. Máquina de Estados de Ventana Deslizante en RAM O(1) (Mark2SlidingWindowGridExecutor)
   con cálculo exacto de PnL realizado y no realizado (Mark-to-Market), Rebalanceo Infinito
   (Trailing Grid Up/Down), contador 24h/Total y MDD 7D.
2. Adaptador Asíncrono Endurecido para Binance Futures USDⓈ-M (BinanceMark2LiveGridBroker)
   con SecretStr, sincronización de reloj (/fapi/v1/time), consulta de comisiones reales
   (/fapi/v1/commissionRate), Margen ISOLATED, Apertura de Posición Base Inicial,
   despliegue por lotes (/fapi/v1/batchOrders), Guarda Anti -2021, Purga Atómica
   (/fapi/v1/allOpenOrders) y Auditoría Real de PnL (/fapi/v1/userTrades).
3. Persistencia Fuera de Banda en SQLite modo WAL (Cold Path).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import datetime
from decimal import Decimal, ROUND_DOWN
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import stat
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

try:
    import aiohttp
    import yarl
except ImportError:
    aiohttp = None
    yarl = None

from core_mark2_contracts import (
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    IGridBrokerExecutionPort,
)
from m3_grid_decision_orchestrator import LotAndTickQuantizer

logger = logging.getLogger("M5_GridExecutionAudit")

MARK2_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_ALGO_QUERY_WINDOW_MS = (7 * 86_400 * 1000) - 60_000   # Estricto < 7 días para consultas históricas AlgoOrder
MAX_ALGO_HISTORY_AGE_MS = (89 * 86_400 * 1000)            # Límite máximo < 90 días de antigüedad en Binance
ALGO_CLIENT_ID_PATTERN = re.compile(r"^[\.A-Z\:/a-z0-9_-]{1,36}$")


class BinanceApiError(RuntimeError):
    """Excepción estructurada para errores de la API de Binance (preserva herencia RuntimeError)."""

    def __init__(self, code: int, endpoint: str, msg: str):
        self.code = int(code)
        self.endpoint = str(endpoint)
        self.msg = str(msg)
        super().__init__(f"Binance API Error [{self.code}] en {self.endpoint}: {self.msg}")


# ============================================================================
# 1. SEGURIDAD DE CREDENCIALES Y CARGADOR DETERMINISTA DE .ENV
# ============================================================================

class SecretStr:
    """Enmascara API Key y Secret en memoria para impedir fugas en logs o tracebacks."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value) and "pon_aqui" not in self._value.lower()

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"


class EnvConfigLoader:
    """Cargador determinista de archivo .env sin dependencias externas."""

    @staticmethod
    def load(filepath: str = ".env") -> Dict[str, str]:
        cfg: Dict[str, str] = {}
        resolved = (
            filepath
            if os.path.isabs(filepath)
            else os.path.join(MARK2_DIR, filepath)
        )
        target_path = resolved if os.path.exists(resolved) else filepath
        if os.path.exists(target_path):
            try:
                st_mode = os.stat(target_path).st_mode
                if os.name != "nt" and (st_mode & (stat.S_IRWXG | stat.S_IRWXO)):
                    logger.warning(
                        "🔒 [M5 Seguridad] El archivo %s tiene permisos abiertos. Usa: chmod 600 %s",
                        target_path,
                        target_path,
                    )
            except OSError:
                pass
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    clean = line.strip()
                    if not clean or clean.startswith("#") or "=" not in clean:
                        continue
                    k, v = clean.split("=", 1)
                    cfg[k.strip()] = v.strip().strip("'").strip('"')
        return cfg


# ============================================================================
# 2. MODELOS DE ESTADO DE ÓRDENES ACTIVAS EN EL LIBRO DE BINANCE
# ============================================================================

@dataclass(slots=True)
class LiveGridOrderRecord:
    """Representa una orden LIMIT viva dentro de la ventana activa del Grid."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str          # "BUY" o "SELL"
    price: float
    quantity: float
    reduce_only: bool
    status: str        # "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED"
    created_at_ms: int


@dataclass(slots=True)
class ActiveGridBotSession:
    """Estado integral en memoria de un Bot Grid ($6.00 USD -> $18.00 USD) en ejecución."""
    session_id: int
    symbol: str
    plan: GeometricGridPlanPOD
    base_position_qty: float
    base_entry_price: float
    open_limit_orders: Dict[float, LiveGridOrderRecord]
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    realized_pnl_usdt: float = 0.0
    matched_trades_total: int = 0


# ============================================================================
# 3. MOTOR DE VENTANA DESLIZANTE Y REBALANCEO INFINITO (HOT-PATH RAM + WAL)
# ============================================================================

class Mark2SlidingWindowGridExecutor:
    """
    Controlador de Ejecución de Malla Geométrica en Memoria RAM O(1) y SQLite WAL:
    - Mantiene los N (2..169) niveles virtuales del Grid en memoria RAM.
    - Administra la ventana activa deslizante de órdenes LIMIT (BUY debajo, SELL arriba).
    - Al llenarse un escalón de compra en P_k, registra el inventario y programa la venta en P_{k+1}.
    - Al llenarse un escalón de venta en P_k, calcula el PnL neto real sobre (P_sell - P_buy) * qty - fees.
    - Actualiza el PnL no realizado en vivo con el markPrice real de Binance.
    """

    def __init__(
        self,
        plan: GeometricGridPlanPOD,
        rules: ExchangeContractRulePOD,
        journal_db_path: str = "prosperity_mark2_journal.db",
        maker_fee: float = 0.00018,
    ):
        self.plan = plan
        self.rules = rules
        if journal_db_path != ":memory:" and not os.path.isabs(journal_db_path):
            self.journal_db_path = os.path.join(MARK2_DIR, journal_db_path)
        else:
            self.journal_db_path = journal_db_path
        self.maker_fee = maker_fee
        self.geom_ratio = 1.0 + (plan.geometric_step_pct / 100.0)
        self.virtual_levels: List[float] = list(plan.all_virtual_levels)
        self.active_buys: List[float] = list(plan.active_buy_window)
        self.active_sells: List[float] = list(plan.active_sell_window)

        self.realized_pnl_usdt: float = 0.0
        self.unrealized_pnl_usdt: float = 0.0
        self.completed_crosses: int = 0
        self.trailing_shifts_count: int = 0

        # Registro de lotes abiertos por nivel para cálculo exacto de PnL realizado y Mark-to-Market
        self.open_lots_entry_map: Dict[float, float] = {}
        self.net_inventory_qty: float = 0.0
        self.avg_entry_price: float = 0.0

        self.start_timestamp_utc: float = time.time()
        self.matched_timestamps_24h: Deque[float] = deque()
        self.equity_curve_7d: Deque[Tuple[float, float]] = deque()
        self.equity_curve_7d.append((self.start_timestamp_utc, plan.initial_capital_usdt))

        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.journal_db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
        self._init_wal_journal()

    def _get_db_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        return sqlite3.connect(self.journal_db_path, timeout=15.0)

    def _init_wal_journal(self) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mark2_grid_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL,
                    symbol TEXT,
                    stage TEXT,
                    filled_side TEXT,
                    filled_price REAL,
                    counter_price REAL,
                    qty REAL,
                    net_pnl_usdt REAL,
                    cumulative_equity_usdt REAL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mark2_bot_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opened_at TEXT,
                    closed_at TEXT,
                    symbol TEXT,
                    direction TEXT,
                    leverage INTEGER,
                    initial_capital_usdt REAL,
                    final_equity_usdt REAL,
                    net_pnl_usdt REAL,
                    roi_pct REAL,
                    total_crosses INTEGER,
                    trailing_shifts INTEGER,
                    mdd_7d_pct REAL,
                    status TEXT,
                    close_reason TEXT
                );
                """
            )
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

        if self.journal_db_path != ":memory:":
            self.rehydrate_from_wal_journal()

    def rehydrate_from_wal_journal(self) -> Dict[str, Any]:
        """
        Rehidrata el estado de ejecución activa desde SQLite WAL (`mark2_grid_cycles`)
        para el símbolo actual, considerando únicamente los ciclos posteriores al
        último cierre registrado en `mark2_bot_sessions`.
        Restaura:
        - `start_timestamp_utc` (para preservar el tiempo de ejecución real > 0d 0h 0m).
        - `realized_pnl_usdt` y `completed_crosses`.
        - `matched_timestamps_24h` y `equity_curve_7d`.
        """
        conn = self._get_db_connection()
        try:
            last_closed_ts = 0.0
            row_sess = conn.execute(
                """
                SELECT closed_at FROM mark2_bot_sessions
                WHERE symbol = ? AND closed_at IS NOT NULL AND closed_at != ''
                ORDER BY session_id DESC LIMIT 1
                """,
                (self.plan.symbol,),
            ).fetchone()
            if row_sess and row_sess[0]:
                try:
                    raw_dt = str(row_sess[0]).replace("Z", "+00:00")
                    dt_obj = datetime.datetime.fromisoformat(raw_dt)
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
                    last_closed_ts = dt_obj.timestamp()
                except Exception:
                    last_closed_ts = 0.0

            rows = conn.execute(
                """
                SELECT timestamp_utc, filled_side, net_pnl_usdt, cumulative_equity_usdt
                FROM mark2_grid_cycles
                WHERE symbol = ? AND timestamp_utc > ?
                ORDER BY timestamp_utc ASC, id ASC
                """,
                (self.plan.symbol, last_closed_ts),
            ).fetchall()

            if not rows:
                return {"rehydrated": False, "cycles_count": 0, "realized_pnl_usdt": 0.0}

            now_ts = time.time()
            cutoff_24h = now_ts - 86400.0
            cutoff_7d = now_ts - (7.0 * 86400.0)

            first_ts = float(rows[0][0])
            if first_ts > 0.0:
                self.start_timestamp_utc = min(self.start_timestamp_utc, first_ts)

            total_pnl = 0.0
            crosses = 0
            self.matched_timestamps_24h.clear()
            self.equity_curve_7d.clear()
            self.equity_curve_7d.append((self.start_timestamp_utc, self.plan.initial_capital_usdt))

            for r_ts, r_side, r_pnl, r_eq in rows:
                ts_val = float(r_ts or now_ts)
                pnl_val = float(r_pnl or 0.0)
                eq_val = float(r_eq or (self.plan.initial_capital_usdt + total_pnl + pnl_val))
                if str(r_side).upper() == "SELL" or abs(pnl_val) > 1e-12:
                    total_pnl += pnl_val
                    crosses += 1
                    if ts_val >= cutoff_24h:
                        self.matched_timestamps_24h.append(ts_val)
                if ts_val >= cutoff_7d:
                    self.equity_curve_7d.append((ts_val, eq_val))

            self.realized_pnl_usdt = round(total_pnl, 4)
            self.completed_crosses = crosses
            return {
                "rehydrated": True,
                "cycles_count": len(rows),
                "completed_crosses": self.completed_crosses,
                "realized_pnl_usdt": self.realized_pnl_usdt,
                "start_timestamp_utc": self.start_timestamp_utc,
            }
        except Exception as exc:
            logger.debug("⚠️ [M5 WAL Rehydrate %s]: %s", self.plan.symbol, exc)
            return {"rehydrated": False, "cycles_count": 0, "realized_pnl_usdt": 0.0}
        finally:
            if self._memory_conn is None:
                conn.close()

    def get_live_effective_leverage(self, current_price: float) -> float:
        """
        Calcula el Apalancamiento Efectivo Real del inventario abierto sobre la equidad viva del bot:
        L_eff = (|Q_net| * P_mark) / Equity_viva
        """
        live_eq = max(
            0.50,
            self.plan.initial_capital_usdt + self.realized_pnl_usdt + self.unrealized_pnl_usdt,
        )
        notional_open = abs(self.net_inventory_qty) * max(0.0, current_price)
        return round(notional_open / live_eq, 2)

    def update_mark_to_market(self, current_mark_price: float) -> float:
        """
        Actualiza el PnL no realizado (unrealized_pnl_usdt) en función del markPrice
        real de Binance (/fapi/v1/premiumIndex) sobre el inventario abierto del Grid.
        """
        if current_mark_price <= 0.0 or abs(self.net_inventory_qty) <= 1e-12 or self.avg_entry_price <= 0.0:
            self.unrealized_pnl_usdt = 0.0
            return 0.0
        if self.net_inventory_qty > 0.0:
            upnl = (current_mark_price - self.avg_entry_price) * self.net_inventory_qty
        else:
            upnl = (self.avg_entry_price - current_mark_price) * abs(self.net_inventory_qty)
        self.unrealized_pnl_usdt = round(upnl, 4)
        return self.unrealized_pnl_usdt

    def on_limit_order_filled(
        self, side: str, fill_price: float, now_ts: Optional[float] = None
    ) -> Dict[str, float | str]:
        """
        Procesa en O(1) la ejecución de una orden límite del Grid con Mitigación Dinámica
        de Riesgo de Apalancamiento:
        - Si se llenó un BUY en P_k -> Registra el precio de compra exacto y coloca SELL en P_{k+1}.
          Solo repone un nuevo escalón BUY inferior si el Apalancamiento Efectivo de Inventario
          está dentro del umbral seguro Anti-Liquidación (`max_adverse_effective_leverage`).
        - Si se llenó un SELL en P_k -> Calcula el PnL neto exacto entre el precio de venta y su
          precio de compra, descontando las comisiones Maker reales, y coloca BUY en P_{k-1}.
        """
        ts = now_ts or time.time()
        side_up = side.upper()
        qty = self.plan.qty_per_line

        if side_up == "BUY":
            if fill_price in self.active_buys:
                self.active_buys.remove(fill_price)

            target_sell = LotAndTickQuantizer.quantize_price(
                fill_price * self.geom_ratio, self.rules.tick_size
            )

            # Si teníamos inventario SHORT previo (net_inventory_qty < 0), este BUY cierra lote corto
            if self.net_inventory_qty < -1e-12:
                entry_short_price = self.open_lots_entry_map.pop(fill_price, self.avg_entry_price or target_sell)
                gross_pnl_usdt = (entry_short_price - fill_price) * qty
                roundtrip_fee_usdt = (fill_price + entry_short_price) * qty * self.maker_fee
                exact_net_harvest_usdt = round(gross_pnl_usdt - roundtrip_fee_usdt, 6)
                self.net_inventory_qty = round(self.net_inventory_qty + qty, 8)
                if abs(self.net_inventory_qty) <= 1e-12:
                    self.unrealized_pnl_usdt = 0.0
                self.completed_crosses += 1
                self.realized_pnl_usdt = round(
                    self.realized_pnl_usdt + exact_net_harvest_usdt, 4
                )
                live_eq = round(
                    self.plan.initial_capital_usdt + self.realized_pnl_usdt + self.unrealized_pnl_usdt,
                    4,
                )
                self._update_rolling_telemetry(ts, live_eq)
                self._record_cycle_wal(
                    ts, "BUY", fill_price, entry_short_price, exact_net_harvest_usdt, live_eq
                )
            else:
                self.open_lots_entry_map[target_sell] = fill_price
                prev_qty = max(0.0, self.net_inventory_qty)
                new_qty = prev_qty + qty
                self.avg_entry_price = (
                    ((self.avg_entry_price * prev_qty) + (fill_price * qty)) / new_qty
                    if new_qty > 0.0
                    else fill_price
                )
                self.net_inventory_qty = round(new_qty, 8)

            if target_sell not in self.active_sells:
                self.active_sells.append(target_sell)
                self.active_sells.sort()

            # Escudo de Mitigación de Inventario: solo abrir un nuevo escalón inferior si el
            # apalancamiento efectivo proyectado no rebasa el techo seguro del plan
            live_eff_lev = self.get_live_effective_leverage(fill_price)
            safe_eff_cap = max(3.5, self.plan.max_adverse_effective_leverage)
            lowest_buy = min(self.active_buys) if self.active_buys else fill_price
            next_buy_down = LotAndTickQuantizer.quantize_price(
                lowest_buy / self.geom_ratio, self.rules.tick_size
            )
            total_active = len(self.active_buys) + len(self.active_sells)
            if (
                live_eff_lev < safe_eff_cap
                and self.virtual_levels
                and next_buy_down >= self.virtual_levels[0]
                and total_active < self.plan.active_window_orders
                and next_buy_down not in self.active_buys
            ):
                self.active_buys.insert(0, next_buy_down)

            return {
                "event": "BUY_FILLED_FLIP_TO_SELL",
                "new_order_side": "SELL",
                "new_order_price": target_sell,
                "qty": qty,
            }

        # Caso SELL FILLED -> Emparejamiento completado (o apertura de lote corto si es SHORT sin inventario LONG)
        if fill_price in self.active_sells:
            self.active_sells.remove(fill_price)

        target_buy = LotAndTickQuantizer.quantize_price(
            fill_price / self.geom_ratio, self.rules.tick_size
        )
        entry_buy_price = self.open_lots_entry_map.pop(
            fill_price,
            self.avg_entry_price if (self.net_inventory_qty > 0.0 and self.avg_entry_price > 0.0) else target_buy,
        )

        gross_pnl_usdt = (fill_price - entry_buy_price) * qty
        roundtrip_fee_usdt = (fill_price + entry_buy_price) * qty * self.maker_fee
        exact_net_harvest_usdt = round(gross_pnl_usdt - roundtrip_fee_usdt, 6)

        if self.net_inventory_qty > 1e-12:
            self.net_inventory_qty = round(max(0.0, self.net_inventory_qty - qty), 8)
            if self.net_inventory_qty <= 1e-12:
                self.unrealized_pnl_usdt = 0.0
        elif self.plan.direction == GridDirection.SHORT:
            prev_short = abs(min(0.0, self.net_inventory_qty))
            new_short = prev_short + qty
            self.avg_entry_price = (
                ((self.avg_entry_price * prev_short) + (fill_price * qty)) / new_short
                if new_short > 0.0
                else fill_price
            )
            self.net_inventory_qty = -round(new_short, 8)
            self.open_lots_entry_map[target_buy] = fill_price

        if target_buy not in self.active_buys:
            self.active_buys.append(target_buy)
            self.active_buys.sort()

        highest_sell = max(self.active_sells) if self.active_sells else fill_price
        next_sell_up = LotAndTickQuantizer.quantize_price(
            highest_sell * self.geom_ratio, self.rules.tick_size
        )
        total_active = len(self.active_buys) + len(self.active_sells)
        if (
            self.virtual_levels
            and next_sell_up <= self.virtual_levels[-1]
            and total_active < self.plan.active_window_orders
            and next_sell_up not in self.active_sells
        ):
            self.active_sells.append(next_sell_up)

        self.completed_crosses += 1
        self.realized_pnl_usdt = round(
            self.realized_pnl_usdt + exact_net_harvest_usdt, 4
        )
        live_eq = round(
            self.plan.initial_capital_usdt + self.realized_pnl_usdt + self.unrealized_pnl_usdt,
            4,
        )

        self._update_rolling_telemetry(ts, live_eq)
        self._record_cycle_wal(
            ts, "SELL", fill_price, entry_buy_price, exact_net_harvest_usdt, live_eq
        )

        return {
            "event": "SELL_FILLED_HARVEST_AND_FLIP_TO_BUY",
            "new_order_side": "BUY",
            "new_order_price": target_buy,
            "qty": qty,
            "net_harvest_usdt": exact_net_harvest_usdt,
            "cumulative_equity_usdt": live_eq,
        }

    def _update_rolling_telemetry(self, ts: float, live_equity_usdt: float) -> None:
        """Actualiza la cola de trades emparejados en 24h y la curva de MDD en 7 días."""
        self.matched_timestamps_24h.append(ts)
        self.equity_curve_7d.append((ts, live_equity_usdt))

        cutoff_24h = ts - 86_400.0
        while self.matched_timestamps_24h and self.matched_timestamps_24h[0] < cutoff_24h:
            self.matched_timestamps_24h.popleft()

        cutoff_7d = ts - (7.0 * 86_400.0)
        while self.equity_curve_7d and self.equity_curve_7d[0][0] < cutoff_7d:
            self.equity_curve_7d.popleft()

    def get_matched_trades_24h_and_total(self) -> Tuple[int, int]:
        """Devuelve (trades_24h, trades_totales) igual que en la tarjeta de Binance."""
        return len(self.matched_timestamps_24h), self.completed_crosses

    def compute_mdd_7d_pct(self) -> float:
        """Calcula el Maximum Drawdown (%) en ventana rodante de 7 días."""
        if not self.equity_curve_7d:
            return 0.0
        peak = self.equity_curve_7d[0][1]
        max_dd = 0.0
        for _, eq in self.equity_curve_7d:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0 if peak > 0.0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    def check_and_apply_trailing_grid(
        self, current_price: float, quote_obi_l1: float
    ) -> Optional[Dict[str, float | str]]:
        """
        Rebalanceo Infinito (Trailing Grid Up / Down):
        Si el precio supera el límite superior o cae bajo el inferior con confirmación
        de desbalance positivo/negativo en el libro L1 de Binance (quote_obi_l1),
        desplaza toda la malla un escalón geométrico.
        """
        if not self.virtual_levels:
            return None

        top_level = self.virtual_levels[-1]
        bottom_level = self.virtual_levels[0]
        next_top = LotAndTickQuantizer.quantize_price(
            top_level * self.geom_ratio, self.rules.tick_size
        )
        next_bottom = LotAndTickQuantizer.quantize_price(
            bottom_level / self.geom_ratio, self.rules.tick_size
        )

        # 1. Ruptura Alcista con libro comprador (quote_obi_l1 > 0) -> Trailing Ascendente
        if self.plan.trailing_up and current_price >= next_top and quote_obi_l1 > 0.0:
            removed_bottom = self.virtual_levels.pop(0)
            self.virtual_levels.append(next_top)
            if self.active_buys and self.active_buys[0] == removed_bottom:
                self.active_buys.pop(0)
            new_support_buy = self.virtual_levels[-2]
            if new_support_buy not in self.active_buys:
                self.active_buys.append(new_support_buy)
                self.active_buys.sort()
            self.trailing_shifts_count += 1
            return {
                "trailing_action": "TRAILING_UP_EXECUTED",
                "cancelled_level": removed_bottom,
                "added_buy_level": new_support_buy,
                "new_range_lower": self.virtual_levels[0],
                "new_range_upper": self.virtual_levels[-1],
            }

        # 2. Ruptura Bajista con libro vendedor (quote_obi_l1 < 0) -> Trailing Descendente
        if self.plan.trailing_down and current_price <= next_bottom and quote_obi_l1 < 0.0:
            removed_top = self.virtual_levels.pop(-1)
            self.virtual_levels.insert(0, next_bottom)
            if self.active_sells and self.active_sells[-1] == removed_top:
                self.active_sells.pop(-1)
            new_resist_sell = self.virtual_levels[1]
            if new_resist_sell not in self.active_sells:
                self.active_sells.insert(0, new_resist_sell)
                self.active_sells.sort()
            self.trailing_shifts_count += 1
            return {
                "trailing_action": "TRAILING_DOWN_EXECUTED",
                "cancelled_level": removed_top,
                "added_sell_level": new_resist_sell,
                "new_range_lower": self.virtual_levels[0],
                "new_range_upper": self.virtual_levels[-1],
            }

        return None

    def _record_cycle_wal(
        self,
        ts: float,
        side: str,
        filled_p: float,
        counter_p: float,
        net_pnl_usdt: float,
        live_eq: float,
    ) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute(
                """
                INSERT INTO mark2_grid_cycles
                (timestamp_utc, symbol, stage, filled_side, filled_price, counter_price, qty, net_pnl_usdt, cumulative_equity_usdt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    self.plan.symbol,
                    self.plan.compounding_stage.value,
                    side,
                    filled_p,
                    counter_p,
                    self.plan.qty_per_line,
                    net_pnl_usdt,
                    live_eq,
                ),
            )
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

    def record_session_closed_wal(
        self,
        close_reason: str,
        final_net_pnl: Optional[float] = None,
        status: str = "COMPLETED",
    ) -> int:
        """
        Registra el cierre de la sesión de un Bot Grid en la tabla `mark2_bot_sessions`
        de SQLite WAL (`prosperity_mark2_journal.db`).
        """
        net_pnl = (
            round(float(final_net_pnl), 4)
            if final_net_pnl is not None
            else round(self.realized_pnl_usdt + self.unrealized_pnl_usdt, 4)
        )
        init_cap = max(0.01, float(self.plan.initial_capital_usdt))
        final_eq = round(init_cap + net_pnl, 4)
        roi_pct = round((net_pnl / init_cap) * 100.0, 2)
        mdd_7d = self.compute_mdd_7d_pct()
        opened_iso = datetime.datetime.fromtimestamp(
            self.start_timestamp_utc, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        closed_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        conn = self._get_db_connection()
        try:
            cur = conn.execute(
                """
                INSERT INTO mark2_bot_sessions (
                    opened_at, closed_at, symbol, direction, leverage,
                    initial_capital_usdt, final_equity_usdt, net_pnl_usdt,
                    roi_pct, total_crosses, trailing_shifts, mdd_7d_pct,
                    status, close_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    opened_iso,
                    closed_iso,
                    self.plan.symbol,
                    self.plan.direction.value,
                    int(self.plan.leverage),
                    round(init_cap, 4),
                    final_eq,
                    net_pnl,
                    roi_pct,
                    int(self.completed_crosses),
                    int(self.trailing_shifts_count),
                    round(mdd_7d, 4),
                    status,
                    close_reason,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            if self._memory_conn is None:
                conn.close()


# ============================================================================
# 4. ADAPTADOR ASÍNCRONO BLINDADO PARA BINANCE FUTURES USDⓈ-M (LIVE / SANDBOX)
# ============================================================================

class BinanceMark2LiveGridBroker(IGridBrokerExecutionPort):
    """
    Adaptador de Producción de MARK-2 hacia Binance Futures USDⓈ-M:
    Conecta a la API oficial de Binance para sincronizar tiempo, consultar balances,
    tasas de comisión (/fapi/v1/commissionRate), riesgo de posición (/fapi/v2/positionRisk),
    desplegar órdenes por lotes (/fapi/v1/batchOrders) y auditar el PnL real (/fapi/v1/userTrades).
    """

    def __init__(self, env_path: str = ".env"):
        self.cfg = EnvConfigLoader.load(env_path)
        self.env_mode = self.get_str("PROSPERITY_ENV", "SANDBOX").upper()
        self.dry_run = self.get_str("DRY_RUN", "FALSE").upper() == "TRUE"
        self._api_key = SecretStr(self.get_str("BINANCE_API_KEY", ""))
        self._api_secret = SecretStr(self.get_str("BINANCE_API_SECRET", ""))
        raw_db_path = self.get_str("DATABASE_PATH", "prosperity_mark2_journal.db")
        self.db_path = (
            raw_db_path
            if os.path.isabs(raw_db_path) or raw_db_path == ":memory:"
            else os.path.join(MARK2_DIR, raw_db_path)
        )

        if not self._api_key or not self._api_secret:
            logger.warning(
                "⚠️ [M5 Seguridad] API Keys vacías o de plantilla en .env -> Activando modo DRY_RUN=TRUE."
            )
            self.dry_run = True

        self.base_url = (
            "https://testnet.binancefuture.com"
            if self.env_mode == "SANDBOX"
            else "https://fapi.binance.com"
        )
        self.time_offset_ms: int = 0
        self.last_time_sync_ts: float = 0.0
        self.multi_assets_mode: bool = False
        self.contract_rules: Dict[str, ExchangeContractRulePOD] = {}
        self.active_sessions: Dict[str, ActiveGridBotSession] = {}
        self.live_position_amt_cache: Dict[str, float] = {}
        self.active_brackets_placed: Dict[str, bool] = {}
        self.active_brackets_side: Dict[str, str] = {}
        self.active_algo_orders: Dict[str, List[Dict[str, Any]]] = {}
        self.algo_retry_after_ts: Dict[str, float] = {}
        self._algo_seq: int = 0
        self._session_http: Optional[Any] = None

    def get_str(self, key: str, default: str) -> str:
        return self.cfg.get(key, os.getenv(key, default))

    def get_float(self, key: str, default: float) -> float:
        try:
            return float(self.get_str(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def round_step(value: float, step: float) -> float:
        """Trunca matemáticamente al tickSize o stepSize exacto usando Decimal."""
        if step <= 0.0:
            return round(value, 4)
        val_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        rounded = (val_dec // step_dec) * step_dec
        return float(rounded.quantize(step_dec, rounding=ROUND_DOWN))

    @staticmethod
    def format_step(value: float, step: float) -> str:
        """Formatea precio o cantidad en notación decimal fija (nunca científica) para Binance."""
        if step <= 0.0:
            return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
        val_dec = Decimal(str(value))
        step_dec = Decimal(str(step)).normalize()
        rounded = ((val_dec // step_dec) * step_dec).quantize(step_dec, rounding=ROUND_DOWN)
        s = format(rounded, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"

    def build_valid_client_algo_id(self, prefix: str, symbol: str) -> str:
        r"""
        Genera un clientAlgoId único, determinista y validado contra el patrón oficial de Binance
        (^[\.A-Z\:/a-z0-9_-]{1,36}$), preservando íntegro el sufijo temporal y secuencial sin truncarlo.
        """
        self._algo_seq = (self._algo_seq + 1) % 10_000
        clean_prefix = re.sub(r"[^A-Za-z0-9_-]", "", prefix)[:8] or "M2A"
        clean_sym = re.sub(r"[^A-Za-z0-9]", "", symbol.upper())[:10]
        ts_tail = str((int(time.time() * 1000) + self.time_offset_ms) % 10_000_000_000)
        candidate = f"{clean_prefix}_{clean_sym}_{ts_tail}{self._algo_seq:02d}"[:32]
        if not ALGO_CLIENT_ID_PATTERN.match(candidate):
            candidate = f"M2A_{ts_tail}_{self._algo_seq:02d}"[:32]
        return candidate

    def clamp_algo_time_range(
        self,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Ajusta estrictamente startTime y endTime a las reglas del motor AlgoOrder de Binance:
        1. Antigüedad máxima menor a 90 días (89d efectivos).
        2. Ventana máxima menor a 7 días entre startTime y endTime (evita error -4509 por Time Range).
        """
        now_srv_ms = int(time.time() * 1000) + self.time_offset_ms
        min_allowed_ms = now_srv_ms - MAX_ALGO_HISTORY_AGE_MS
        resolved_end = min(int(end_time_ms), now_srv_ms) if end_time_ms is not None else now_srv_ms
        resolved_end = max(min_allowed_ms + 60_000, resolved_end)
        if start_time_ms is None:
            resolved_start = max(min_allowed_ms, resolved_end - MAX_ALGO_QUERY_WINDOW_MS)
        else:
            resolved_start = max(min_allowed_ms, int(start_time_ms))
            if resolved_start > resolved_end:
                resolved_start = max(min_allowed_ms, resolved_end - 3_600_000)
            if (resolved_end - resolved_start) > MAX_ALGO_QUERY_WINDOW_MS:
                resolved_start = resolved_end - MAX_ALGO_QUERY_WINDOW_MS
        return resolved_start, resolved_end

    def _sign_params(self, params: Dict[str, Any]) -> str:
        params["timestamp"] = int(time.time() * 1000) + self.time_offset_ms
        params.setdefault("recvWindow", 10000)
        query = urlencode(params, quote_via=quote)
        sig = hmac.new(
            self._api_secret.get_secret_value().encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={sig}"

    async def _signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Ejecuta peticiones HTTP firmadas con sincronización periódica de reloj y captura tipada."""
        if self.dry_run:
            return {
                "dry_run": True,
                "status": "FILLED",
                "algoStatus": "NEW",
                "orderId": int(time.time() * 1000) % 100_000_000,
                "algoId": int(time.time() * 1000) % 100_000_000,
                "clientAlgoId": (params or {}).get("clientAlgoId", "DRY_ALGO_ID"),
                "endpoint": endpoint,
            }

        http_sess = session or self._session_http
        if http_sess is None:
            raise RuntimeError("Sesión aiohttp no inicializada en BinanceMark2LiveGridBroker.")

        # Re-sincronización preventiva de reloj cada 180s para evitar deriva NTP/timestamp
        if (time.time() - self.last_time_sync_ts) > 180.0 and endpoint != "/fapi/v1/time":
            await self.sync_server_time(http_sess, quiet=True)

        for attempt in range(1, 4):
            payload = dict(params) if params else {}
            signed_query = self._sign_params(payload)
            raw_url = f"{self.base_url}{endpoint}?{signed_query}"
            target_url = yarl.URL(raw_url, encoded=True) if yarl is not None else raw_url
            headers = {"X-MBX-APIKEY": self._api_key.get_secret_value()}

            async with http_sess.request(method, target_url, headers=headers, timeout=8) as resp:
                if resp.status in (429, 418):
                    wait_s = int(resp.headers.get("Retry-After", "5"))
                    logger.warning(
                        "⏳ [M5 Anti-Ban] HTTP %d en %s. Pausando %ds...",
                        resp.status,
                        endpoint,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue

                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    err_code = int(data.get("code", resp.status)) if isinstance(data, dict) else resp.status
                    err_msg = data.get("msg", "Error desconocido") if isinstance(data, dict) else str(data)
                    if err_code == -1021 and attempt < 3:
                        await self.sync_server_time(http_sess)
                        continue
                    if err_code == -2015:
                        logger.error(
                            "🚫 [M5 Error de Permisos/IP en Binance (-2015)]: %s\n"
                            "   👉 Solución en Binance -> Gestión de API (API Management):\n"
                            "      1. En 'Restricciones de acceso de IP', agrega tu IP actual que muestra el mensaje.\n"
                            "      2. Marca la casilla 'Habilitar Futuros' (Enable Futures) y guarda los cambios.\n"
                            "      3. Si tus llaves son de Testnet, cambia PROSPERITY_ENV=SANDBOX en .env.",
                            err_msg,
                        )
                    raise BinanceApiError(err_code, endpoint, err_msg)
                return data
        raise RuntimeError(f"Fallo definitivo tras 3 intentos en {endpoint}")

    async def sync_server_time(self, session: Any, quiet: bool = False) -> None:
        """Sincroniza el offset de reloj local contra /fapi/v1/time (mitigación de deriva NTP)."""
        self._session_http = session
        try:
            t0_ms = int(time.time() * 1000)
            async with session.get(f"{self.base_url}/fapi/v1/time", timeout=6) as resp:
                if resp.status == 200:
                    t1_ms = int(time.time() * 1000)
                    srv = await resp.json()
                    midpoint_local_ms = (t0_ms + t1_ms) // 2
                    self.time_offset_ms = int(srv["serverTime"]) - midpoint_local_ms
                    self.last_time_sync_ts = time.time()
                    if not quiet:
                        logger.info(
                            "⏱️ [M5 Reloj] Offset sincronizado con Binance: %+d ms (RTT=%d ms)",
                            self.time_offset_ms,
                            max(0, t1_ms - t0_ms),
                        )
        except Exception as exc:
            logger.debug("Aviso sincronizando reloj con Binance: %s", exc)

    async def sync_multi_assets_mode(self, session: Optional[Any] = None) -> bool:
        """
        Verifica el estado de Multi-Assets Mode (/fapi/v1/multiAssetsMargin) en Binance:
        Si está activo e impide margen ISOLATED o genera restricciones en órdenes de futuros/AlgoOrder,
        intenta cambiar a Single-Asset Mode cuando no hay posiciones cruzadas incompatibles; si no es
        posible cambiarlo en caliente, activa compatibilidad de margen cruzado sin abortar el sistema.
        """
        if self.dry_run:
            self.multi_assets_mode = False
            return False
        try:
            res = await self._signed_request(
                "GET", "/fapi/v1/multiAssetsMargin", session=session
            )
            is_multi = bool(res.get("multiAssetsMargin", False)) if isinstance(res, dict) else False
            if is_multi:
                try:
                    await self._signed_request(
                        "POST",
                        "/fapi/v1/multiAssetsMargin",
                        {"multiAssetsMargin": "false"},
                        session=session,
                    )
                    self.multi_assets_mode = False
                    logger.info(
                        "⚙️ [M5 Margen] Multi-Assets Mode desactivado automáticamente -> Single-Asset (ISOLATED habilitado)."
                    )
                except BinanceApiError as exc:
                    self.multi_assets_mode = True
                    logger.warning(
                        "⚠️ [M5 Multi-Assets Mode Activo (%d)] No se pudo cambiar a Single-Asset (%s). "
                        "Se operará respetando margen cruzado actual para evitar rechazos -4168/-4509.",
                        exc.code,
                        exc.msg,
                    )
            else:
                self.multi_assets_mode = False
        except Exception as exc:
            logger.debug("Aviso consultando multiAssetsMargin: %s", exc)
        return self.multi_assets_mode

    async def fetch_commission_rates(
        self, symbol: str = "BTCUSDT", session: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Extrae las tasas reales de comisión Maker y Taker de la cuenta en Binance
        mediante /fapi/v1/commissionRate.
        """
        default_maker = self.get_float("MAKER_FEE_RATE", 0.00018)
        default_taker = self.get_float("TAKER_FEE_RATE", 0.00045)
        if not self._api_key or not self._api_secret or self.dry_run:
            return default_maker, default_taker
        try:
            res = await self._signed_request(
                "GET", "/fapi/v1/commissionRate", {"symbol": symbol}, session=session
            )
            if isinstance(res, dict) and "makerCommissionRate" in res:
                return (
                    float(res["makerCommissionRate"]),
                    float(res.get("takerCommissionRate", default_taker)),
                )
        except Exception:
            pass
        return default_maker, default_taker

    async def fetch_usdt_wallet_balance(self, session: Optional[Any] = None) -> float:
        """Consulta el balance total USDT en la billetera de futuros USDⓈ-M."""
        if self.dry_run:
            return self.get_float("INITIAL_CAPITAL_USDT", 18.0)
        balances = await self._signed_request("GET", "/fapi/v2/balance", session=session)
        for b in balances:
            if b.get("asset") == "USDT":
                return float(b.get("balance", 18.0))
        return self.get_float("INITIAL_CAPITAL_USDT", 18.0)

    async def fetch_live_positions_map(
        self, session: Optional[Any] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Consulta /fapi/v2/positionRisk en vivo para extraer positionAmt, entryPrice y
        unRealizedProfit reales reportados por el motor de Binance, limpiando el caché local
        de brackets/AlgoOrders cuando la posición llega a 0 o invierte su dirección.
        """
        if self.dry_run:
            return {}
        try:
            risk_list = await self._signed_request("GET", "/fapi/v2/positionRisk", session=session)
            out: Dict[str, Dict[str, float]] = {}
            if isinstance(risk_list, list):
                for item in risk_list:
                    sym = item.get("symbol", "")
                    if sym:
                        pos_amt = float(item.get("positionAmt", 0.0))
                        self.live_position_amt_cache[sym] = pos_amt
                        if abs(pos_amt) <= 1e-12:
                            self.active_brackets_placed.pop(sym, None)
                            self.active_brackets_side.pop(sym, None)
                            self.active_algo_orders.pop(sym, None)
                        else:
                            curr_side = "LONG" if pos_amt > 0.0 else "SHORT"
                            prev_side = self.active_brackets_side.get(sym)
                            if prev_side is not None and prev_side != curr_side:
                                self.active_brackets_placed.pop(sym, None)
                                self.active_brackets_side.pop(sym, None)
                        out[sym] = {
                            "positionAmt": pos_amt,
                            "entryPrice": float(item.get("entryPrice", 0.0)),
                            "markPrice": float(item.get("markPrice", 0.0)),
                            "unRealizedProfit": float(item.get("unRealizedProfit", 0.0)),
                        }
            return out
        except Exception:
            return {}

    async def configure_isolated_and_leverage(
        self, symbol: str, leverage: int, session: Optional[Any] = None
    ) -> bool:
        """
        Configura margen ISOLATED (o respeta Multi-Assets Mode si no puede desactivarse)
        y ajusta el apalancamiento nominal hasta el máximo permitido por el contrato.
        """
        rule = self.contract_rules.get(symbol)
        contract_max = rule.max_leverage if rule is not None else 125
        requested_lev = min(contract_max, max(1, leverage))
        if self.dry_run:
            return True

        if not self.multi_assets_mode:
            try:
                await self._signed_request(
                    "POST",
                    "/fapi/v1/marginType",
                    {"symbol": symbol, "marginType": "ISOLATED"},
                    session=session,
                )
            except RuntimeError as exc:
                err_txt = str(exc)
                # -4046: No need to change margin type | -4168: Multi-Assets mode active | -4067: Open orders exist
                if "-4046" in err_txt:
                    pass
                elif "-4168" in err_txt:
                    self.multi_assets_mode = True
                    logger.warning(
                        "⚠️ [M5 Margen %s] Cuenta en Multi-Assets Mode (-4168); se mantiene margen cruzado activo.",
                        symbol,
                    )
                elif "-4067" in err_txt or "-4509" in err_txt:
                    logger.info(
                        "ℹ️ [M5 Margen %s] Se conserva tipo de margen actual por posición/orden activa (%s).",
                        symbol,
                        exc,
                    )
                else:
                    raise

        fallback_ladder = [requested_lev] + [
            lv for lv in (100, 75, 50, 25, 20, 15, 10, 5) if lv < requested_lev
        ]
        for cand_lev in fallback_ladder:
            try:
                await self._signed_request(
                    "POST",
                    "/fapi/v1/leverage",
                    {"symbol": symbol, "leverage": cand_lev},
                    session=session,
                )
                return True
            except RuntimeError as exc:
                if "-4028" in str(exc) and cand_lev != fallback_ladder[-1]:
                    continue
                raise
        return True

    async def open_initial_base_position(
        self,
        symbol: str,
        direction: GridDirection,
        base_qty: float,
        rules: ExchangeContractRulePOD,
        session: Optional[Any] = None,
    ) -> float:
        """
        Implementa 'Abrir una posición con la creación: Activado':
        Si el Grid es LONG o SHORT, abre el inventario base inicial a mercado antes de colocar la malla.
        """
        if direction == GridDirection.NEUTRAL or base_qty <= 0.0:
            return 0.0

        q_qty = self.round_step(base_qty, rules.step_size)
        if q_qty < rules.min_qty:
            return 0.0
        q_str = self.format_step(q_qty, rules.step_size)

        side = "BUY" if direction == GridDirection.LONG else "SELL"
        client_oid = self.build_valid_client_algo_id("M2BASE", symbol)
        resp = await self._signed_request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": q_str,
                "newOrderRespType": "RESULT",
                "newClientOrderId": client_oid,
            },
            session=session,
        )
        avg_p = float(resp.get("avgPrice", 0.0))
        signed_qty = q_qty if direction == GridDirection.LONG else -q_qty
        self.live_position_amt_cache[symbol] = (
            self.live_position_amt_cache.get(symbol, 0.0) + signed_qty
        )
        logger.info(
            "🚀 [M5 POSICIÓN INICIAL GRID %s] %s %s Qty=%s (avgPrice=%.4f | clientOrderId=%s)",
            symbol,
            direction.value,
            side,
            q_str,
            avg_p,
            client_oid,
        )
        return avg_p

    async def fetch_symbol_commission_rates(
        self, symbol: str = "BTCUSDT", session: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Consulta en vivo GET /fapi/v1/commissionRate para extraer exactamente
        `makerCommissionRate` y `takerCommissionRate` reales de la cuenta en Binance.
        """
        default_maker = self.get_float("MAKER_FEE_RATE", 0.00020)
        default_taker = self.get_float("TAKER_FEE_RATE", 0.00050)
        if self.dry_run:
            return default_maker, default_taker
        try:
            resp = await self._signed_request(
                "GET",
                "/fapi/v1/commissionRate",
                {"symbol": symbol},
                session=session,
            )
            if isinstance(resp, dict):
                mk = float(resp.get("makerCommissionRate", default_maker))
                tk = float(resp.get("takerCommissionRate", default_taker))
                return max(0.0, mk), max(0.0, tk)
        except Exception as exc:
            logger.debug("Aviso consultando /fapi/v1/commissionRate en %s: %s", symbol, exc)
        return default_maker, default_taker

    async def fetch_open_regular_orders(
        self, symbol: str, session: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Consulta en vivo GET /fapi/v1/openOrders para verificar qué órdenes LIMIT GTX
        continúan abiertas en el libro de Binance antes de asumir un fill.
        """
        if self.dry_run:
            return []
        try:
            resp = await self._signed_request(
                "GET",
                "/fapi/v1/openOrders",
                {"symbol": symbol},
                session=session,
            )
            if isinstance(resp, list):
                return [o for o in resp if isinstance(o, dict)]
        except Exception as exc:
            logger.debug("Aviso consultando /fapi/v1/openOrders en %s: %s", symbol, exc)
        return []

    async def fetch_recent_user_trades(
        self,
        symbol: str,
        start_time_ms: Optional[int] = None,
        limit: int = 50,
        session: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Consulta GET /fapi/v1/userTrades para auditar ejecuciones reales (precio, comisión exacta
        y realizedPnl) directamente desde el motor de emparejamiento de Binance.
        """
        if self.dry_run:
            return []
        params: Dict[str, Any] = {"symbol": symbol, "limit": max(1, min(1000, int(limit)))}
        if start_time_ms is not None and start_time_ms > 0:
            params["startTime"] = int(start_time_ms)
        try:
            resp = await self._signed_request(
                "GET",
                "/fapi/v1/userTrades",
                params,
                session=session,
            )
            if isinstance(resp, list):
                return [t for t in resp if isinstance(t, dict)]
        except Exception as exc:
            logger.debug("Aviso consultando /fapi/v1/userTrades en %s: %s", symbol, exc)
        return []

    async def fetch_open_algo_orders(
        self, symbol: str, session: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Consulta en vivo GET /fapi/v1/openAlgoOrders para sincronizar el caché activo del servidor
        y descartar identificadores algoId / clientAlgoId ya expirados, ejecutados o cancelados.
        """
        if self.dry_run:
            return list(self.active_algo_orders.get(symbol, []))
        try:
            resp = await self._signed_request(
                "GET",
                "/fapi/v1/openAlgoOrders",
                {"symbol": symbol},
                session=session,
            )
            orders_raw: List[Dict[str, Any]] = []
            if isinstance(resp, list):
                orders_raw = [o for o in resp if isinstance(o, dict)]
            elif isinstance(resp, dict) and isinstance(resp.get("orders"), list):
                orders_raw = [o for o in resp["orders"] if isinstance(o, dict)]

            live_filtered = [
                o
                for o in orders_raw
                if str(o.get("symbol", symbol)).upper() == symbol.upper()
                and str(o.get("algoStatus", "NEW")).upper() in ("NEW", "WORKING")
            ]
            if live_filtered:
                self.active_algo_orders[symbol] = live_filtered
            else:
                self.active_algo_orders.pop(symbol, None)
            return live_filtered
        except BinanceApiError as exc:
            if exc.code in (-4509, -2013, -2011) or "404" in str(exc):
                self.active_algo_orders.pop(symbol, None)
                return []
            logger.debug("Aviso consultando openAlgoOrders en %s: %s", symbol, exc)
            return []
        except Exception as exc:
            logger.debug("Aviso consultando openAlgoOrders en %s: %s", symbol, exc)
            return []

    async def query_algo_order(
        self,
        symbol: str,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Consulta puntual de una orden algorítmica (GET /fapi/v1/algoOrder) validando previamente
        algoId / clientAlgoId y capturando -4509 si ya salió del caché activo del exchange.
        """
        if algo_id is None and not client_algo_id:
            return None
        query_params: Dict[str, Any] = {"symbol": symbol}
        if algo_id is not None:
            query_params["algoId"] = int(algo_id)
        elif client_algo_id:
            query_params["clientAlgoId"] = str(client_algo_id)[:36]

        logger.debug(
            "🔎 [M5 AlgoOrder Query Pre-Flight %s] algoId=%s | clientAlgoId=%s",
            symbol,
            algo_id,
            client_algo_id,
        )
        if self.dry_run:
            return {"symbol": symbol, "algoId": algo_id, "clientAlgoId": client_algo_id, "algoStatus": "NEW"}
        try:
            res = await self._signed_request(
                "GET", "/fapi/v1/algoOrder", query_params, session=session
            )
            return res if isinstance(res, dict) else None
        except BinanceApiError as exc:
            if exc.code in (-4509, -2013):
                # Limpiar del caché local si ya expiró o fue purgada en Binance
                cached = self.active_algo_orders.get(symbol, [])
                self.active_algo_orders[symbol] = [
                    o
                    for o in cached
                    if o.get("algoId") != algo_id and o.get("clientAlgoId") != client_algo_id
                ]
                if not self.active_algo_orders.get(symbol):
                    self.active_algo_orders.pop(symbol, None)
                logger.info(
                    "ℹ️ [M5 AlgoOrder %s] Orden fuera de caché activo (%d: %s). Registro local depurado.",
                    symbol,
                    exc.code,
                    exc.msg,
                )
                return None
            raise

    async def query_historical_algo_orders(
        self,
        symbol: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 50,
        session: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Consulta histórica GET /fapi/v1/allAlgoOrders forzando estrictamente ventana < 7 días
        y antigüedad < 90 días para prevenir rechazos -4509 por Time Range.
        """
        if self.dry_run:
            return []
        safe_start, safe_end = self.clamp_algo_time_range(start_time_ms, end_time_ms)
        params: Dict[str, Any] = {
            "symbol": symbol,
            "startTime": safe_start,
            "endTime": safe_end,
            "limit": max(1, min(100, int(limit))),
        }
        try:
            res = await self._signed_request(
                "GET", "/fapi/v1/allAlgoOrders", params, session=session
            )
            if isinstance(res, list):
                return [o for o in res if isinstance(o, dict)]
            if isinstance(res, dict) and isinstance(res.get("orders"), list):
                return [o for o in res["orders"] if isinstance(o, dict)]
            return []
        except BinanceApiError as exc:
            if exc.code == -4509:
                logger.info(
                    "ℹ️ [M5 Historial AlgoOrder %s] Sin registros en ventana <7d (-4509 controlado).",
                    symbol,
                )
                return []
            raise

    async def cancel_algo_order_by_id(
        self,
        symbol: str,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> bool:
        """
        Cancela una orden algorítmica específica (DELETE /fapi/v1/algoOrder) únicamente tras
        validar su identificador contra el servidor y capturando -4509/-2011 sin desconexión.
        """
        if algo_id is None and not client_algo_id:
            return False
        if self.dry_run:
            return True

        req_params: Dict[str, Any] = {"symbol": symbol}
        if algo_id is not None:
            req_params["algoId"] = int(algo_id)
        else:
            req_params["clientAlgoId"] = str(client_algo_id)[:36]

        logger.info(
            "🔎 [M5 AlgoOrder Cancel Pre-Flight %s] Validando cancelación -> algoId=%s | clientAlgoId=%s",
            symbol,
            algo_id,
            client_algo_id,
        )
        try:
            await self._signed_request(
                "DELETE", "/fapi/v1/algoOrder", req_params, session=session
            )
            return True
        except BinanceApiError as exc:
            if exc.code in (-4509, -2011, -2013):
                logger.info(
                    "ℹ️ [M5 AlgoOrder Cancel %s] Orden algoId=%s / clientAlgoId=%s ya no estaba activa (%d).",
                    symbol,
                    algo_id,
                    client_algo_id,
                    exc.code,
                )
                return False
            logger.warning("⚠️ [M5 AlgoOrder Cancel %s]: %s", symbol, exc)
            return False

    async def cancel_all_symbol_orders(
        self, symbol: str, include_algo: bool = True, session: Optional[Any] = None
    ) -> None:
        """
        Cancela todas las órdenes abiertas del símbolo tanto en el libro estándar
        (/fapi/v1/allOpenOrders) como en el motor Algo/Condicional (/fapi/v1/algoOpenOrders),
        verificando primero órdenes Algo activas reales para no invocar cancelaciones sobre IDs inexistentes.
        """
        if include_algo:
            self.active_brackets_placed.pop(symbol, None)
            self.active_brackets_side.pop(symbol, None)
            self.active_algo_orders.pop(symbol, None)
        if self.dry_run:
            return

        try:
            await self._signed_request(
                "DELETE",
                "/fapi/v1/allOpenOrders",
                {"symbol": symbol},
                session=session,
            )
        except BinanceApiError as exc:
            if exc.code not in (-2011, -4509):
                logger.debug("Aviso en allOpenOrders %s: %s", symbol, exc)
        except Exception as exc:
            logger.debug("Aviso en allOpenOrders %s: %s", symbol, exc)

        if include_algo:
            open_algos = await self.fetch_open_algo_orders(symbol, session=session)
            if not open_algos:
                return
            algo_ids_summary = [
                f"{o.get('algoId')}:{o.get('clientAlgoId')}" for o in open_algos
            ]
            logger.info(
                "🧹 [M5 AlgoOrder Purge Pre-Flight %s] Cancelando %d AlgoOrders activas verificadas: %s",
                symbol,
                len(open_algos),
                ", ".join(algo_ids_summary[:6]),
            )
            try:
                await self._signed_request(
                    "DELETE",
                    "/fapi/v1/algoOpenOrders",
                    {"symbol": symbol},
                    session=session,
                )
            except BinanceApiError as exc:
                if exc.code in (-4509, -2011):
                    # Respaldo individual por algoId exacto generado por el exchange si el lote reportó -4509
                    for ao in open_algos:
                        aid = ao.get("algoId")
                        cid = ao.get("clientAlgoId")
                        if aid is not None or cid:
                            await self.cancel_algo_order_by_id(
                                symbol=symbol,
                                algo_id=int(aid) if aid is not None else None,
                                client_algo_id=str(cid) if cid else None,
                                session=session,
                            )
                else:
                    logger.warning("⚠️ [M5 Purga AlgoOpenOrders %s]: %s", symbol, exc)
            except Exception as exc:
                logger.debug("Aviso purgando algoOpenOrders en %s: %s", symbol, exc)
            finally:
                self.active_algo_orders.pop(symbol, None)

    async def place_conditional_protection_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price_str: str,
        client_id: str,
        working_type: str = "MARK_PRICE",
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Envía una orden condicional (STOP_MARKET o TAKE_PROFIT_MARKET con closePosition='true')
        validando y registrando en log el clientAlgoId antes del envío y almacenando el algoId
        exacto devuelto por Binance para trazabilidad libre de -4509.
        """
        clean_id = re.sub(r"[^\.A-Z\:/a-z0-9_-]", "", client_id)[:32]
        if not clean_id or not ALGO_CLIENT_ID_PATTERN.match(clean_id):
            clean_id = self.build_valid_client_algo_id("M2COND", symbol)

        algo_payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "algoType": "CONDITIONAL",
            "type": order_type.upper(),
            "triggerPrice": trigger_price_str,
            "closePosition": "true",
            "workingType": working_type,
            "clientAlgoId": clean_id,
        }
        legacy_payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "stopPrice": trigger_price_str,
            "closePosition": "true",
            "workingType": working_type,
            "newClientOrderId": clean_id,
        }

        logger.debug(
            "🔎 [M5 AlgoOrder Submit Pre-Flight %s] type=%s | side=%s | triggerPrice=%s | clientAlgoId=%s",
            symbol,
            order_type.upper(),
            side.upper(),
            trigger_price_str,
            clean_id,
        )

        resp: Dict[str, Any]
        if self.env_mode == "SANDBOX":
            try:
                resp = await self._signed_request(
                    "POST", "/fapi/v1/order", legacy_payload, session=session
                )
            except RuntimeError as exc:
                if "-4120" in str(exc):
                    resp = await self._signed_request(
                        "POST", "/fapi/v1/algoOrder", algo_payload, session=session
                    )
                else:
                    raise
        else:
            try:
                resp = await self._signed_request(
                    "POST", "/fapi/v1/algoOrder", algo_payload, session=session
                )
            except RuntimeError as exc:
                err_txt = str(exc)
                if "404" in err_txt or "-1102" in err_txt:
                    resp = await self._signed_request(
                        "POST", "/fapi/v1/order", legacy_payload, session=session
                    )
                else:
                    raise

        if isinstance(resp, dict):
            srv_algo_id = resp.get("algoId") or resp.get("orderId")
            srv_client_id = resp.get("clientAlgoId") or resp.get("clientOrderId") or clean_id
            tracked = {
                "symbol": symbol,
                "algoId": srv_algo_id,
                "clientAlgoId": srv_client_id,
                "type": order_type.upper(),
                "side": side.upper(),
                "triggerPrice": trigger_price_str,
                "algoStatus": resp.get("algoStatus", resp.get("status", "NEW")),
                "created_ts": time.time(),
            }
            self.active_algo_orders.setdefault(symbol, []).append(tracked)
            logger.debug(
                "✅ [M5 AlgoOrder Confirmada %s] %s @ %s -> Exchange algoId=%s | clientAlgoId=%s",
                symbol,
                order_type.upper(),
                trigger_price_str,
                srv_algo_id,
                srv_client_id,
            )
        return resp

    @staticmethod
    def build_grid_limit_order_payload(
        symbol: str,
        side: str,
        qty_str: str,
        price_str: str,
        current_position_amt: float,
        order_qty: float,
        time_in_force: str = "GTX",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Construye el payload de una orden LIMIT de cuadrícula aplicando la comprobación
        de inventario pre-vuelo en memoria:
        - Antes de marcar reduceOnly='true', verifica que current_position_amt != 0 y que
          la orden reduzca la posición existente con abs(current_position_amt) >= order_qty.
        - Si current_position_amt == 0 (o no reduce inventario existente), se registra como
          orden límite de rejilla normal (GTX / Post-Only) con reduceOnly='false'.
        """
        side_up = side.upper()
        is_closing = (current_position_amt > 0 and side_up == "SELL") or (
            current_position_amt < 0 and side_up == "BUY"
        )
        order_payload: Dict[str, str] = {
            "symbol": symbol,
            "side": side_up,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": qty_str,
            "price": price_str,
            # Solo activar reduceOnly si realmente reduce una posición existente:
            "reduceOnly": "true"
            if is_closing and abs(current_position_amt) >= order_qty - 1e-12
            else "false",
        }
        if client_order_id:
            order_payload["newClientOrderId"] = client_order_id[:32]
        return order_payload

    async def sync_active_limit_window(
        self,
        symbol: str,
        buy_levels: Tuple[float, ...],
        sell_levels: Tuple[float, ...],
        qty: float,
        rules: Optional[ExchangeContractRulePOD] = None,
        post_only_gtx: bool = True,
        cancel_existing: bool = False,
        current_position_amt: Optional[float] = None,
        session: Optional[Any] = None,
    ) -> int:
        """
        Despliega la ventana activa de órdenes límite en el libro de Binance usando
        /fapi/v1/batchOrders (lotes de hasta 5 órdenes por llamada HTTP) con comprobación
        de inventario pre-vuelo en memoria para cada nivel.
        """
        if cancel_existing and not self.dry_run:
            await self.cancel_all_symbol_orders(symbol, include_algo=False, session=session)

        r = rules or self.contract_rules.get(
            symbol, ExchangeContractRulePOD(symbol, 0.0001, 0.1, 0.1, 5.0, 200)
        )
        order_qty = max(r.min_qty, self.round_step(qty, r.step_size))
        qty_str = self.format_step(order_qty, r.step_size)
        tif = "GTX" if post_only_gtx else "GTC"
        ts_base = int(time.time() * 1000)

        pos_amt = (
            current_position_amt
            if current_position_amt is not None
            else self.live_position_amt_cache.get(symbol, 0.0)
        )
        self.live_position_amt_cache[symbol] = pos_amt
        rem_long_to_close = pos_amt if pos_amt > 0.0 else 0.0
        rem_short_to_close = abs(pos_amt) if pos_amt < 0.0 else 0.0

        order_payloads: List[Dict[str, str]] = []
        for idx, bp in enumerate(buy_levels):
            price_str = self.format_step(bp, r.tick_size)
            eff_pos_amt = -rem_short_to_close if rem_short_to_close > 0.0 else 0.0
            payload = self.build_grid_limit_order_payload(
                symbol=symbol,
                side="BUY",
                qty_str=qty_str,
                price_str=price_str,
                current_position_amt=eff_pos_amt,
                order_qty=order_qty,
                time_in_force=tif,
                client_order_id=f"M2_GB_{idx}_{ts_base}",
            )
            if payload["reduceOnly"] == "true":
                rem_short_to_close = max(0.0, rem_short_to_close - order_qty)
            order_payloads.append(payload)

        for idx, sp in enumerate(sell_levels):
            price_str = self.format_step(sp, r.tick_size)
            eff_pos_amt = rem_long_to_close if rem_long_to_close > 0.0 else 0.0
            payload = self.build_grid_limit_order_payload(
                symbol=symbol,
                side="SELL",
                qty_str=qty_str,
                price_str=price_str,
                current_position_amt=eff_pos_amt,
                order_qty=order_qty,
                time_in_force=tif,
                client_order_id=f"M2_GS_{idx}_{ts_base}",
            )
            if payload["reduceOnly"] == "true":
                rem_long_to_close = max(0.0, rem_long_to_close - order_qty)
            order_payloads.append(payload)

        placed_count = 0
        for i in range(0, len(order_payloads), 5):
            chunk = order_payloads[i : i + 5]
            if self.dry_run:
                placed_count += len(chunk)
                continue
            batch_json = json.dumps(chunk, separators=(",", ":"))
            resp_batch = await self._signed_request(
                "POST",
                "/fapi/v1/batchOrders",
                {"batchOrders": batch_json},
                session=session,
            )
            if isinstance(resp_batch, list):
                for ord_req, ord_res in zip(chunk, resp_batch):
                    if isinstance(ord_res, dict) and "orderId" in ord_res:
                        placed_count += 1
                    elif isinstance(ord_res, dict) and ord_res.get("code") == -5022:
                        # Si Post-Only (GTX) cruzaría el spread, recotizar 1 tick hacia el lado pasivo manteniendo GTX (0% Taker)
                        fallback_req = dict(ord_req)
                        raw_p = float(ord_req.get("price", 0.0))
                        side_req = str(ord_req.get("side", "BUY")).upper()
                        passive_p = (
                            max(rules.tick_size, raw_p - rules.tick_size)
                            if side_req == "BUY"
                            else (raw_p + rules.tick_size)
                        )
                        fallback_req["price"] = self.format_step(passive_p, rules.tick_size)
                        fallback_req["timeInForce"] = "GTX"
                        fallback_req["newClientOrderId"] = f"{ord_req['newClientOrderId']}_P"[:32]
                        try:
                            await self._signed_request("POST", "/fapi/v1/order", fallback_req, session=session)
                            placed_count += 1
                        except Exception as exc:
                            logger.warning("⚠️ [M5 Recotización Pasiva GTX %s]: %s", symbol, exc)
                    elif isinstance(ord_res, dict) and ord_res.get("code") == -2022:
                        # Si Binance rechaza reduceOnly por discrepancia de inventario en tránsito, reenviar como orden normal GTX
                        fallback_req = dict(ord_req)
                        fallback_req["reduceOnly"] = "false"
                        fallback_req["newClientOrderId"] = f"{ord_req['newClientOrderId']}_N"[:32]
                        try:
                            await self._signed_request("POST", "/fapi/v1/order", fallback_req, session=session)
                            placed_count += 1
                        except Exception as exc:
                            logger.warning("⚠️ [M5 Orden Individual Normal %s]: %s", symbol, exc)
                    elif isinstance(ord_res, dict) and "code" in ord_res:
                        logger.warning(
                            "⚠️ [M5 BatchOrder Rechazada en %s (%s @ %s)]: [%s] %s",
                            symbol,
                            ord_req.get("side"),
                            ord_req.get("price"),
                            ord_res.get("code"),
                            ord_res.get("msg"),
                        )
            else:
                placed_count += len(chunk)

        logger.info(
            "📊 [M5 VENTANA ACTIVA %s] %d órdenes LIMIT (%s) colocadas (%d BUY / %d SELL | Qty=%s | PosAmt=%.4f)",
            symbol,
            placed_count,
            tif,
            len(buy_levels),
            len(sell_levels),
            qty_str,
            pos_amt,
        )
        return placed_count

    async def place_single_counter_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        rules: ExchangeContractRulePOD,
        reduce_only: bool = False,
        current_position_amt: Optional[float] = None,
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Coloca inmediatamente la contra-orden límite cuando un escalón del Grid se llena,
        aplicando la comprobación de inventario pre-vuelo en memoria (position_amt != 0).
        """
        price_str = self.format_step(price, rules.tick_size)
        order_qty = max(rules.min_qty, self.round_step(qty, rules.step_size))
        qty_str = self.format_step(order_qty, rules.step_size)
        ts_id = int(time.time() * 1000)

        pos_amt = (
            current_position_amt
            if current_position_amt is not None
            else self.live_position_amt_cache.get(symbol, 0.0)
        )
        self.live_position_amt_cache[symbol] = pos_amt

        order_payload = self.build_grid_limit_order_payload(
            symbol=symbol,
            side=side,
            qty_str=qty_str,
            price_str=price_str,
            current_position_amt=pos_amt if reduce_only or abs(pos_amt) > 0.0 else 0.0,
            order_qty=order_qty,
            time_in_force="GTX",
            client_order_id=f"M2_FLIP_{side[:1]}_{ts_id}",
        )

        try:
            return await self._signed_request("POST", "/fapi/v1/order", order_payload, session=session)
        except RuntimeError as exc:
            err_str = str(exc)
            if "-5022" in err_str:
                side_up = side.upper()
                for tick_shift in (1, 2):
                    passive_price = (
                        max(rules.tick_size, price - tick_shift * rules.tick_size)
                        if side_up == "BUY"
                        else (price + tick_shift * rules.tick_size)
                    )
                    order_payload["price"] = self.format_step(passive_price, rules.tick_size)
                    order_payload["timeInForce"] = "GTX"
                    order_payload["newClientOrderId"] = f"M2_FLIP_{side[:1]}_{ts_id}_P{tick_shift}"[:32]
                    try:
                        return await self._signed_request("POST", "/fapi/v1/order", order_payload, session=session)
                    except RuntimeError as retry_exc:
                        if "-5022" in str(retry_exc) and tick_shift < 2:
                            continue
                        raise
            if "-2022" in err_str:
                order_payload["reduceOnly"] = "false"
                return await self._signed_request("POST", "/fapi/v1/order", order_payload, session=session)
            raise

    async def _fetch_verified_symbol_position_amt(
        self, symbol: str, retries_on_zero: int = 1, session: Optional[Any] = None
    ) -> float:
        """
        Consulta directamente /fapi/v2/positionRisk?symbol=... con micro-backoff opcional
        para absorber latencias de propagación entre el motor de matching y el motor AlgoOrder.
        """
        if self.dry_run:
            return float(self.live_position_amt_cache.get(symbol, 0.0))

        attempts = max(1, int(retries_on_zero))
        for idx in range(attempts):
            try:
                risk_data = await self._signed_request(
                    "GET", "/fapi/v2/positionRisk", {"symbol": symbol}, session=session
                )
                pos_val = 0.0
                if isinstance(risk_data, list):
                    for item in risk_data:
                        if str(item.get("symbol", "")).upper() == symbol.upper():
                            pos_val = float(item.get("positionAmt", 0.0))
                            break
                if abs(pos_val) > 1e-12:
                    return pos_val
            except Exception as exc:
                logger.debug("Aviso consultando positionRisk puntual en %s: %s", symbol, exc)
            if idx < attempts - 1:
                await asyncio.sleep(0.25)
        return 0.0

    @staticmethod
    def resolve_directional_bracket_prices(
        eff_direction: GridDirection,
        sl_price: float,
        tp_price: float,
        current_mark_price: float,
        tick_size: float,
        lower_price: float = 0.0,
        upper_price: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Orienta geométricamente SL y TP según la dirección efectiva REAL de la posición abierta:
        - LONG / NEUTRAL: SL por debajo del canal inferior y del markPrice; TP por encima del canal superior.
        - SHORT:          SL por encima del canal superior y del markPrice; TP por debajo del canal inferior.
        Evita que una posición SHORT nacida de un plan NEUTRAL/LONG invierta SL/TP y los colapse a markPrice ± 3 ticks.
        """
        low_bound = min(float(sl_price), float(tp_price))
        high_bound = max(float(sl_price), float(tp_price))

        if lower_price > 0.0 and upper_price > lower_price:
            low_span = max(tick_size * 10.0, lower_price - low_bound)
            high_span = max(tick_size * 10.0, high_bound - upper_price)
            low_bound = min(low_bound, max(tick_size, lower_price - low_span))
            high_bound = max(high_bound, upper_price + high_span)

        if eff_direction == GridDirection.SHORT:
            raw_sl = high_bound
            raw_tp = low_bound
        else:
            raw_sl = low_bound
            raw_tp = high_bound

        structural_guard = max(
            25.0 * tick_size,
            (current_mark_price * 0.005) if current_mark_price > 0.0 else (25.0 * tick_size),
        )
        safe_dist = max(5.0 * tick_size, current_mark_price * 0.0015) if current_mark_price > 0.0 else 5.0 * tick_size

        sl_clean = BinanceMark2LiveGridBroker.round_step(raw_sl, tick_size)
        tp_clean = BinanceMark2LiveGridBroker.round_step(raw_tp, tick_size)

        if current_mark_price > 0.0:
            if eff_direction in (GridDirection.LONG, GridDirection.NEUTRAL):
                if sl_clean >= current_mark_price - safe_dist:
                    sl_clean = BinanceMark2LiveGridBroker.round_step(
                        max(tick_size, current_mark_price - structural_guard), tick_size
                    )
                if tp_clean <= current_mark_price + safe_dist:
                    tp_clean = BinanceMark2LiveGridBroker.round_step(
                        current_mark_price + structural_guard, tick_size
                    )
            else:
                if sl_clean <= current_mark_price + safe_dist:
                    sl_clean = BinanceMark2LiveGridBroker.round_step(
                        current_mark_price + structural_guard, tick_size
                    )
                if tp_clean >= current_mark_price - safe_dist:
                    tp_clean = BinanceMark2LiveGridBroker.round_step(
                        max(tick_size, current_mark_price - structural_guard), tick_size
                    )

        return max(tick_size, sl_clean), max(tick_size, tp_clean)

    async def place_global_protection_brackets(
        self,
        symbol: str,
        direction: GridDirection,
        sl_price: float,
        tp_price: float,
        current_mark_price: float = 0.0,
        rules: Optional[ExchangeContractRulePOD] = None,
        current_position_amt: Optional[float] = None,
        order_filled: bool = False,
        verified_exchange_pos_amt: Optional[float] = None,
        lower_price: float = 0.0,
        upper_price: float = 0.0,
        session: Optional[Any] = None,
    ) -> bool:
        """
        Brackets asíncronos vinculados a Fills y blindados contra el error [-4509]:
        1. Confirma siempre en vivo en /fapi/v2/positionRisk (con micro-reintentos si acaba de llenarse
           una orden) que positionAmt != 0 antes de invocar /fapi/v1/algoOrder (TIF=GTE_GTC).
        2. Orienta correctamente SL y TP según el signo real de positionAmt (LONG vs SHORT).
        3. Sincroniza /fapi/v1/openAlgoOrders para reutilizar o depurar IDs activos sin órfanos.
        4. Captura específicamente BinanceApiError [-4509] con enfriamiento por símbolo.
        """
        now_s = time.time()
        if not self.dry_run and now_s < self.algo_retry_after_ts.get(symbol, 0.0):
            return False

        pos_amt = (
            current_position_amt
            if current_position_amt is not None
            else self.live_position_amt_cache.get(symbol, 0.0)
        )

        if not self.dry_run:
            # Siempre validar contra /fapi/v2/positionRisk cuando no haya confirmación fresca
            # o cuando acabe de ocurrir una ejecución de orden (evita caché obsoleto -> -4509).
            if verified_exchange_pos_amt is None or order_filled:
                pos_amt = await self._fetch_verified_symbol_position_amt(
                    symbol=symbol,
                    retries_on_zero=3 if order_filled else 1,
                    session=session,
                )
            else:
                pos_amt = float(verified_exchange_pos_amt)

        self.live_position_amt_cache[symbol] = pos_amt

        if abs(pos_amt) <= 1e-12 and (not self.dry_run or not order_filled):
            self.active_brackets_placed.pop(symbol, None)
            self.active_brackets_side.pop(symbol, None)
            self.algo_retry_after_ts[symbol] = now_s + 4.0
            logger.info(
                "⏳ [M5 Brackets Diferidos %s] Sin posición abierta en Binance (positionAmt=0). "
                "Los brackets SL/TP (TIF=GTE_GTC) se despacharán al confirmarse posición real != 0.",
                symbol,
            )
            return False

        eff_dir = direction
        if abs(pos_amt) > 1e-12:
            eff_dir = GridDirection.LONG if pos_amt > 0.0 else GridDirection.SHORT

        if self.active_brackets_placed.get(symbol, False):
            if self.active_brackets_side.get(symbol) == eff_dir.value:
                return True
            # Si la posición invirtió signo (LONG <-> SHORT), purgar AlgoOrders previas antes de rearmar
            await self.cancel_all_symbol_orders(symbol, include_algo=True, session=session)

        r = rules or self.contract_rules.get(
            symbol, ExchangeContractRulePOD(symbol, 0.0001, 0.1, 0.1, 5.0, 200)
        )
        sl_clean, tp_clean = self.resolve_directional_bracket_prices(
            eff_direction=eff_dir,
            sl_price=sl_price,
            tp_price=tp_price,
            current_mark_price=current_mark_price,
            tick_size=r.tick_size,
            lower_price=lower_price,
            upper_price=upper_price,
        )

        sl_str = self.format_step(sl_clean, r.tick_size)
        tp_str = self.format_step(tp_clean, r.tick_size)
        exit_side = "BUY" if eff_dir == GridDirection.SHORT else "SELL"

        # Sincronizar órdenes Algo abiertas en el servidor para no duplicar ni acumular órfanos
        if not self.dry_run:
            existing_algos = await self.fetch_open_algo_orders(symbol, session=session)
            same_side_types = {
                str(o.get("orderType") or o.get("type") or "").upper()
                for o in existing_algos
                if str(o.get("side", "")).upper() == exit_side
            }
            if "STOP_MARKET" in same_side_types and "TAKE_PROFIT_MARKET" in same_side_types:
                self.active_brackets_placed[symbol] = True
                self.active_brackets_side[symbol] = eff_dir.value
                self.algo_retry_after_ts.pop(symbol, None)
                return True
            if existing_algos:
                for stale_ao in existing_algos:
                    await self.cancel_algo_order_by_id(
                        symbol=symbol,
                        algo_id=int(stale_ao["algoId"]) if stale_ao.get("algoId") is not None else None,
                        client_algo_id=str(stale_ao.get("clientAlgoId")) if stale_ao.get("clientAlgoId") else None,
                        session=session,
                    )
                self.active_algo_orders.pop(symbol, None)

        for algo_attempt in range(1, 3):
            sl_cid = self.build_valid_client_algo_id("M2GSL", symbol)
            tp_cid = self.build_valid_client_algo_id("M2GTP", symbol)
            placed_sl_resp: Optional[Dict[str, Any]] = None
            try:
                placed_sl_resp = await self.place_conditional_protection_order(
                    symbol=symbol,
                    side=exit_side,
                    order_type="STOP_MARKET",
                    trigger_price_str=sl_str,
                    client_id=sl_cid,
                    working_type="MARK_PRICE",
                    session=session,
                )
                placed_tp_resp = await self.place_conditional_protection_order(
                    symbol=symbol,
                    side=exit_side,
                    order_type="TAKE_PROFIT_MARKET",
                    trigger_price_str=tp_str,
                    client_id=tp_cid,
                    working_type="MARK_PRICE",
                    session=session,
                )
                sl_aid = placed_sl_resp.get("algoId") if isinstance(placed_sl_resp, dict) else None
                tp_aid = placed_tp_resp.get("algoId") if isinstance(placed_tp_resp, dict) else None
                self.active_brackets_placed[symbol] = True
                self.active_brackets_side[symbol] = eff_dir.value
                self.algo_retry_after_ts.pop(symbol, None)
                logger.info(
                    "🛡️ [M5 BRACKETS GLOBALES %s] SL=%s (algoId=%s) | TP=%s (algoId=%s) [MARK_PRICE | %s PosAmt=%.4f]",
                    symbol,
                    sl_str,
                    sl_aid,
                    tp_str,
                    tp_aid,
                    eff_dir.value,
                    pos_amt,
                )
                return True
            except RuntimeError as exc:
                if "-4509" in str(exc):
                    # Si la pata SL se creó pero la pata TP falló por -4509 (posición cerrada o en tránsito), limpiar pata huérfana
                    if isinstance(placed_sl_resp, dict) and not self.dry_run:
                        orphan_aid = placed_sl_resp.get("algoId")
                        if orphan_aid is not None:
                            await self.cancel_algo_order_by_id(
                                symbol=symbol, algo_id=int(orphan_aid), session=session
                            )
                    self.active_brackets_placed.pop(symbol, None)
                    self.active_brackets_side.pop(symbol, None)
                    self.active_algo_orders.pop(symbol, None)

                    if algo_attempt == 1 and not self.dry_run:
                        await asyncio.sleep(0.35)
                        rechecked_pos = await self._fetch_verified_symbol_position_amt(
                            symbol=symbol, retries_on_zero=1, session=session
                        )
                        self.live_position_amt_cache[symbol] = rechecked_pos
                        if abs(rechecked_pos) > 1e-12:
                            pos_amt = rechecked_pos
                            continue

                    self.algo_retry_after_ts[symbol] = time.time() + 6.0
                    logger.info(
                        "⏳ [M5 Brackets Diferidos %s] Posición aún en sincronización o cerrada en motor AlgoOrder (-4509). "
                        "Reintento controlado en >=6s al confirmarse positionAmt != 0; supervisión activa en M4.",
                        symbol,
                    )
                    return False
                raise
        return False

    async def atomic_close_and_purge_bot(
        self, symbol: str, reason: str = "TARGET_18_USD_REACHED", session: Optional[Any] = None
    ) -> float:
        """
        Purga Atómica Innegociable (Cerrar posiciones al detener = Activado):
        1. Cancela todas las órdenes límite (/fapi/v1/allOpenOrders) y condicionales (/fapi/v1/algoOpenOrders).
        2. Verifica si existe inventario vivo en /fapi/v2/positionRisk y lo cierra con reduceOnly='true'.
        3. Audita el PnL neto real desde /fapi/v1/userTrades.
        """
        await self.cancel_all_symbol_orders(symbol, include_algo=True, session=session)

        if not self.dry_run:
            risk_list = await self._signed_request("GET", "/fapi/v2/positionRisk", session=session)
            for item in risk_list:
                if item.get("symbol") == symbol:
                    pos_amt = float(item.get("positionAmt", 0.0))
                    if abs(pos_amt) > 0.0:
                        close_side = "SELL" if pos_amt > 0.0 else "BUY"
                        rule = self.contract_rules.get(symbol)
                        qty_str = (
                            self.format_step(abs(pos_amt), rule.step_size)
                            if rule is not None
                            else str(abs(pos_amt))
                        )
                        try:
                            await self._signed_request(
                                "POST",
                                "/fapi/v1/order",
                                {
                                    "symbol": symbol,
                                    "side": close_side,
                                    "type": "MARKET",
                                    "quantity": qty_str,
                                    "reduceOnly": "true",
                                },
                                session=session,
                            )
                        except BinanceApiError as exc:
                            if exc.code not in (-2022, -4509):
                                raise
                    break

        self.live_position_amt_cache[symbol] = 0.0
        self.active_brackets_placed.pop(symbol, None)
        self.active_brackets_side.pop(symbol, None)
        self.active_algo_orders.pop(symbol, None)
        self.algo_retry_after_ts.pop(symbol, None)
        audited_pnl = await self._audit_realized_pnl(symbol, session=session)
        logger.info(
            "🧹 [M5 PURGA ATÓMICA COMPLETADA EN %s] Motivo: %s | PnL Auditado: %+.4f USDT",
            symbol,
            reason,
            audited_pnl,
        )
        return audited_pnl

    async def _audit_realized_pnl(
        self, symbol: str, session: Optional[Any] = None
    ) -> float:
        """Suma el realizedPnl menos comisiones de los últimos fills en /fapi/v1/userTrades."""
        if self.dry_run:
            return 0.0
        try:
            trades = await self._signed_request(
                "GET",
                "/fapi/v1/userTrades",
                {"symbol": symbol, "limit": 50},
                session=session,
            )
            net_sum = 0.0
            for t in trades:
                net_sum += float(t.get("realizedPnl", 0.0)) - float(t.get("commission", 0.0))
            return round(net_sum, 4)
        except Exception:
            return 0.0


# ============================================================================
# 5. SUITE DE VERIFICACIÓN INDEPENDIENTE DEL MÓDULO 5
# ============================================================================

def _self_test_m5_module() -> None:
    """Verifica en RAM y SQLite el funcionamiento del Módulo 5."""
    from core_mark2_contracts import CompoundingStage, MarketSafetyTier

    dummy_rules = ExchangeContractRulePOD(
        symbol="SUIUSDT",
        tick_size=0.0001,
        step_size=0.1,
        min_qty=0.1,
        min_notional=5.0,
        max_num_orders=200,
    )
    dummy_plan = GeometricGridPlanPOD(
        symbol="SUIUSDT",
        safety_tier=MarketSafetyTier.TIER_4_VOLATILE_ALPHA,
        direction=GridDirection.LONG,
        compounding_stage=CompoundingStage.STAGE_1_IGNITION_6_TO_9,
        initial_capital_usdt=6.00,
        current_equity_usdt=6.00,
        target_equity_usdt=18.00,
        equity_ratchet_floor_usdt=4.68,
        leverage=12,
        lower_price=3.0000,
        upper_price=3.3600,
        total_virtual_grids=20,
        active_window_orders=8,
        geometric_step_pct=0.57,
        net_profit_per_cross_pct=0.534,
        net_profit_per_cross_usdt=0.0320,
        qty_per_line=1.9,
        notional_per_line_usdt=6.00,
        all_virtual_levels=(3.0000, 3.0171, 3.0343, 3.0516, 3.0690, 3.0865, 3.1041),
        active_buy_window=(3.0171, 3.0343, 3.0516),
        active_sell_window=(3.0865, 3.1041),
        tp_price=3.4500,
        tp_pnl_usdt=12.00,
        tp_roi_pct=200.0,
        sl_price=2.9200,
        sl_pnl_usdt=-1.32,
        sl_roi_pct=-22.0,
        est_daily_crosses=24.0,
        est_days_to_18_usd=15.6,
        trailing_up=True,
        trailing_down=False,
    )

    executor = Mark2SlidingWindowGridExecutor(
        plan=dummy_plan, rules=dummy_rules, journal_db_path=":memory:"
    )

    res_buy = executor.on_limit_order_filled("BUY", 3.0516)
    assert res_buy["new_order_side"] == "SELL"

    sell_target = float(res_buy["new_order_price"])
    res_sell = executor.on_limit_order_filled("SELL", sell_target)
    assert res_sell["new_order_side"] == "BUY"
    assert executor.completed_crosses == 1
    assert executor.realized_pnl_usdt > 0.0

    trail = executor.check_and_apply_trailing_grid(current_price=3.1500, quote_obi_l1=0.25)
    assert trail is not None and trail["trailing_action"] == "TRAILING_UP_EXECUTED"

    t_24h, t_total = executor.get_matched_trades_24h_and_total()
    assert t_24h == 1 and t_total == 1

    # Verificación de comprobación de inventario pre-vuelo (reduceOnly='true' vs 'false')
    payload_zero_pos = BinanceMark2LiveGridBroker.build_grid_limit_order_payload(
        symbol="SUIUSDT",
        side="SELL",
        qty_str="1.9",
        price_str="3.0865",
        current_position_amt=0.0,
        order_qty=1.9,
    )
    assert payload_zero_pos["reduceOnly"] == "false" and payload_zero_pos["timeInForce"] == "GTX"

    payload_closing_long = BinanceMark2LiveGridBroker.build_grid_limit_order_payload(
        symbol="SUIUSDT",
        side="SELL",
        qty_str="1.9",
        price_str="3.0865",
        current_position_amt=1.9,
        order_qty=1.9,
    )
    assert payload_closing_long["reduceOnly"] == "true"

    payload_opening_long = BinanceMark2LiveGridBroker.build_grid_limit_order_payload(
        symbol="SUIUSDT",
        side="BUY",
        qty_str="1.9",
        price_str="3.0171",
        current_position_amt=1.9,
        order_qty=1.9,
    )
    assert payload_opening_long["reduceOnly"] == "false"

    # Verificación anti -4509: orientación geométrica correcta de SL/TP en posición SHORT sobre plan NEUTRAL/LONG
    short_sl, short_tp = BinanceMark2LiveGridBroker.resolve_directional_bracket_prices(
        eff_direction=GridDirection.SHORT,
        sl_price=1.0800,
        tp_price=1.2500,
        current_mark_price=1.1589,
        tick_size=0.0001,
        lower_price=1.1100,
        upper_price=1.2100,
    )
    assert short_sl >= 1.2500 and short_tp <= 1.0800, (short_sl, short_tp)

    # Verificación de clientAlgoId válido y ventana temporal < 7 días / < 90 días
    dummy_broker = BinanceMark2LiveGridBroker(env_path="__nonexistent_env__")
    cid_a = dummy_broker.build_valid_client_algo_id("M2GSL", "1000SHIBUSDT")
    cid_b = dummy_broker.build_valid_client_algo_id("M2GSL", "1000SHIBUSDT")
    assert cid_a != cid_b and len(cid_a) <= 32 and ALGO_CLIENT_ID_PATTERN.match(cid_a)

    now_ms = int(time.time() * 1000)
    st_ms, et_ms = dummy_broker.clamp_algo_time_range(now_ms - 30 * 86_400_000, now_ms)
    assert 0 < (et_ms - st_ms) <= MAX_ALGO_QUERY_WINDOW_MS

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        f"✅ [M5_GRID_EXECUTION_AUDIT OK] Cruces={t_total} | PnL=+${executor.realized_pnl_usdt:.4f} | Trailing={trail['trailing_action']}"
    )


if __name__ == "__main__":
    _self_test_m5_module()
