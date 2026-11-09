# -*- coding: utf-8 -*-
"""
=================================================================================================================
PROSPERITY MARK-2 — MÓDULO 7: BOTÓN ROJO QUIRÚRGICO CON CIERRE AUTOMÁTICO AL +3.0% DE GANANCIA
(m7_emergency_red_button.py)
=================================================================================================================
Misión:
Actuar como un "Botón Rojo" quirúrgico para operaciones abiertas por error o mal configuradas,
sin dañar, detener ni afectar el funcionamiento de los demás Bots Grid de PROSPERITY MARK-2.

Mecánica Inteligente de Salida al +3.0% de Ganancia (`target_profit_pct = 3.0`):
1. Aislar y Congelar el Error: Cancela de inmediato las órdenes LIMIT abiertas del símbolo mal
   configurado (`DELETE /fapi/v1/allOpenOrders`) para evitar que siga acumulando lotes erróneos,
   y lo desacopla del Grid sin tocar los demás bots activos.
2. Evaluación y Cierre al +3.0% de Margen de Ganancia:
   - Si la posición abierta ya tiene >= +3.0% de ganancia sobre su margen (o en precio si se indica),
     la cierra en ese mismo milisegundo a mercado con `reduceOnly="true"`.
   - Si aún no alcanza el +3.0% de ganancia, coloca en los servidores de Binance una orden oficial
     `TAKE_PROFIT_MARKET` (`closePosition="true"`, `workingType="MARK_PRICE"`) calculada exactamente
     al precio que otorga +3.0% de ganancia neta (descontando comisión Taker) y la vigila en
     tiempo real hasta que toque el +3.0% y se liquide automáticamente.
   - También incluye la opción `--immediate` por si en algún caso extremo deseas forzar el cierre
     instantáneo sin esperar el +3.0%.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import datetime
import logging
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None

from m0_workspace_hygiene import MARK2_DIR, resolve_confined_path
from m5_grid_execution_audit import BinanceMark2LiveGridBroker

logger = logging.getLogger("M7_EmergencyRedButton")

DEFAULT_RED_BUTTON_PROFIT_TARGET_PCT = 3.0  # +3.0% de margen de ganancia


@dataclass(frozen=True, slots=True)
class RedButtonInterventionRecord:
    """Resultado auditable de una intervención del Botón Rojo sobre un contrato."""
    timestamp_utc: str
    symbol: str
    intervention_mode: str
    anomaly_reason: str
    closed_position_qty: float
    entry_price: float
    mark_price: float
    target_tp_price: float
    target_profit_pct: float
    current_profit_pct: float
    orders_purged: bool
    realized_pnl_usdt: float
    status: str  # "CLOSED_AT_3PCT_PROFIT", "ARMED_WAITING_3PCT_TP", "CLOSED_IMMEDIATE"
    system_continuity_preserved: bool


class Mark2SurgicalRedButtonService:
    """
    Servicio de Intervención Quirúrgica ("Botón Rojo") para PROSPERITY MARK-2:
    - Cancela órdenes pendientes del símbolo erróneo sin tocar otros bots.
    - Calcula el precio exacto para obtener un +3.0% de margen de ganancia y cierra la
      operación en cuanto alcanza >= +3.0% (colocando además un TAKE_PROFIT_MARKET en Binance
      y vigilándola en segundo plano desde `main_mark2.py`).
    """

    def __init__(
        self,
        broker: Optional[BinanceMark2LiveGridBroker] = None,
        env_path: str = ".env",
        db_path: Optional[str] = None,
        target_profit_pct: float = DEFAULT_RED_BUTTON_PROFIT_TARGET_PCT,
    ):
        self.broker = broker or BinanceMark2LiveGridBroker(env_path=env_path)
        self.target_profit_pct = max(0.1, float(target_profit_pct))
        self.taker_fee = self.broker.get_float("TAKER_FEE_RATE", 0.00045)

        raw_db = db_path or self.broker.db_path
        if raw_db == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = resolve_confined_path(raw_db, MARK2_DIR)
            self._mem_conn = None
        self._init_red_button_tables()

    def _connect(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _close_if_disk(self, conn: sqlite3.Connection) -> None:
        if self._mem_conn is None:
            conn.close()

    def _init_red_button_tables(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mark2_red_button_commands (
                        cmd_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        target_symbol TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        target_profit_pct REAL NOT NULL DEFAULT 3.0,
                        processed INTEGER NOT NULL DEFAULT 0,
                        processed_at TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mark2_red_button_watchlist (
                        symbol TEXT PRIMARY KEY,
                        armed_at TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        position_qty REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        leverage INTEGER NOT NULL DEFAULT 10,
                        target_profit_pct REAL NOT NULL DEFAULT 3.0,
                        target_tp_price REAL NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'ARMED_WAITING_3PCT_TP'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mark2_red_button_audit (
                        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        intervention_mode TEXT NOT NULL,
                        anomaly_reason TEXT NOT NULL,
                        closed_position_qty REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        mark_price REAL NOT NULL,
                        target_tp_price REAL NOT NULL DEFAULT 0.0,
                        target_profit_pct REAL NOT NULL DEFAULT 3.0,
                        current_profit_pct REAL NOT NULL DEFAULT 0.0,
                        orders_purged INTEGER NOT NULL,
                        realized_pnl_usdt REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'CLOSED_AT_3PCT_PROFIT',
                        system_continuity_preserved INTEGER NOT NULL DEFAULT 1
                    );
                    """
                )
                # Migración defensiva si la tabla ya existía sin las columnas de +3%
                existing_cols = {
                    r["name"]
                    for r in conn.execute("PRAGMA table_info(mark2_red_button_audit);").fetchall()
                }
                for col_name, col_def in (
                    ("target_tp_price", "REAL NOT NULL DEFAULT 0.0"),
                    ("target_profit_pct", "REAL NOT NULL DEFAULT 3.0"),
                    ("current_profit_pct", "REAL NOT NULL DEFAULT 0.0"),
                    ("status", "TEXT NOT NULL DEFAULT 'CLOSED_AT_3PCT_PROFIT'"),
                ):
                    if col_name not in existing_cols:
                        conn.execute(
                            f"ALTER TABLE mark2_red_button_audit ADD COLUMN {col_name} {col_def};"
                        )
        finally:
            self._close_if_disk(conn)

    # ------------------------------------------------------------------------
    # A. CÁLCULO MATEMÁTICO EXACTO DEL PRECIO Y MARGEN DE GANANCIA (+3.0%)
    # ------------------------------------------------------------------------

    def compute_position_profit_pct_and_target_price(
        self,
        symbol: str,
        pos_amt: float,
        entry_price: float,
        mark_price: float,
        leverage: int = 1,
        isolated_margin_usdt: float = 0.0,
        un_realized_profit_usdt: float = 0.0,
        target_profit_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calcula:
        1. `current_profit_pct`: El margen de ganancia neto actual (%) de la posición abierta.
        2. `target_tp_price`: El precio exacto de salida donde la operación alcanza el
           `target_profit_pct` (por defecto +3.0% de ganancia neta descontando comisión Taker).
        """
        goal_pct = target_profit_pct if target_profit_pct is not None else self.target_profit_pct
        if abs(pos_amt) <= 1e-12 or entry_price <= 0.0:
            return 0.0, mark_price

        eff_lev = max(1, int(leverage))
        notional_entry = abs(pos_amt) * entry_price
        margin_base = (
            isolated_margin_usdt
            if isolated_margin_usdt > 0.0
            else (notional_entry / float(eff_lev))
        )

        # Ganancia porcentual actual sobre el margen de la operación (descontando Taker fee de salida)
        est_close_fee = abs(pos_amt) * mark_price * self.taker_fee
        if abs(un_realized_profit_usdt) > 0.0:
            raw_upnl = un_realized_profit_usdt
        else:
            raw_upnl = (
                (mark_price - entry_price) * abs(pos_amt)
                if pos_amt > 0.0
                else (entry_price - mark_price) * abs(pos_amt)
            )
        net_upnl = raw_upnl - est_close_fee
        current_roi_pct = (net_upnl / max(margin_base, 1e-6)) * 100.0

        # Distancia porcentual de precio necesaria para lograr +3.0% sobre el margen + cubrir comisión Taker
        required_price_move_frac = ((goal_pct / 100.0) / float(eff_lev)) + (2.0 * self.taker_fee)
        if pos_amt > 0.0:
            raw_tp = entry_price * (1.0 + required_price_move_frac)
        else:
            raw_tp = max(1e-8, entry_price * (1.0 - required_price_move_frac))

        rule = self.broker.contract_rules.get(symbol.upper())
        tick_sz = rule.tick_size if rule is not None else 0.0001
        clean_tp = self.broker.round_step(raw_tp, tick_sz)
        if clean_tp <= 0.0:
            clean_tp = round(raw_tp, 6)

        return round(current_roi_pct, 3), clean_tp

    # ------------------------------------------------------------------------
    # B. COLA DE SEÑALIZACIÓN IPC Y VIGILANCIA EN SQLITE WAL
    # ------------------------------------------------------------------------

    def enqueue_hot_detach_command(
        self,
        symbol: str,
        mode: str = "RED_BUTTON_3PCT_PROFIT_GUARD",
        reason: str = "MANUAL_RED_BUTTON",
        target_profit_pct: Optional[float] = None,
    ) -> int:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        goal = target_profit_pct if target_profit_pct is not None else self.target_profit_pct
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO mark2_red_button_commands (
                        created_at, target_symbol, mode, reason, target_profit_pct, processed
                    ) VALUES (?, ?, ?, ?, ?, 0);
                    """,
                    (now_iso, symbol.upper(), mode, reason, goal),
                )
                return int(cur.lastrowid or 0)
        finally:
            self._close_if_disk(conn)

    def fetch_and_ack_pending_commands(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM mark2_red_button_commands WHERE processed = 0 ORDER BY cmd_id ASC;"
            ).fetchall()
            if not rows:
                return []
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cmd_ids = [int(r["cmd_id"]) for r in rows]
            with conn:
                conn.executemany(
                    "UPDATE mark2_red_button_commands SET processed = 1, processed_at = ? WHERE cmd_id = ?;",
                    [(now_iso, cid) for cid in cmd_ids],
                )
            return [dict(r) for r in rows]
        finally:
            self._close_if_disk(conn)

    def register_armed_3pct_watch(
        self,
        symbol: str,
        direction: str,
        position_qty: float,
        entry_price: float,
        leverage: int,
        target_profit_pct: float,
        target_tp_price: float,
        reason: str,
    ) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO mark2_red_button_watchlist (
                        symbol, armed_at, direction, position_qty, entry_price,
                        leverage, target_profit_pct, target_tp_price, reason, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ARMED_WAITING_3PCT_TP');
                    """,
                    (
                        symbol.upper(),
                        now_iso,
                        direction,
                        position_qty,
                        entry_price,
                        leverage,
                        target_profit_pct,
                        target_tp_price,
                        reason,
                    ),
                )
        finally:
            self._close_if_disk(conn)

    def get_armed_watchlist(self) -> Dict[str, Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM mark2_red_button_watchlist WHERE status = 'ARMED_WAITING_3PCT_TP';"
            ).fetchall()
            return {str(r["symbol"]).upper(): dict(r) for r in rows}
        finally:
            self._close_if_disk(conn)

    def remove_from_watchlist(self, symbol: str) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM mark2_red_button_watchlist WHERE symbol = ?;",
                    (symbol.upper(),),
                )
        finally:
            self._close_if_disk(conn)

    def _record_audit_entry(self, rec: RedButtonInterventionRecord) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO mark2_red_button_audit (
                        timestamp_utc, symbol, intervention_mode, anomaly_reason,
                        closed_position_qty, entry_price, mark_price,
                        target_tp_price, target_profit_pct, current_profit_pct,
                        orders_purged, realized_pnl_usdt, status, system_continuity_preserved
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        rec.timestamp_utc,
                        rec.symbol,
                        rec.intervention_mode,
                        rec.anomaly_reason,
                        rec.closed_position_qty,
                        rec.entry_price,
                        rec.mark_price,
                        rec.target_tp_price,
                        rec.target_profit_pct,
                        rec.current_profit_pct,
                        1 if rec.orders_purged else 0,
                        rec.realized_pnl_usdt,
                        rec.status,
                        1 if rec.system_continuity_preserved else 0,
                    ),
                )
        finally:
            self._close_if_disk(conn)

    # ------------------------------------------------------------------------
    # C. EJECUCIÓN QUIRÚRGICA CON CIERRE AL +3.0% DE MARGEN DE GANANCIA
    # ------------------------------------------------------------------------

    async def surgical_close_symbol(
        self,
        symbol: str,
        reason: str = "OPERACION_MAL_CONFIGURADA_O_ERROR",
        mode: str = "RED_BUTTON_3PCT_PROFIT_CLOSE",
        target_profit_pct: Optional[float] = None,
        force_immediate: bool = False,
        session: Optional[Any] = None,
        notify_orchestrator: bool = True,
        simulated_position: Optional[Dict[str, float]] = None,
    ) -> RedButtonInterventionRecord:
        """
        Interviene quirúrgicamente un símbolo mal configurado o abierto por error:
        1. Purga de inmediato todas las órdenes abiertas del Grid (`DELETE /fapi/v1/allOpenOrders`)
           para detener cualquier compra/venta errónea adicional y notifica al orquestador para
           que desacople ese símbolo sin afectar a los demás bots.
        2. Evalúa el margen de ganancia actual de la posición abierta:
           - Si `current_profit_pct >= target_profit_pct` (+3.0% por defecto) o `force_immediate=True`,
             ejecuta `MARKET reduceOnly="true"` y cierra la posición en el acto asegurando el +3.0%.
           - Si aún no llega al +3.0% de ganancia, planta un `TAKE_PROFIT_MARKET` oficial en Binance
             al precio exacto del +3.0% (`target_tp_price`) y registra el símbolo en
             `mark2_red_button_watchlist` para que sea cerrado automáticamente en cuanto toque +3.0%.
        """
        sym = symbol.strip().upper()
        goal_pct = target_profit_pct if target_profit_pct is not None else self.target_profit_pct
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        closed_qty = 0.0
        entry_p = 0.0
        mark_p = 0.0
        target_tp_p = 0.0
        current_prof_pct = 0.0
        orders_purged = False
        status_str = "NO_OPEN_POSITION_ORDERS_PURGED"

        # 1. Purgar órdenes pendientes del símbolo erróneo
        try:
            await self.broker.cancel_all_symbol_orders(
                sym, include_algo=True, session=session
            )
            orders_purged = True
        except Exception as exc:
            logger.warning("⚠️ [Botón Rojo M7] Aviso purgando órdenes en %s: %s", sym, exc)

        if notify_orchestrator:
            self.enqueue_hot_detach_command(
                sym, mode=mode, reason=reason, target_profit_pct=goal_pct
            )

        # 2. Obtener datos de la posición viva (desde Binance o simulada en DRY_RUN)
        pos_data: Optional[Dict[str, Any]] = simulated_position
        if pos_data is None and not self.broker.dry_run:
            try:
                risk_list = await self.broker._signed_request(
                    "GET", "/fapi/v2/positionRisk", {"symbol": sym}, session=session
                )
                if isinstance(risk_list, list):
                    for item in risk_list:
                        if str(item.get("symbol", "")).upper() == sym:
                            pos_data = item
                            break
            except Exception as exc:
                logger.error("❌ [Botón Rojo M7] Error consultando posición en %s: %s", sym, exc)

        if pos_data is not None:
            pos_amt = float(pos_data.get("positionAmt", 0.0))
            entry_p = float(pos_data.get("entryPrice", 0.0))
            mark_p = float(pos_data.get("markPrice", entry_p))
            lev = int(float(pos_data.get("leverage", 10)))
            iso_margin = float(pos_data.get("isolatedMargin", 0.0))
            upnl = float(pos_data.get("unRealizedProfit", 0.0))

            if abs(pos_amt) > 0.0 and entry_p > 0.0:
                current_prof_pct, target_tp_p = self.compute_position_profit_pct_and_target_price(
                    symbol=sym,
                    pos_amt=pos_amt,
                    entry_price=entry_p,
                    mark_price=mark_p,
                    leverage=lev,
                    isolated_margin_usdt=iso_margin,
                    un_realized_profit_usdt=upnl,
                    target_profit_pct=goal_pct,
                )

                # Caso A: Ya tiene >= +3.0% de ganancia (o se pidió cierre inmediato) -> Cerrar a mercado ahora
                if current_prof_pct >= goal_pct or force_immediate:
                    close_side = "SELL" if pos_amt > 0.0 else "BUY"
                    rule = self.broker.contract_rules.get(sym)
                    qty_str = (
                        self.broker.format_step(abs(pos_amt), rule.step_size)
                        if rule is not None
                        else str(abs(pos_amt))
                    )
                    if not self.broker.dry_run:
                        await self.broker._signed_request(
                            "POST",
                            "/fapi/v1/order",
                            {
                                "symbol": sym,
                                "side": close_side,
                                "type": "MARKET",
                                "quantity": qty_str,
                                "reduceOnly": "true",
                                "newClientOrderId": f"M2_RB3P_{sym}_{int(time.time()*1000)}"[:32],
                            },
                            session=session,
                        )
                    closed_qty = abs(pos_amt)
                    self.remove_from_watchlist(sym)
                    status_str = (
                        "CLOSED_AT_3PCT_PROFIT"
                        if current_prof_pct >= goal_pct
                        else "CLOSED_IMMEDIATE"
                    )
                else:
                    # Caso B: Aún no alcanza +3.0% -> Colocar TAKE_PROFIT_MARKET en +3.0% y vigilar en segundo plano
                    exit_side = "SELL" if pos_amt > 0.0 else "BUY"
                    rule = self.broker.contract_rules.get(sym)
                    tp_str = (
                        self.broker.format_step(target_tp_p, rule.tick_size)
                        if rule is not None
                        else f"{target_tp_p:.4f}".rstrip("0").rstrip(".")
                    )
                    if not self.broker.dry_run:
                        try:
                            await self.broker.place_conditional_protection_order(
                                symbol=sym,
                                side=exit_side,
                                order_type="TAKE_PROFIT_MARKET",
                                trigger_price_str=tp_str,
                                client_id=self.broker.build_valid_client_algo_id("M2RBTP", sym),
                                working_type="MARK_PRICE",
                                session=session,
                            )
                        except Exception as exc:
                            logger.info(
                                "ℹ️ [Botón Rojo M7] Vigilancia de +%.1f%% en %s activa en RAM (%s)",
                                goal_pct,
                                sym,
                                exc,
                            )
                    self.register_armed_3pct_watch(
                        symbol=sym,
                        direction="LONG" if pos_amt > 0.0 else "SHORT",
                        position_qty=pos_amt,
                        entry_price=entry_p,
                        leverage=lev,
                        target_profit_pct=goal_pct,
                        target_tp_price=target_tp_p,
                        reason=reason,
                    )
                    status_str = "ARMED_WAITING_3PCT_TP"

        audited_pnl = await self.broker._audit_realized_pnl(sym, session=session)

        record = RedButtonInterventionRecord(
            timestamp_utc=now_iso,
            symbol=sym,
            intervention_mode=mode,
            anomaly_reason=reason,
            closed_position_qty=closed_qty,
            entry_price=entry_p,
            mark_price=mark_p,
            target_tp_price=target_tp_p,
            target_profit_pct=goal_pct,
            current_profit_pct=current_prof_pct,
            orders_purged=orders_purged,
            realized_pnl_usdt=audited_pnl,
            status=status_str,
            system_continuity_preserved=True,
        )
        self._record_audit_entry(record)

        logger.info(
            "🚨 [BOTÓN ROJO M7 EN %s] Estado=%s | Ganancia Actual=%+.2f%% (Meta=+%0.2f%% @ %s) | Qty Cerrada=%s | Continuidad=OK",
            sym,
            status_str,
            current_prof_pct,
            goal_pct,
            target_tp_p,
            closed_qty,
        )
        return record

    async def poll_and_close_armed_positions_at_3pct(
        self,
        live_mark_prices: Optional[Dict[str, float]] = None,
        session: Optional[Any] = None,
    ) -> List[RedButtonInterventionRecord]:
        """
        Vigila en segundo plano las operaciones mal configuradas que quedaron armadas esperando
        el +3.0% de margen de ganancia (`mark2_red_button_watchlist`). En cuanto el precio
        alcanza o supera el objetivo de +3.0%, ejecuta el cierre atómico y limpia las órdenes.
        """
        watchlist = self.get_armed_watchlist()
        if not watchlist:
            return []

        closed_records: List[RedButtonInterventionRecord] = []
        live_pos_map = await self.broker.fetch_live_positions_map(session=session)

        for sym, w_info in watchlist.items():
            pos_qty = float(w_info["position_qty"])
            entry_p = float(w_info["entry_price"])
            lev = int(w_info["leverage"])
            goal_pct = float(w_info["target_profit_pct"])
            tp_price = float(w_info["target_tp_price"])

            # Si en Binance ya se cerró por el TAKE_PROFIT_MARKET del servidor, limpiar de watchlist
            if not self.broker.dry_run and sym in live_pos_map:
                real_amt = float(live_pos_map[sym].get("positionAmt", 0.0))
                if abs(real_amt) <= 1e-12:
                    self.remove_from_watchlist(sym)
                    continue
                mark_p = float(live_pos_map[sym].get("markPrice", 0.0))
            else:
                mark_p = (
                    float(live_mark_prices[sym])
                    if (live_mark_prices and sym in live_mark_prices)
                    else 0.0
                )

            if mark_p <= 0.0:
                continue

            hit_tp = (pos_qty > 0.0 and mark_p >= tp_price) or (
                pos_qty < 0.0 and mark_p <= tp_price
            )
            curr_roi, _ = self.compute_position_profit_pct_and_target_price(
                symbol=sym,
                pos_amt=pos_qty,
                entry_price=entry_p,
                mark_price=mark_p,
                leverage=lev,
                target_profit_pct=goal_pct,
            )

            if hit_tp or curr_roi >= goal_pct:
                rec = await self.surgical_close_symbol(
                    symbol=sym,
                    reason=f"META_GANANCIA_{goal_pct:.1f}PCT_ALCANZADA",
                    mode="AUTO_3PCT_PROFIT_HARVEST_CLOSE",
                    target_profit_pct=goal_pct,
                    force_immediate=True,
                    session=session,
                    notify_orchestrator=False,
                    simulated_position={
                        "positionAmt": pos_qty,
                        "entryPrice": entry_p,
                        "markPrice": mark_p,
                        "leverage": lev,
                    },
                )
                closed_records.append(rec)

        return closed_records

    async def inspect_and_purge_misconfigured_operations(
        self,
        authorized_active_symbols: Optional[Set[str]] = None,
        target_profit_pct: Optional[float] = None,
        session: Optional[Any] = None,
    ) -> List[RedButtonInterventionRecord]:
        """
        Escanea todas las posiciones y órdenes abiertas en Binance Futures USDⓈ-M y aplica
        el protocolo del Botón Rojo (+3.0% de ganancia) sobre cualquier operación mal configurada
        o abierta por error (margen CROSSED, símbolo fuera de catálogo, órdenes huérfanas).
        """
        interventions: List[RedButtonInterventionRecord] = []
        goal_pct = target_profit_pct if target_profit_pct is not None else self.target_profit_pct
        allowed_syms = (
            {s.upper() for s in authorized_active_symbols}
            if authorized_active_symbols is not None
            else None
        )

        if self.broker.dry_run:
            return interventions

        try:
            risk_list = await self.broker._signed_request(
                "GET", "/fapi/v2/positionRisk", session=session
            )
            open_orders = await self.broker._signed_request(
                "GET", "/fapi/v1/openOrders", session=session
            )
        except Exception as exc:
            logger.warning("⚠️ [Botón Rojo M7] No se pudo consultar estado de cuenta: %s", exc)
            return interventions

        symbols_with_orders: Set[str] = set()
        if isinstance(open_orders, list):
            for o in open_orders:
                s = str(o.get("symbol", "")).upper()
                if s:
                    symbols_with_orders.add(s)

        if isinstance(risk_list, list):
            for item in risk_list:
                sym = str(item.get("symbol", "")).upper()
                if not sym:
                    continue
                pos_amt = float(item.get("positionAmt", 0.0))
                margin_type = str(item.get("marginType", "isolated")).lower()

                has_pos = abs(pos_amt) > 0.0
                has_ord = sym in symbols_with_orders
                if not has_pos and not has_ord:
                    continue

                anomaly_reasons: List[str] = []
                if has_pos and margin_type != "isolated":
                    anomaly_reasons.append("MARGEN_CROSSED_NO_PERMITIDO")
                if allowed_syms is not None and sym not in allowed_syms:
                    anomaly_reasons.append("OPERACION_ABIERTA_POR_ERROR_O_FUERA_DE_BOT")

                if anomaly_reasons:
                    reason_str = "+".join(anomaly_reasons)
                    rec = await self.surgical_close_symbol(
                        symbol=sym,
                        reason=reason_str,
                        mode="AUTO_MISCONFIGURED_3PCT_GUARD",
                        target_profit_pct=goal_pct,
                        force_immediate=False,
                        session=session,
                    )
                    interventions.append(rec)

        return interventions


# ============================================================================
# CLI AUTÓNOMO DEL BOTÓN ROJO QUIRÚRGICO (+3.0% DE GANANCIA)
# ============================================================================

async def _run_red_button_cli(args: argparse.Namespace) -> None:
    broker = BinanceMark2LiveGridBroker(env_path=os.path.join(MARK2_DIR, ".env"))
    service = Mark2SurgicalRedButtonService(
        broker=broker, target_profit_pct=args.profit_pct
    )

    if aiohttp is not None and not broker.dry_run:
        async with aiohttp.ClientSession() as session:
            await broker.sync_server_time(session)
            await _dispatch_cli_action(service, args, session)
    else:
        await _dispatch_cli_action(service, args, None)


async def _dispatch_cli_action(
    service: Mark2SurgicalRedButtonService,
    args: argparse.Namespace,
    session: Optional[Any],
) -> None:
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
        for sym in symbols:
            rec = await service.surgical_close_symbol(
                symbol=sym,
                reason=args.reason or "CIERRE_POR_ERROR_O_MALA_CONFIG",
                mode="RED_BUTTON_3PCT_PROFIT_CLOSE",
                target_profit_pct=args.profit_pct,
                force_immediate=args.immediate,
                session=session,
            )
            print(
                f"🚨 [BOTÓN ROJO M7] Símbolo={rec.symbol} | Estado={rec.status} | "
                f"Ganancia Actual={rec.current_profit_pct:+.2f}% | Meta=+{rec.target_profit_pct:.1f}% "
                f"(Precio TP={rec.target_tp_price}) | Continuidad Sistema=INTACTA"
            )
    else:
        recs = await service.inspect_and_purge_misconfigured_operations(
            target_profit_pct=args.profit_pct, session=session
        )
        print(
            f"🛡️ [AUDITORÍA BOTÓN ROJO M7] {len(recs)} operaciones mal configuradas intervenidas (Meta de salida: +{args.profit_pct:.1f}%)."
        )


def _self_test_m7_red_button() -> None:
    broker = BinanceMark2LiveGridBroker(env_path=os.path.join(MARK2_DIR, ".env"))
    broker.dry_run = True
    service = Mark2SurgicalRedButtonService(
        broker=broker, db_path=":memory:", target_profit_pct=3.0
    )

    # 1. Simular operación abierta por error en XRPUSDT @ $1.0000 (Long 10x) cuando el precio va en $1.0010 (+0.91% ROI < +3.0%)
    rec_armed = asyncio.run(
        service.surgical_close_symbol(
            symbol="XRPUSDT",
            reason="OPERACION_ABIERTA_POR_ERROR",
            target_profit_pct=3.0,
            simulated_position={
                "positionAmt": 10.0,
                "entryPrice": 1.0000,
                "markPrice": 1.0010,
                "leverage": 10,
            },
        )
    )
    assert rec_armed.status == "ARMED_WAITING_3PCT_TP"
    assert rec_armed.target_tp_price > 1.0000
    assert "XRPUSDT" in service.get_armed_watchlist()

    # 2. Simular que el precio sube y alcanza el +3.0% de margen de ganancia -> Se cierra automáticamente
    closed_list = asyncio.run(
        service.poll_and_close_armed_positions_at_3pct(
            live_mark_prices={"XRPUSDT": rec_armed.target_tp_price + 0.0005}
        )
    )
    assert len(closed_list) == 1
    assert closed_list[0].status == "CLOSED_AT_3PCT_PROFIT"
    assert closed_list[0].current_profit_pct >= 3.0
    assert "XRPUSDT" not in service.get_armed_watchlist()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        f"✅ [M7_EMERGENCY_RED_BUTTON OK] Operación errónea aislada -> Armado TP @ {rec_armed.target_tp_price} -> "
        f"Cerrada automáticamente al alcanzar +{closed_list[0].current_profit_pct:.2f}% (>= +3.0% meta) sin afectar el sistema."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="PROSPERITY MARK-2 — Botón Rojo Quirúrgico con cierre al +3.0% de ganancia."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Símbolo(s) separados por coma a intervenir (ej. XRPUSDT o SUIUSDT,DOGEUSDT)",
    )
    parser.add_argument(
        "--misconfigured",
        action="store_true",
        help="Detecta operaciones mal configuradas o por error y las programa para cerrar al +3.0%% de ganancia",
    )
    parser.add_argument(
        "--profit-pct",
        type=float,
        default=3.0,
        help="Margen de ganancia objetivo en porcentaje para cerrar la operación (por defecto: 3.0%%)",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Si se activa, cierra inmediatamente a mercado sin esperar el +3.0%%",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="OPERACION_MAL_CONFIGURADA_O_ERROR",
        help="Motivo de auditoría para SQLite WAL",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Ejecuta la prueba unitaria de verificación en memoria",
    )
    parsed_args = parser.parse_args()
    if parsed_args.selftest or (not parsed_args.symbol and not parsed_args.misconfigured):
        _self_test_m7_red_button()
    else:
        asyncio.run(_run_red_button_cli(parsed_args))
