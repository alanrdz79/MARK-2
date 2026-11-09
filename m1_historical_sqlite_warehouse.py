# -*- coding: utf-8 -*-
"""
=================================================================================================================
PROSPERITY MARK-2 — MÓDULO 1B: EXTRACCIÓN DE DATOS HISTÓRICOS Y ALMACÉN SQLITE WAL
(m1_historical_sqlite_warehouse.py)
=================================================================================================================
Arquitectura de Extracción Histórica y Feature Store en SQLite WAL para MARK-2:
1. Migración Instantánea Zero-Copy (`ATTACH DATABASE`): Importa de manera idempotente el acervo histórico
   de `Model MARK-1/prosperity_backtest_warehouse.db` (velas MTF, derivados y filtros) hacia
   `MARK-2/prosperity_backtest_warehouse.db` sin pérdida de datos.
2. Ingesta Paginada Asíncrona desde Binance Futures USDⓈ-M (`Mark2BinanceHistoricalETLService`):
   Descarga y actualiza incrementalmente `/fapi/v1/klines`, `/fapi/v1/fundingRate`,
   `/futures/data/openInterestHist` y `/futures/data/topLongShortPositionRatio` con Aduana O(1).
3. Tablas de Optimización Walk-Forward de Grids (`grid_walk_forward_runs` y `active_grid_model_config`):
   Almacena el desempeño histórico real de los Grids Geométricos (In-Sample / Out-of-Sample) y publica
   los parámetros óptimos para el motor en vivo (`main_mark2.py`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import datetime
import logging
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

from core_mark2_contracts import ExchangeContractRulePOD
from m0_workspace_hygiene import MARK1_SIBLING_DIR, MARK2_DIR, resolve_confined_path

logger = logging.getLogger("M1_HistoricalSQLiteWarehouse")

TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
}


# ============================================================================
# 1. CONTRATOS INMUTABLES DE SERIES HISTÓRICAS Y WALK-FORWARD DE GRIDS
# ============================================================================

@dataclass(frozen=True, slots=True)
class HistoricalCandleRecordM2:
    symbol: str
    timeframe: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float
    quote_volume: float = 0.0
    trade_count: int = 0
    taker_buy_quote_volume: float = 0.0


@dataclass(frozen=True, slots=True)
class HistoricalDerivativeRecordM2:
    symbol: str
    timestamp_ms: int
    funding_rate: float
    open_interest_usdt: float
    top_trader_ls_ratio: float


@dataclass(frozen=True, slots=True)
class GridWalkForwardRecordM2:
    wf_id: str
    symbol: str
    timeframe: str
    direction: str
    nominal_leverage: int
    effective_max_leverage: float
    grid_count: int
    geometric_step_pct: float
    is_crosses: int
    is_net_roi_pct: float
    is_max_dd_pct: float
    oos_crosses: int
    oos_net_roi_pct: float
    oos_max_dd_pct: float
    full_crosses: int
    full_net_pnl_usdt: float
    full_net_roi_pct: float
    min_liq_buffer_pct: float
    bayesian_survival_prob: float
    wf_composite_score: float


# ============================================================================
# 2. ADUANA DETERMINISTA O(1) DE SERIES HISTÓRICAS
# ============================================================================

class Mark2HistoricalDataValidator:
    """Validador determinista en tiempo constante O(1) para velas y derivados."""

    @staticmethod
    def is_valid_candle(c: HistoricalCandleRecordM2) -> bool:
        vals = (
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume,
            c.taker_buy_volume,
            c.quote_volume,
            c.taker_buy_quote_volume,
        )
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            return False
        if c.open <= 0.0 or c.high <= 0.0 or c.low <= 0.0 or c.close <= 0.0:
            return False
        if c.high < c.low or c.high < c.open or c.high < c.close:
            return False
        if c.low > c.open or c.low > c.close:
            return False
        if c.volume < 0.0 or c.taker_buy_volume < 0.0 or c.taker_buy_volume > c.volume * 1.001:
            return False
        if c.trade_count < 0:
            return False
        return True


# ============================================================================
# 3. REPOSITORIO SQLITE WAL PARA EL ALMACÉN HISTÓRICO DE MARK-2
# ============================================================================

class Mark2SQLiteWarehouseRepository:
    """
    Gestiona `MARK-2/prosperity_backtest_warehouse.db` en modo SQLite WAL:
    - Almacena velas históricas MTF (`1m`, `5m`, `15m`, `30m`, `1h`, `4h`).
    - Almacena métricas históricas de derivados (`funding_rate`, `open_interest_usdt`, `top_trader_ls_ratio`).
    - Sincroniza automáticamente los registros históricos previos desde `Model MARK-1` si existen.
    - Persiste corridas Walk-Forward de Grids y publica la tabla `active_grid_model_config`.
    """

    def __init__(
        self,
        db_path: str = "prosperity_backtest_warehouse.db",
        auto_migrate_from_mark1: bool = True,
    ):
        if db_path == ":memory:":
            self.db_path = ":memory:"
            self._mem_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = resolve_confined_path(db_path, MARK2_DIR)
            self._mem_conn = None

        self._init_schema()
        if auto_migrate_from_mark1 and self.db_path != ":memory:":
            self.migrate_from_mark1_warehouse_if_available()

    def _connect(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _close_if_disk(self, conn: sqlite3.Connection) -> None:
        if self._mem_conn is None:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS exchange_contract_rules (
                        symbol TEXT PRIMARY KEY,
                        tick_size REAL NOT NULL,
                        step_size REAL NOT NULL,
                        min_qty REAL NOT NULL,
                        min_notional REAL NOT NULL,
                        max_num_orders INTEGER NOT NULL DEFAULT 200,
                        maint_margin_rate REAL NOT NULL DEFAULT 0.005,
                        max_leverage INTEGER NOT NULL DEFAULT 125,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS historical_klines (
                        symbol TEXT NOT NULL,
                        open_time INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        taker_buy_volume REAL NOT NULL,
                        quote_volume REAL NOT NULL DEFAULT 0.0,
                        trade_count INTEGER NOT NULL DEFAULT 0,
                        taker_buy_quote_volume REAL NOT NULL DEFAULT 0.0,
                        PRIMARY KEY (symbol, open_time)
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_m2_klines_sym_time
                    ON historical_klines (symbol, open_time ASC);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS historical_klines_mtf (
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        open_time INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        taker_buy_volume REAL NOT NULL,
                        quote_volume REAL NOT NULL DEFAULT 0.0,
                        trade_count INTEGER NOT NULL DEFAULT 0,
                        taker_buy_quote_volume REAL NOT NULL DEFAULT 0.0,
                        is_synthetic INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (symbol, timeframe, open_time)
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_m2_klines_mtf_sym_tf_time
                    ON historical_klines_mtf (symbol, timeframe, open_time ASC);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS historical_derivatives_metrics (
                        symbol TEXT NOT NULL,
                        timestamp_ms INTEGER NOT NULL,
                        funding_rate REAL NOT NULL DEFAULT 0.0,
                        open_interest_usdt REAL NOT NULL DEFAULT 0.0,
                        top_trader_ls_ratio REAL NOT NULL DEFAULT 1.0,
                        PRIMARY KEY (symbol, timestamp_ms)
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_m2_deriv_sym_time
                    ON historical_derivatives_metrics (symbol, timestamp_ms ASC);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS grid_walk_forward_runs (
                        wf_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        nominal_leverage INTEGER NOT NULL,
                        effective_max_leverage REAL NOT NULL,
                        grid_count INTEGER NOT NULL,
                        geometric_step_pct REAL NOT NULL,
                        is_crosses INTEGER NOT NULL,
                        is_net_roi_pct REAL NOT NULL,
                        is_max_dd_pct REAL NOT NULL,
                        oos_crosses INTEGER NOT NULL,
                        oos_net_roi_pct REAL NOT NULL,
                        oos_max_dd_pct REAL NOT NULL,
                        full_crosses INTEGER NOT NULL,
                        full_net_pnl_usdt REAL NOT NULL,
                        full_net_roi_pct REAL NOT NULL,
                        min_liq_buffer_pct REAL NOT NULL,
                        bayesian_survival_prob REAL NOT NULL,
                        wf_composite_score REAL NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS active_grid_model_config (
                        rank_order INTEGER PRIMARY KEY,
                        updated_at TEXT NOT NULL,
                        symbol TEXT NOT NULL UNIQUE,
                        optimal_timeframe TEXT NOT NULL,
                        recommended_direction TEXT NOT NULL,
                        optimal_grid_count INTEGER NOT NULL,
                        optimal_step_pct REAL NOT NULL,
                        recommended_nominal_leverage INTEGER NOT NULL,
                        max_effective_leverage REAL NOT NULL,
                        expected_daily_crosses REAL NOT NULL,
                        expected_days_6_to_18 REAL NOT NULL,
                        bayesian_survival_prob REAL NOT NULL,
                        wf_composite_score REAL NOT NULL
                    );
                    """
                )
        finally:
            self._close_if_disk(conn)

    def migrate_from_mark1_warehouse_if_available(self) -> Dict[str, int]:
        """
        Si existe `Model MARK-1/prosperity_backtest_warehouse.db` y el almacén de MARK-2 está
        vacío o tiene menos registros, migra de forma transaccional e idempotente todas las
        velas históricas (`historical_klines`, `historical_klines_mtf`) y métricas de derivados
        (`historical_derivatives_metrics`) hacia `MARK-2/prosperity_backtest_warehouse.db`.
        """
        m1_wh_path = os.path.join(MARK1_SIBLING_DIR, "prosperity_backtest_warehouse.db")
        if not os.path.isfile(m1_wh_path) or os.path.abspath(m1_wh_path) == os.path.abspath(self.db_path):
            return {"mtf_candles": 0, "klines_5m": 0, "derivatives": 0}

        conn = self._connect()
        migrated_mtf = 0
        migrated_5m = 0
        migrated_deriv = 0
        try:
            cur_mtf = conn.execute("SELECT COUNT(*) FROM historical_klines_mtf;").fetchone()[0]
            if cur_mtf > 50_000:
                return {"mtf_candles": 0, "klines_5m": 0, "derivatives": 0}

            conn.execute("ATTACH DATABASE ? AS mark1_src;", (m1_wh_path,))
            try:
                src_tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM mark1_src.sqlite_master WHERE type='table';"
                    ).fetchall()
                }
                with conn:
                    if "historical_klines_mtf" in src_tables:
                        c1 = conn.execute(
                            """
                            INSERT OR IGNORE INTO main.historical_klines_mtf (
                                symbol, timeframe, open_time, open, high, low, close,
                                volume, taker_buy_volume, quote_volume, trade_count,
                                taker_buy_quote_volume, is_synthetic
                            )
                            SELECT
                                symbol, timeframe, open_time, open, high, low, close,
                                volume, taker_buy_volume, quote_volume, trade_count,
                                taker_buy_quote_volume, is_synthetic
                            FROM mark1_src.historical_klines_mtf;
                            """
                        ).rowcount
                        migrated_mtf = max(0, c1)

                    if "historical_klines" in src_tables:
                        c2 = conn.execute(
                            """
                            INSERT OR IGNORE INTO main.historical_klines (
                                symbol, open_time, open, high, low, close,
                                volume, taker_buy_volume, quote_volume, trade_count,
                                taker_buy_quote_volume
                            )
                            SELECT
                                symbol, open_time, open, high, low, close,
                                volume, taker_buy_volume, quote_volume, trade_count,
                                taker_buy_quote_volume
                            FROM mark1_src.historical_klines;
                            """
                        ).rowcount
                        migrated_5m = max(0, c2)

                    if "historical_derivatives_metrics" in src_tables:
                        c3 = conn.execute(
                            """
                            INSERT OR IGNORE INTO main.historical_derivatives_metrics (
                                symbol, timestamp_ms, funding_rate, open_interest_usdt, top_trader_ls_ratio
                            )
                            SELECT
                                symbol, timestamp_ms, funding_rate, open_interest_usdt, top_trader_ls_ratio
                            FROM mark1_src.historical_derivatives_metrics;
                            """
                        ).rowcount
                        migrated_deriv = max(0, c3)
            finally:
                conn.execute("DETACH DATABASE mark1_src;")

            if migrated_mtf > 0 or migrated_5m > 0 or migrated_deriv > 0:
                logger.info(
                    "🗄️ [M1 SQLite Warehouse MARK-2] Migración histórica desde MARK-1 completada: "
                    "%d velas MTF | %d velas 5m | %d registros de derivados.",
                    migrated_mtf,
                    migrated_5m,
                    migrated_deriv,
                )
        except Exception as exc:
            logger.warning("⚠️ [M1 SQLite Warehouse] Aviso en migración desde MARK-1: %s", exc)
        finally:
            self._close_if_disk(conn)

        return {
            "mtf_candles": migrated_mtf,
            "klines_5m": migrated_5m,
            "derivatives": migrated_deriv,
        }

    def upsert_contract_rules(self, rules_map: Dict[str, ExchangeContractRulePOD]) -> int:
        if not rules_map:
            return 0
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO exchange_contract_rules (
                        symbol, tick_size, step_size, min_qty, min_notional,
                        max_num_orders, maint_margin_rate, max_leverage, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            r.symbol,
                            r.tick_size,
                            r.step_size,
                            r.min_qty,
                            r.min_notional,
                            r.max_num_orders,
                            r.maint_margin_rate,
                            r.max_leverage,
                            now_iso,
                        )
                        for r in rules_map.values()
                    ],
                )
            return len(rules_map)
        finally:
            self._close_if_disk(conn)

    def load_contract_rules(self) -> Dict[str, ExchangeContractRulePOD]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM exchange_contract_rules;").fetchall()
            out: Dict[str, ExchangeContractRulePOD] = {}
            for r in rows:
                sym = str(r["symbol"]).upper()
                out[sym] = ExchangeContractRulePOD(
                    symbol=sym,
                    tick_size=float(r["tick_size"]),
                    step_size=float(r["step_size"]),
                    min_qty=float(r["min_qty"]),
                    min_notional=float(r["min_notional"]),
                    max_num_orders=int(r["max_num_orders"]),
                    maint_margin_rate=float(r["maint_margin_rate"]),
                    max_leverage=int(r["max_leverage"]),
                )
            return out
        finally:
            self._close_if_disk(conn)

    def upsert_candles(
        self,
        candles: List[HistoricalCandleRecordM2],
        is_synthetic: int = 0,
    ) -> int:
        valid = [c for c in candles if Mark2HistoricalDataValidator.is_valid_candle(c)]
        if not valid:
            return 0
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO historical_klines_mtf (
                        symbol, timeframe, open_time, open, high, low, close,
                        volume, taker_buy_volume, quote_volume, trade_count,
                        taker_buy_quote_volume, is_synthetic
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            c.symbol,
                            c.timeframe,
                            c.open_time,
                            c.open,
                            c.high,
                            c.low,
                            c.close,
                            c.volume,
                            c.taker_buy_volume,
                            c.quote_volume,
                            c.trade_count,
                            c.taker_buy_quote_volume,
                            is_synthetic,
                        )
                        for c in valid
                    ],
                )
                klines_5m = [c for c in valid if c.timeframe == "5m" and is_synthetic == 0]
                if klines_5m:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO historical_klines (
                            symbol, open_time, open, high, low, close,
                            volume, taker_buy_volume, quote_volume, trade_count,
                            taker_buy_quote_volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        [
                            (
                                c.symbol,
                                c.open_time,
                                c.open,
                                c.high,
                                c.low,
                                c.close,
                                c.volume,
                                c.taker_buy_volume,
                                c.quote_volume,
                                c.trade_count,
                                c.taker_buy_quote_volume,
                            )
                            for c in klines_5m
                        ],
                    )
            return len(valid)
        finally:
            self._close_if_disk(conn)

    def upsert_derivatives(self, records: List[HistoricalDerivativeRecordM2]) -> int:
        if not records:
            return 0
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO historical_derivatives_metrics (
                        symbol, timestamp_ms, funding_rate, open_interest_usdt, top_trader_ls_ratio
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            r.symbol,
                            r.timestamp_ms,
                            r.funding_rate,
                            r.open_interest_usdt,
                            r.top_trader_ls_ratio,
                        )
                        for r in records
                        if not (
                            math.isnan(r.funding_rate)
                            or math.isnan(r.open_interest_usdt)
                            or math.isnan(r.top_trader_ls_ratio)
                        )
                    ],
                )
            return len(records)
        finally:
            self._close_if_disk(conn)

    def get_candle_bounds(
        self, symbol: str, timeframe: str = "5m"
    ) -> Tuple[int, Optional[int], Optional[int]]:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt, MIN(open_time) AS mn, MAX(open_time) AS mx
                FROM historical_klines_mtf
                WHERE symbol = ? AND timeframe = ? AND is_synthetic = 0;
                """,
                (symbol.upper(), timeframe),
            ).fetchone()
            if not row or int(row["cnt"]) == 0:
                return 0, None, None
            return int(row["cnt"]), int(row["mn"]), int(row["mx"])
        finally:
            self._close_if_disk(conn)

    def load_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 20000,
    ) -> List[HistoricalCandleRecordM2]:
        sym = symbol.upper()
        query = (
            "SELECT * FROM historical_klines_mtf "
            "WHERE symbol = ? AND timeframe = ? AND is_synthetic = 0"
        )
        params: List[Any] = [sym, timeframe]
        if start_ms is not None:
            query += " AND open_time >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND open_time <= ?"
            params.append(end_ms)
        query += " ORDER BY open_time DESC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM ({query}) ORDER BY open_time ASC;", params
            ).fetchall()
            out: List[HistoricalCandleRecordM2] = []
            for r in rows:
                out.append(
                    HistoricalCandleRecordM2(
                        symbol=str(r["symbol"]),
                        timeframe=str(r["timeframe"]),
                        open_time=int(r["open_time"]),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                        taker_buy_volume=float(r["taker_buy_volume"]),
                        quote_volume=float(r["quote_volume"]),
                        trade_count=int(r["trade_count"]),
                        taker_buy_quote_volume=float(r["taker_buy_quote_volume"]),
                    )
                )
            return out
        finally:
            self._close_if_disk(conn)

    def load_numpy_ohlcv_arrays(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 5000,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extrae desde SQLite WAL arreglos contiguos `np.float64` de alta velocidad
        para simulaciones Walk-Forward y cálculo de telemetría de Ciencia de Datos.
        """
        candles = self.load_candles(symbol=symbol, timeframe=timeframe, limit=limit)
        if not candles:
            return None
        return {
            "open_times": np.array([c.open_time for c in candles], dtype=np.int64),
            "opens": np.array([c.open for c in candles], dtype=np.float64),
            "highs": np.array([c.high for c in candles], dtype=np.float64),
            "lows": np.array([c.low for c in candles], dtype=np.float64),
            "closes": np.array([c.close for c in candles], dtype=np.float64),
            "volumes": np.array([c.volume for c in candles], dtype=np.float64),
            "taker_buy_volumes": np.array([c.taker_buy_volume for c in candles], dtype=np.float64),
            "quote_volumes": np.array([c.quote_volume for c in candles], dtype=np.float64),
            "trade_counts": np.array([c.trade_count for c in candles], dtype=np.float64),
        }

    def save_grid_walk_forward_results(
        self,
        records: List[GridWalkForwardRecordM2],
        publish_top_n: int = 5,
    ) -> None:
        if not records:
            return
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO grid_walk_forward_runs (
                        wf_id, created_at, symbol, timeframe, direction,
                        nominal_leverage, effective_max_leverage, grid_count, geometric_step_pct,
                        is_crosses, is_net_roi_pct, is_max_dd_pct,
                        oos_crosses, oos_net_roi_pct, oos_max_dd_pct,
                        full_crosses, full_net_pnl_usdt, full_net_roi_pct,
                        min_liq_buffer_pct, bayesian_survival_prob, wf_composite_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            r.wf_id,
                            now_iso,
                            r.symbol,
                            r.timeframe,
                            r.direction,
                            r.nominal_leverage,
                            r.effective_max_leverage,
                            r.grid_count,
                            r.geometric_step_pct,
                            r.is_crosses,
                            r.is_net_roi_pct,
                            r.is_max_dd_pct,
                            r.oos_crosses,
                            r.oos_net_roi_pct,
                            r.oos_max_dd_pct,
                            r.full_crosses,
                            r.full_net_pnl_usdt,
                            r.full_net_roi_pct,
                            r.min_liq_buffer_pct,
                            r.bayesian_survival_prob,
                            r.wf_composite_score,
                        )
                        for r in records
                    ],
                )

                sorted_recs = sorted(records, key=lambda x: x.wf_composite_score, reverse=True)
                seen_syms = set()
                top_unique: List[GridWalkForwardRecordM2] = []
                for rec in sorted_recs:
                    if rec.symbol not in seen_syms:
                        seen_syms.add(rec.symbol)
                        top_unique.append(rec)
                    if len(top_unique) >= publish_top_n:
                        break

                conn.execute("DELETE FROM active_grid_model_config;")
                for rank_idx, rec in enumerate(top_unique, start=1):
                    conn.execute(
                        """
                        INSERT INTO active_grid_model_config (
                            rank_order, updated_at, symbol, optimal_timeframe,
                            recommended_direction, optimal_grid_count, optimal_step_pct,
                            recommended_nominal_leverage, max_effective_leverage,
                            expected_daily_crosses, expected_days_6_to_18,
                            bayesian_survival_prob, wf_composite_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            rank_idx,
                            now_iso,
                            rec.symbol,
                            rec.timeframe,
                            rec.direction,
                            rec.grid_count,
                            rec.geometric_step_pct,
                            rec.nominal_leverage,
                            rec.effective_max_leverage,
                            round(rec.full_crosses / 30.0, 2),
                            round(200.0 / max(rec.full_net_roi_pct / 30.0, 0.1), 1),
                            rec.bayesian_survival_prob,
                            rec.wf_composite_score,
                        ),
                    )
        finally:
            self._close_if_disk(conn)

    def load_active_grid_configs(self) -> Dict[str, Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM active_grid_model_config ORDER BY rank_order ASC;"
            ).fetchall()
            return {str(r["symbol"]).upper(): dict(r) for r in rows}
        finally:
            self._close_if_disk(conn)

    def get_warehouse_summary(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            mtf_cnt = conn.execute("SELECT COUNT(*) FROM historical_klines_mtf;").fetchone()[0]
            k5m_cnt = conn.execute("SELECT COUNT(*) FROM historical_klines;").fetchone()[0]
            drv_cnt = conn.execute("SELECT COUNT(*) FROM historical_derivatives_metrics;").fetchone()[0]
            wf_cnt = conn.execute("SELECT COUNT(*) FROM grid_walk_forward_runs;").fetchone()[0]
            syms = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM historical_klines_mtf ORDER BY symbol ASC;"
                ).fetchall()
            ]
            return {
                "db_path": self.db_path,
                "total_mtf_candles": int(mtf_cnt),
                "total_5m_candles": int(k5m_cnt),
                "total_derivatives_rows": int(drv_cnt),
                "total_grid_wf_runs": int(wf_cnt),
                "symbols_stored": syms,
            }
        finally:
            self._close_if_disk(conn)


# ============================================================================
# 4. SERVICIO ETL ASÍNCRONO DE HISTÓRICOS DE BINANCE FUTURES USDⓈ-M
# ============================================================================

class Mark2BinanceHistoricalETLService:
    """
    Conector ETL asíncrono paginado hacia Binance Futures USDⓈ-M:
    Descarga y actualiza incrementalmente velas históricas (`/fapi/v1/klines`),
    tasas de financiación (`/fapi/v1/fundingRate`) e interés abierto (`/futures/data/openInterestHist`)
    hacia `MARK-2/prosperity_backtest_warehouse.db`.
    """

    KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
    FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
    TOP_LS_URL = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"

    def __init__(self, repo: Mark2SQLiteWarehouseRepository):
        self.repo = repo

    async def sync_symbol_klines_incremental(
        self,
        session: Any,
        symbol: str,
        timeframe: str = "5m",
        days_back: int = 30,
        max_batches: int = 12,
    ) -> int:
        """
        Sincroniza incrementalmente las velas históricas cerradas de `symbol` en `timeframe`.
        Si la base SQLite WAL ya cuenta con >= 90% de las velas esperadas y está fresca,
        descarga únicamente el delta faltante.
        """
        if session is None:
            return 0

        sym = symbol.upper()
        tf_min = TIMEFRAME_MINUTES.get(timeframe, 5)
        step_ms = tf_min * 60_000
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days_back * 86_400_000)
        expected_bars = max(10, int((now_ms - start_ms) // step_ms))

        cnt, min_ot, max_ot = self.repo.get_candle_bounds(sym, timeframe)
        if (
            cnt >= int(expected_bars * 0.90)
            and max_ot is not None
            and (now_ms - max_ot) <= (step_ms * 3)
        ):
            return cnt

        cursor_ms = (max_ot + step_ms) if (cnt > 0 and max_ot is not None and min_ot is not None and min_ot <= start_ms + step_ms * 10) else start_ms
        saved = 0
        batches = 0

        while cursor_ms < now_ms and batches < max_batches:
            batches += 1
            params = {
                "symbol": sym,
                "interval": timeframe,
                "startTime": cursor_ms,
                "endTime": now_ms,
                "limit": 1000,
            }
            try:
                async with session.get(self.KLINES_URL, params=params, timeout=6.0) as resp:
                    if resp.status in (429, 418):
                        wait_s = int(resp.headers.get("Retry-After", "3"))
                        await asyncio.sleep(wait_s)
                        continue
                    if resp.status != 200:
                        break
                    batch = await resp.json()
                    if not isinstance(batch, list) or not batch:
                        break
                    records = [
                        HistoricalCandleRecordM2(
                            symbol=sym,
                            timeframe=timeframe,
                            open_time=int(k[0]),
                            open=float(k[1]),
                            high=float(k[2]),
                            low=float(k[3]),
                            close=float(k[4]),
                            volume=float(k[5]),
                            taker_buy_volume=float(k[9]),
                            quote_volume=float(k[7]),
                            trade_count=int(k[8]),
                            taker_buy_quote_volume=float(k[10]),
                        )
                        for k in batch
                        if int(k[6]) < now_ms
                    ]
                    saved += self.repo.upsert_candles(records, is_synthetic=0)
                    last_open = int(batch[-1][0])
                    if last_open <= cursor_ms or len(batch) < 1000:
                        break
                    cursor_ms = last_open + step_ms
                    await asyncio.sleep(0.05)
            except Exception as exc:
                logger.debug("Aviso descargando klines históricos de %s: %s", sym, exc)
                break

        total_after, _, _ = self.repo.get_candle_bounds(sym, timeframe)
        return total_after
