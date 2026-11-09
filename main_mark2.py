# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — main_mark2.py (Versión Totalmente Actualizada)
Orquestador Principal Asíncrono End-to-End de Bots y Grids Geométricos de Futuros USDⓈ-M.

Flujo Operativo Integral:
1. Higiene y Pre-Flight Check: Carga .env, inicializa SQLite WAL y sincroniza el reloj con Binance.
2. Escáner Universal USDⓈ-M (M1 + M2): Analiza los mercados perpetuos de Binance desde el más
   seguro (Tier 1: BTC/ETH) hasta el más volátil (Tier 3/4: XRP, SUI, DOGE, HYPE, AKE, etc.).
3. Selección y Síntesis Matemática (M3 + M4): Filtra los mejores activos operables con la inversión
   mínima de $6.00 USD (verificando correlación cruzada rho < 0.70), autocalcula la dirección
   (NEUTRAL, LONG o SHORT), el rango (Bollinger + ATR), la densidad (2 a 169 grids), el apalancamiento
   (<= 15x en ISOLATED) y los niveles exactos de TP ($18.00 USD / +200% ROI) y SL (Precio, ROI y PnL).
4. Ejecución de Ventana Deslizante y Rebalanceo Infinito (M5): Coloca la ventana activa de órdenes
   LIMIT Maker, invierte escalones ejecutados (BUY <-> SELL), sincroniza en vivo el Trailing Grid
   ascendente/descendente con confirmación OBI L1, escala lotes en $9.00 y $12.00 USD, purga
   atómicamente al llegar a $18.00 USD y ejecuta mantenimiento automático de RAM y SQLite WAL.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import os
import signal
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

from core_mark2_contracts import (
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    Market3VTelemetryPOD,
    MarketSafetyTier,
    RankedMarketCandidatePOD,
)
from m1_universe_3v_ingestion import Mark2UniverseIngestionEngine
from m2_intelligence_regime_ml import Mark2IntelligenceEngine
from m3_grid_decision_orchestrator import Mark2GridDecisionOrchestrator
from m4_risk_treasury_6to18 import (
    BotEquityAction,
    Mark2RiskAndTreasuryController,
)
from m5_grid_execution_audit import (
    BinanceMark2LiveGridBroker,
    Mark2SlidingWindowGridExecutor,
)

# Opcional: si m0_workspace_hygiene.py está presente en el entorno, lo ejecuta al iniciar
try:
    import m0_workspace_hygiene  # type: ignore
except ImportError:
    m0_workspace_hygiene = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MAIN_MARK2")


# ============================================================================
# 1. CONTENEDOR DE CONTEXTO EN VIVO POR CADA BOT GRID ACTIVO ($6 -> $18 USD)
# ============================================================================

class LiveGridBotContext:
    """Agrupa en RAM el estado operativo, el ejecutor de ventana y los retornos de un Bot Grid."""

    __slots__ = (
        "symbol",
        "rules",
        "telemetry",
        "executor",
        "log_returns",
        "created_at_ts",
        "sim_wave_phase",
        "live_obi_l1",
    )

    def __init__(
        self,
        symbol: str,
        rules: ExchangeContractRulePOD,
        telemetry: Market3VTelemetryPOD,
        executor: Mark2SlidingWindowGridExecutor,
        log_returns: np.ndarray,
    ):
        self.symbol = symbol
        self.rules = rules
        self.telemetry = telemetry
        self.executor = executor
        self.log_returns = log_returns
        self.created_at_ts = time.time()
        self.sim_wave_phase = 0.0
        self.live_obi_l1 = float(telemetry.quote_obi_l1)


# ============================================================================
# 2. ORQUESTADOR ASÍNCRONO PRINCIPAL PROSPERITY MARK-2
# ============================================================================

class ProsperityMark2Orchestrator:
    """
    Coordina los módulos de MARK-2 para operar Bots Grid Geométricos de Futuros
    desde $6.00 USD hasta $18.00 USD (+200% ROI) y escalar entre las Fases 1 a 4.
    """

    PUBLIC_MAINNET_FAPI = "https://fapi.binance.com"

    def __init__(self, env_path: str = ".env"):
        if m0_workspace_hygiene is not None and hasattr(
            m0_workspace_hygiene, "enforce_workspace_hygiene"
        ):
            try:
                m0_workspace_hygiene.enforce_workspace_hygiene()
            except Exception:
                pass

        # 1. Inicializar Módulo 5 (Adaptador Binance USDⓈ-M + Lector .env + SQLite WAL)
        self.m5_broker = BinanceMark2LiveGridBroker(env_path=env_path)

        # Parámetros maestros leídos desde .env
        self.scan_mode = self.m5_broker.get_str(
            "UNIVERSE_SCAN_MODE", "ALL_USDM_PERPETUALS"
        ).upper()
        curated_raw = self.m5_broker.get_str(
            "CANDIDATE_UNIVERSE_SYMBOLS",
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,SUIUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,WIFUSDT",
        )
        self.curated_symbols = [
            s.strip().upper() for s in curated_raw.split(",") if s.strip()
        ]
        self.interval = self.m5_broker.get_str("KLINE_INTERVAL", "5m")
        self.warmup_candles = int(self.m5_broker.get_float("WARMUP_CANDLES", 200))
        self.max_scan_limit = int(self.m5_broker.get_float("MAX_SCAN_SYMBOLS_LIMIT", 25))

        self.initial_wallet_usdt = self.m5_broker.get_float("INITIAL_CAPITAL_USDT", 18.00)
        self.mxn_capital_ref = self.m5_broker.get_float("INITIAL_CAPITAL_MXN", 600.00)
        self.bot_min_usd = self.m5_broker.get_float("BOT_MIN_INVESTMENT_USDT", 6.00)
        self.bot_target_usd = self.m5_broker.get_float("BOT_TARGET_EQUITY_USDT", 18.00)
        self.max_concurrent_bots = int(
            self.m5_broker.get_float("MAX_CONCURRENT_GRID_BOTS", 3)
        )
        self.max_leverage_cap = int(self.m5_broker.get_float("MAX_LEVERAGE_CAP", 15))
        self.open_pos_on_create = (
            self.m5_broker.get_str("OPEN_POSITION_ON_CREATE", "TRUE").upper() == "TRUE"
        )
        self.enable_super_roi_trail = (
            self.m5_broker.get_str("ENABLE_SUPER_ROI_TRAIL", "FALSE").upper() == "TRUE"
        )

        maker_fee = self.m5_broker.get_float("MAKER_FEE_RATE", 0.00020)
        taker_fee = self.m5_broker.get_float("TAKER_FEE_RATE", 0.00050)
        max_corr_rho = self.m5_broker.get_float("MAX_CORRELATION_RHO", 0.70)
        max_daily_dd = self.m5_broker.get_float("MAX_DAILY_DRAWDOWN_PCT", 0.05)
        max_p99_ms = self.m5_broker.get_float("MAX_P99_LATENCY_MS", 180.0)

        # 2. Inicializar Módulo 3 (Cerebro Orquestador y Sintetizador de Grids Geométricos)
        self.m3_brain = Mark2GridDecisionOrchestrator(
            maker_fee=maker_fee, taker_fee=taker_fee
        )

        # 3. Inicializar Módulo 4 (Tesorería $6->$18 USD, Gobernador de Fases y Correlación)
        self.m4_treasury = Mark2RiskAndTreasuryController(
            initial_wallet_balance_usdt=self.initial_wallet_usdt,
            min_bot_capital_usdt=self.bot_min_usd,
            target_bot_capital_usdt=self.bot_target_usd,
            max_correlation_rho=max_corr_rho,
            max_daily_global_dd_pct=max_daily_dd,
            max_p99_latency_ms=max_p99_ms,
            enable_super_roi_trailing=self.enable_super_roi_trail,
        )

        # Catálogos en memoria RAM O(1)
        self.rules_catalog: Dict[str, ExchangeContractRulePOD] = {}
        self.telemetry_catalog: Dict[str, Market3VTelemetryPOD] = {}
        self.returns_catalog: Dict[str, np.ndarray] = {}
        self.active_bots: Dict[str, LiveGridBotContext] = {}

        self._running = True
        self._execution_lock = asyncio.Lock()
        self._last_maintenance_ts = time.time()

    # ------------------------------------------------------------------------
    # A. HIGIENE DE MEMORIA RAM Y MANTENIMIENTO SQLITE WAL
    # ------------------------------------------------------------------------

    def perform_memory_and_wal_maintenance(self, truncate_wal: bool = False) -> None:
        """
        Libera objetos cíclicos no referenciados en RAM y ejecuta un checkpoint
        sobre prosperity_mark2_journal.db para evitar crecimiento de archivos -wal y -shm.
        """
        try:
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

            mode = "TRUNCATE" if truncate_wal else "PASSIVE"
            db_file = str(self.m5_broker.db_path)
            if os.path.exists(db_file):
                with sqlite3.connect(db_file, timeout=5.0) as conn:
                    conn.execute(f"PRAGMA wal_checkpoint({mode});")
        except Exception as exc:
            logger.debug("Aviso en mantenimiento RAM/WAL: %s", exc)

    def _increment_trailing_shift_in_db(self, symbol: str) -> None:
        """Registra en SQLite WAL cada vez que el Trailing Grid desplaza el rango activo."""
        try:
            db_file = str(self.m5_broker.db_path)
            if os.path.exists(db_file):
                with sqlite3.connect(db_file, timeout=5.0) as conn:
                    conn.execute(
                        """
                        UPDATE mark2_bot_sessions
                        SET trailing_shifts = COALESCE(trailing_shifts, 0) + 1
                        WHERE symbol = ? AND close_reason IN ('RUNNING', 'ACTIVE')
                        """,
                        (symbol,),
                    )
                    conn.commit()
        except Exception as db_err:
            logger.debug("Aviso de auditoría trailing en SQLite: %s", db_err)

    # ------------------------------------------------------------------------
    # B. CLIENTE HTTP RESILIENTE CON FALLBACK LOCAL (INMUNE A GAIERROR 11001)
    # ------------------------------------------------------------------------

    async def _safe_http_get_json(
        self,
        session: Optional[Any],
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Realiza peticiones GET públicas hacia Binance con medición de latencia RTT
        y captura total de errores de red/DNS (socket.gaierror 11001) sin tumbar el bot.
        """
        if session is None:
            return None
        for attempt in range(1, 3):
            t0 = time.perf_counter()
            try:
                async with session.get(url, params=params, timeout=8) as resp:
                    rtt_ms = (time.perf_counter() - t0) * 1000.0
                    self.m4_treasury.record_network_rtt_ms(rtt_ms)
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in (429, 418):
                        await asyncio.sleep(3.0)
            except Exception as exc:
                if attempt == 2:
                    logger.debug(
                        "🌐 [Red Local / DNS Fallback] %s no disponible (%s).", url, exc
                    )
                await asyncio.sleep(1.0)
        return None

    # ------------------------------------------------------------------------
    # C. ESCÁNER UNIVERSAL USDⓈ-M (DEL MÁS SEGURO AL MÁS VOLÁTIL)
    # ------------------------------------------------------------------------

    async def sync_universe_and_telemetry(self, session: Optional[Any]) -> None:
        """
        Descarga /fapi/v1/exchangeInfo, /fapi/v1/ticker/24hr y /fapi/v1/klines de Binance
        para clasificar todos los mercados del más seguro al más volátil.
        Si no hay internet local, utiliza el catálogo determinista de respaldo en RAM.
        """
        ex_info = await self._safe_http_get_json(
            session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/exchangeInfo"
        )
        tickers_24h = await self._safe_http_get_json(
            session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/24hr"
        )
        book_tickers = await self._safe_http_get_json(
            session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/bookTicker"
        )

        if ex_info and isinstance(ex_info, dict) and "symbols" in ex_info:
            self.rules_catalog = Mark2UniverseIngestionEngine.parse_binance_exchange_info(
                ex_info
            )
            self.m5_broker.contract_rules = self.rules_catalog

        if not self.rules_catalog:
            self._load_offline_fallback_universe()
            return

        # Mapear volumen 24h y cotizaciones L1 Bid/Ask de todos los perpetuos
        vol_map: Dict[str, float] = {}
        if isinstance(tickers_24h, list):
            for item in tickers_24h:
                sym = item.get("symbol", "")
                if sym in self.rules_catalog:
                    vol_map[sym] = float(item.get("quoteVolume", 0.0))

        l1_map: Dict[str, Tuple[float, float, float, float]] = {}
        if isinstance(book_tickers, list):
            for b in book_tickers:
                sym = b.get("symbol", "")
                if sym in self.rules_catalog:
                    bid = float(b.get("bidPrice", 0.0))
                    ask = float(b.get("askPrice", 0.0))
                    bid_q = float(b.get("bidQty", 0.0))
                    ask_q = float(b.get("askQty", 0.0))
                    if Mark2UniverseIngestionEngine.validate_l1_quote_o1(
                        bid, ask, bid_q, ask_q
                    ):
                        l1_map[sym] = (bid, ask, bid_q, ask_q)

        # Seleccionar los símbolos a evaluar: Curated + Top por liquidez y rotación 24h
        candidate_pool: List[str] = [
            s for s in self.curated_symbols if s in self.rules_catalog
        ]
        if self.scan_mode == "ALL_USDM_PERPETUALS" and vol_map:
            sorted_by_vol = sorted(
                vol_map.keys(), key=lambda k: vol_map[k], reverse=True
            )
            for s in sorted_by_vol:
                if s not in candidate_pool and len(candidate_pool) < self.max_scan_limit:
                    candidate_pool.append(s)

        # Descargar velas recientes para construir el perfil 3V de cada candidato
        for sym in candidate_pool[: self.max_scan_limit]:
            klines = await self._safe_http_get_json(
                session,
                f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/klines",
                {"symbol": sym, "interval": self.interval, "limit": self.warmup_candles},
            )
            if not isinstance(klines, list) or len(klines) < 30:
                continue

            closes = np.array([float(k[4]) for k in klines], dtype=np.float64)
            highs = np.array([float(k[2]) for k in klines], dtype=np.float64)
            lows = np.array([float(k[3]) for k in klines], dtype=np.float64)
            vols = np.array([float(k[5]) for k in klines], dtype=np.float64)

            p_last = float(closes[-1])
            bid, ask, bid_q, ask_q = l1_map.get(
                sym, (p_last * 0.9998, p_last * 1.0002, 1000.0, 1000.0)
            )
            q_vol = vol_map.get(sym, 100_000_000.0)

            tel = Mark2IntelligenceEngine.compute_telemetry_from_ohlcv(
                symbol=sym,
                closes=closes,
                highs=highs,
                lows=lows,
                volumes=vols,
                bid=bid,
                ask=ask,
                bid_qty=bid_q,
                ask_qty=ask_q,
                mark_price=p_last,
                quote_vol_24h=q_vol,
                timestamp_ms=int(time.time() * 1000),
            )
            self.telemetry_catalog[sym] = tel
            self.returns_catalog[sym] = np.diff(np.log(np.maximum(closes, 1e-10)))
            await asyncio.sleep(0.05)

        if not self.telemetry_catalog:
            self._load_offline_fallback_universe()

    def _load_offline_fallback_universe(self) -> None:
        """
        Respaldo determinista en memoria por si se ejecuta sin internet local,
        permitiendo verificar toda la cadena desde BTCUSDT hasta SUIUSDT/XRPUSDT.
        """
        raw_mock = {
            "symbols": [
                {
                    "symbol": "BTCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "100.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
                {
                    "symbol": "ETHUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "20.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
                {
                    "symbol": "BNBUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
                {
                    "symbol": "XRPUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
                {
                    "symbol": "DOGEUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.00001"},
                        {"filterType": "LOT_SIZE", "stepSize": "1.0", "minQty": "1.0"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
                {
                    "symbol": "SUIUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "MAX_NUM_ORDERS", "limit": 200},
                    ],
                },
            ]
        }
        self.rules_catalog = Mark2UniverseIngestionEngine.parse_binance_exchange_info(
            raw_mock
        )
        self.m5_broker.contract_rules = self.rules_catalog

        t = np.linspace(0, 10 * np.pi, 240)
        presets = [
            ("BTCUSDT", 85000.0, 0.30, 2_900_000_000.0, 0.0),
            ("ETHUSDT", 2950.0, 0.45, 1_600_000_000.0, 0.5),
            ("BNBUSDT", 640.0, 0.65, 420_000_000.0, 1.1),
            ("XRPUSDT", 2.4200, 1.15, 810_000_000.0, 2.2),
            ("DOGEUSDT", 0.26500, 1.38, 690_000_000.0, 3.4),
            ("SUIUSDT", 3.1800, 1.62, 540_000_000.0, 4.7),
        ]
        for sym, base_p, amp, vol24, phase_shift in presets:
            closes = base_p * (1.0 + (amp / 100.0) * np.sin(t + phase_shift))
            highs = closes * (1.0 + (amp / 170.0))
            lows = closes * (1.0 - (amp / 170.0))
            vols = np.full(240, 12000.0)
            self.telemetry_catalog[sym] = (
                Mark2IntelligenceEngine.compute_telemetry_from_ohlcv(
                    symbol=sym,
                    closes=closes,
                    highs=highs,
                    lows=lows,
                    volumes=vols,
                    bid=float(closes[-1] * 0.9999),
                    ask=float(closes[-1] * 1.0001),
                    bid_qty=6500.0,
                    ask_qty=5100.0,
                    mark_price=float(closes[-1]),
                    quote_vol_24h=vol24,
                    timestamp_ms=int(time.time() * 1000),
                )
            )
            self.returns_catalog[sym] = np.diff(np.log(closes))

    # ------------------------------------------------------------------------
    # D. IMPRESIÓN DE TABLA UNIVERSAL Y FICHA ESTILO APP DE BINANCE
    # ------------------------------------------------------------------------

    def print_universe_ranking_table(
        self, ranked_list: List[RankedMarketCandidatePOD]
    ) -> None:
        print("\n" + "=" * 112)
        print(
            "🌍 ESCÁNER UNIVERSAL USDⓈ-M MARK-2 (CLASIFICACIÓN DEL MÁS SEGURO AL MÁS VOLÁTIL PARA BOT DE $6.00 USD)"
        )
        print("=" * 112)
        print(
            f"{'ORD':<4} | {'MERCADO':<10} | {'ESTRATO DE RIESGO':<22} | {'IVR':<6} | "
            f"{'LOTE MÍN ($)':<12} | {'CAP MÍN GRID':<12} | {'CRUCES/D':<8} | {'DÍAS $6->$18':<12} | {'ESTADO PARA $6 USD'}"
        )
        print("-" * 112)
        for idx, c in enumerate(ranked_list, start=1):
            badge = "✅ APTO ($6 OK)" if c.viable_for_6usd_bot else "🔒 BLOQUEADO"
            print(
                f"#{idx:<3} | {c.symbol:<10} | {c.safety_tier.value:<22} | {c.risk_score_ivr:<6.4f} | "
                f"${c.min_line_notional_usdt:>10.2f} | ${c.min_capital_for_grid_usdt:>10.2f} | "
                f"{c.est_daily_crosses:>7.1f} | {c.est_days_6_to_18_usd:>10.1f}d | {badge}"
            )
        print("=" * 112)

    @staticmethod
    def format_binance_bot_details_card(
        plan: GeometricGridPlanPOD,
        matched_24h: int,
        matched_total: int,
        mdd_7d_pct: float,
        runtime_sec: float = 0.0,
    ) -> str:
        """
        Genera en consola exactamente la misma ficha 'Detalles del bot / Grid de futuros'
        de la interfaz de Binance Futures.
        """
        days = int(runtime_sec) // 86400
        hours = (int(runtime_sec) % 86400) // 3600
        mins = (int(runtime_sec) % 3600) // 60

        pnl_now = round(plan.current_equity_usdt - plan.initial_capital_usdt, 2)
        roi_now = round((pnl_now / max(plan.initial_capital_usdt, 0.01)) * 100.0, 2)

        range_span_pct = (
            (plan.upper_price - plan.lower_price) / max(plan.lower_price, 1e-9)
        ) * 100.0
        density_grids_pct = round(
            plan.total_virtual_grids / max(range_span_pct, 0.01), 2
        )

        return (
            f"\n┌──────────────────────────────────────────────────────────────────────────┐\n"
            f"│ 🤖 DETALLES DEL BOT GRID DE FUTUROS (MARK-2) — {plan.symbol:<10} [{plan.direction.value} {plan.leverage}x] │\n"
            f"├──────────────────────────────────────────────────────────────────────────┤\n"
            f"│ • PnL (USD) / ROI Actual : {pnl_now:+.2f} USD ({roi_now:+.2f}%) | Equidad: ${plan.current_equity_usdt:.2f} USD\n"
            f"│ • Tiempo de ejecución    : {days}d {hours}h {mins}m | Tramo: {plan.compounding_stage.value}\n"
            f"│ • Inversión mín. / Meta  : ${plan.initial_capital_usdt:.2f} USDT ──► Meta Cierre: ${plan.target_equity_usdt:.2f} USDT (+200% ROI)\n"
            f"│ • Total trades emparejados: {matched_24h}/{matched_total} (24h/Total) | MDD en 7D: {mdd_7d_pct:.2f}%\n"
            f"│ • Dirección / Margen     : {plan.direction.value} | ISOLATED {plan.leverage}x (<= 15x)\n"
            f"│ • Rango de precios (USDT): {plan.lower_price} - {plan.upper_price}\n"
            f"│ • Cant. de grids         : {plan.total_virtual_grids} niveles ({plan.active_window_orders} órdenes activas en libro)\n"
            f"│ • Modo / Lote por línea  : Geométrico | Qty={plan.qty_per_line} (${plan.notional_per_line_usdt:.2f} USDT)\n"
            f"│ • Ganancia/grid (neta)   : {plan.net_profit_per_cross_pct:.3f}% (+${plan.net_profit_per_cross_usdt:.4f} USDT/cruce desc. fee)\n"
            f"│ • Densidad de grid       : {density_grids_pct} grids/% | Paso geométrico: {plan.geometric_step_pct:.3f}%\n"
            f"│ • Trailing ascendente    : {'Activado' if plan.trailing_up else 'Desactivado'}\n"
            f"│ • Trailing descendente   : {'Activado' if plan.trailing_down else 'Desactivado'}\n"
            f"│ • Stop Loss (Precio/Piso): {plan.sl_price} USDT | Piso Ratchet: ${plan.equity_ratchet_floor_usdt:.2f} USD ({plan.sl_roi_pct:+.1f}% ROI)\n"
            f"│ • Take Profit (Meta 3x)  : {plan.tp_price} USDT | +${plan.tp_pnl_usdt:.2f} USD (+{plan.tp_roi_pct:.0f}% ROI)\n"
            f"│ • Abrir pos. en creación : {'Activado' if plan.direction != GridDirection.NEUTRAL else 'Neutral (Límites)'}\n"
            f"└──────────────────────────────────────────────────────────────────────────┘"
        )

    # ------------------------------------------------------------------------
    # E. DESPLIEGUE DE BOTS GRID ($6.00 USD -> $18.00 USD)
    # ------------------------------------------------------------------------

    async def deploy_eligible_grid_bots(self, session: Optional[Any]) -> None:
        """
        Clasifica todos los mercados del más seguro al más volátil, selecciona los
        óptimos para el capital actual de la Fase y despliega los Bots Grid en Binance.
        """
        phase_profile = self.m4_treasury.resolve_phase_behavior_profile()
        bot_cap = phase_profile.capital_allocation_per_bot_usdt

        ranked_universe = self.m3_brain.rank_all_markets_safe_to_volatile(
            rules_map=self.rules_catalog,
            telemetry_map=self.telemetry_catalog,
            bot_capital_usdt=bot_cap,
            max_leverage=min(self.max_leverage_cap, phase_profile.max_leverage),
        )
        self.print_universe_ranking_table(ranked_universe)

        # Ordenar los candidatos viables priorizando menor tiempo estimado hacia $18.00 USD
        viable_candidates = [
            c
            for c in ranked_universe
            if c.viable_for_6usd_bot and c.symbol not in self.active_bots
        ]
        viable_candidates.sort(key=lambda c: (c.est_days_6_to_18_usd, c.risk_score_ivr))

        max_slots = min(self.max_concurrent_bots, phase_profile.max_concurrent_bots)

        for cand in viable_candidates:
            if len(self.active_bots) >= max_slots:
                break

            active_returns_map = {
                sym: ctx.log_returns for sym, ctx in self.active_bots.items()
            }
            cand_returns = self.returns_catalog.get(cand.symbol, np.zeros(30))

            verdict = self.m4_treasury.authorize_new_grid_bot_deployment(
                candidate_symbol=cand.symbol,
                active_bots_count=len(self.active_bots),
                candidate_returns=cand_returns,
                active_bots_returns=active_returns_map,
            )

            if not verdict.approved:
                logger.info("🛡️ [M4 Aduana en %s]: %s", cand.symbol, verdict.reason)
                continue

            rules = self.rules_catalog[cand.symbol]
            tel = self.telemetry_catalog[cand.symbol]
            target_eq = round(
                verdict.assigned_capital_usdt
                * (1.0 + phase_profile.target_roi_per_bot_pct / 100.0),
                2,
            )

            plan = self.m3_brain.build_geometric_grid_6_to_18(
                rules=rules,
                tel=tel,
                safety_tier=cand.safety_tier,
                initial_capital_usdt=verdict.assigned_capital_usdt,
                current_equity_usdt=verdict.assigned_capital_usdt,
                target_equity_usdt=target_eq,
            )

            # Despacho al Broker M5 (ISOLATED + Posición Inicial + Ventana LIMIT + SL/TP MARK_PRICE)
            await self.m5_broker.configure_isolated_and_leverage(
                cand.symbol, plan.leverage, session=session
            )
            if self.open_pos_on_create and plan.direction != GridDirection.NEUTRAL:
                await self.m5_broker.open_initial_base_position(
                    symbol=cand.symbol,
                    direction=plan.direction,
                    base_qty=plan.qty_per_line,
                    rules=rules,
                    session=session,
                )

            await self.m5_broker.sync_active_limit_window(
                symbol=cand.symbol,
                buy_levels=plan.active_buy_window,
                sell_levels=plan.active_sell_window,
                qty=plan.qty_per_line,
                rules=rules,
                session=session,
            )
            await self.m5_broker.place_global_protection_brackets(
                symbol=cand.symbol,
                direction=plan.direction,
                sl_price=plan.sl_price,
                tp_price=plan.tp_price,
                current_mark_price=tel.mark_price,
                rules=rules,
                session=session,
            )

            executor = Mark2SlidingWindowGridExecutor(
                plan=plan,
                rules=rules,
                journal_db_path=self.m5_broker.db_path,
            )
            self.active_bots[cand.symbol] = LiveGridBotContext(
                symbol=cand.symbol,
                rules=rules,
                telemetry=tel,
                executor=executor,
                log_returns=cand_returns,
            )
            self.m4_treasury.correlation_guard.register_active_bot(
                cand.symbol, plan.direction
            )

            print(
                self.format_binance_bot_details_card(
                    plan=plan, matched_24h=0, matched_total=0, mdd_7d_pct=0.0
                )
            )

    # ------------------------------------------------------------------------
    # F. CORRUTINAS ASÍNCRONAS DE EMPAREJAMIENTO, TRAILING Y TRINQUETE M4
    # ------------------------------------------------------------------------

    async def pre_flight_check(self, session: Optional[Any]) -> None:
        logger.info("==========================================================================")
        logger.info(
            "🚀 INICIANDO PROSPERITY MARK-2 | Modo: %s (DRY_RUN=%s) | Meta Bot: $%.2f -> $%.2f USD",
            self.m5_broker.env_mode,
            self.m5_broker.dry_run,
            self.bot_min_usd,
            self.bot_target_usd,
        )
        logger.info("==========================================================================")

        self.perform_memory_and_wal_maintenance(truncate_wal=False)

        if session is not None:
            await self.m5_broker.sync_server_time(session)
            real_bal = await self.m5_broker.fetch_usdt_wallet_balance(session)
            self.m4_treasury.reconcile_wallet_with_exchange(real_bal)

        phase_prof = self.m4_treasury.resolve_phase_behavior_profile()
        logger.info(
            "💰 [TESORERÍA M4] Balance Billetera: $%.2f USDT (~$%.2f MXN) | Fase Activa: %s | Max Lev: %dx",
            self.m4_treasury.wallet_balance_usdt,
            self.m4_treasury.wallet_balance_usdt
            * (self.mxn_capital_ref / max(self.initial_wallet_usdt, 1.0)),
            phase_prof.phase.value,
            phase_prof.max_leverage,
        )

        await self.sync_universe_and_telemetry(session)
        await self.deploy_eligible_grid_bots(session)

    async def grid_hot_path_matching_loop(self, session: Optional[Any]) -> None:
        """
        Bucle de Ruta Caliente:
        Evalúa en vivo las oscilaciones de precio en cada Bot Grid activo, empareja
        compras y ventas (BUY <-> SELL) y ejecuta el Rebalanceo Infinito (Trailing Grid
        Ascendente y Descendente) con sincronización de órdenes en Binance.
        """
        while self._running:
            try:
                await asyncio.sleep(2.5)
                if not self.active_bots:
                    continue

                async with self._execution_lock:
                    for sym, ctx in list(self.active_bots.items()):
                        ex = ctx.executor

                        if self.m5_broker.dry_run:
                            # Modo Simulación: Empareja escalón inmediato y prueba Trailing si aplica
                            if ex.active_buys:
                                buy_lv = ex.active_buys[-1]
                                flip = ex.on_limit_order_filled("BUY", buy_lv)
                                sell_lv = float(flip["new_order_price"])
                                harvest = ex.on_limit_order_filled("SELL", sell_lv)
                                logger.info(
                                    "⚡ [CRUCE EMPAREJADO %s | %s] BUY @ %s -> SELL @ %s | Neto: +$%.4f USDT | Equidad Bot: $%.2f USD",
                                    sym,
                                    ex.plan.compounding_stage.value,
                                    buy_lv,
                                    sell_lv,
                                    float(harvest["net_harvest_usdt"]),
                                    float(harvest["cumulative_equity_usdt"]),
                                )
                        else:
                            # Modo LIVE / SANDBOX con órdenes reales: consulta bookTicker en vivo
                            bt = await self._safe_http_get_json(
                                session,
                                f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/bookTicker",
                                {"symbol": sym},
                            )
                            if isinstance(bt, dict) and "bidPrice" in bt:
                                live_bid = float(bt["bidPrice"])
                                live_ask = float(bt["askPrice"])
                                bid_qty = float(bt.get("bidQty", 0.0))
                                ask_qty = float(bt.get("askQty", 0.0))
                                live_mid = (live_bid + live_ask) * 0.5

                                # Actualizar Quote OBI L1 en vivo para validar rupturas de Trailing
                                total_l1_qty = bid_qty + ask_qty
                                if total_l1_qty > 0.0:
                                    ctx.live_obi_l1 = (bid_qty - ask_qty) / total_l1_qty

                                # 1. Verificar si el precio tocó algún nivel BUY inferior
                                for b_price in list(ex.active_buys):
                                    if live_bid <= b_price:
                                        flip = ex.on_limit_order_filled("BUY", b_price)
                                        await self.m5_broker.place_single_counter_limit_order(
                                            symbol=sym,
                                            side="SELL",
                                            price=float(flip["new_order_price"]),
                                            qty=ex.plan.qty_per_line,
                                            rules=ctx.rules,
                                            reduce_only=True,
                                            session=session,
                                        )

                                # 2. Verificar si el precio tocó algún nivel SELL superior
                                for s_price in list(ex.active_sells):
                                    if live_ask >= s_price:
                                        flip = ex.on_limit_order_filled("SELL", s_price)
                                        await self.m5_broker.place_single_counter_limit_order(
                                            symbol=sym,
                                            side="BUY",
                                            price=float(flip["new_order_price"]),
                                            qty=ex.plan.qty_per_line,
                                            rules=ctx.rules,
                                            reduce_only=False,
                                            session=session,
                                        )

                                # 3. Evaluar y Ejecutar Trailing Grid Ascendente / Descendente
                                trail_evt = ex.check_and_apply_trailing_grid(
                                    current_price=live_mid,
                                    quote_obi_l1=ctx.live_obi_l1,
                                )
                                if trail_evt is not None:
                                    action = str(trail_evt.get("trailing_action", "TRAILING_SHIFT"))
                                    new_lower = float(trail_evt["new_range_lower"])
                                    new_upper = float(trail_evt["new_range_upper"])
                                    logger.info(
                                        "🔄 [TRAILING GRID ACTIVADO | %s] %s | Nuevo Rango: [%.4f - %.4f] | OBI L1: %+.2f",
                                        sym,
                                        action,
                                        new_lower,
                                        new_upper,
                                        ctx.live_obi_l1,
                                    )

                                    # Sincronizar órdenes físicas en Binance tras desplazar el canal
                                    if hasattr(self.m5_broker, "cancel_all_symbol_orders"):
                                        await self.m5_broker.cancel_all_symbol_orders(
                                            sym, session=session
                                        )
                                        await asyncio.sleep(0.15)

                                    await self.m5_broker.sync_active_limit_window(
                                        symbol=sym,
                                        buy_levels=ex.active_buys,
                                        sell_levels=ex.active_sells,
                                        qty=ex.plan.qty_per_line,
                                        rules=ctx.rules,
                                        session=session,
                                    )
                                    await self.m5_broker.place_global_protection_brackets(
                                        symbol=sym,
                                        direction=ex.plan.direction,
                                        sl_price=ex.plan.sl_price,
                                        tp_price=ex.plan.tp_price,
                                        current_mark_price=live_mid,
                                        rules=ctx.rules,
                                        session=session,
                                    )
                                    self._increment_trailing_shift_in_db(sym)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("❌ [Error en Grid Hot-Path Loop]: %s", exc)
                await asyncio.sleep(2.0)

    async def treasury_ratchet_and_audit_loop(self, session: Optional[Any]) -> None:
        """
        Bucle Soberano de Tesorería M4:
        Vigila cada 4 segundos la transición por los Tramos ($6 -> $9 -> $12 -> $18 USD),
        ajusta la ventana activa al subir de tramo, ejecuta el cierre atómico al llegar a $18.00 USD
        y corre el mantenimiento periódico de RAM y SQLite WAL.
        """
        while self._running:
            try:
                await asyncio.sleep(4.0)

                # Mantenimiento automático de RAM y SQLite WAL cada 300 segundos (5 min)
                now_ts = time.time()
                if now_ts - self._last_maintenance_ts >= 300.0:
                    self.perform_memory_and_wal_maintenance(truncate_wal=False)
                    self._last_maintenance_ts = now_ts

                if not self.active_bots:
                    continue

                async with self._execution_lock:
                    for sym, ctx in list(self.active_bots.items()):
                        ex = ctx.executor
                        action, new_plan = self.m4_treasury.evaluate_bot_equity_transition(
                            current_plan=ex.plan,
                            realized_pnl_usdt=ex.realized_pnl_usdt,
                            unrealized_pnl_usdt=ex.unrealized_pnl_usdt,
                            rules=ctx.rules,
                            tel=ctx.telemetry,
                            orchestrator=self.m3_brain,
                        )

                        # 1. ¿Subió de Tramo ($6->$9 o $9->$12 USD)?
                        if action == BotEquityAction.STAGE_UPGRADED_RESIZE.value:
                            ex.plan = new_plan
                            ex.active_buys = list(new_plan.active_buy_window)
                            ex.active_sells = list(new_plan.active_sell_window)
                            await self.m5_broker.sync_active_limit_window(
                                symbol=sym,
                                buy_levels=new_plan.active_buy_window,
                                sell_levels=new_plan.active_sell_window,
                                qty=new_plan.qty_per_line,
                                rules=ctx.rules,
                                session=session,
                            )
                            t_24h, t_tot = ex.get_matched_trades_24h_and_total()
                            print(
                                self.format_binance_bot_details_card(
                                    plan=new_plan,
                                    matched_24h=t_24h,
                                    matched_total=t_tot,
                                    mdd_7d_pct=ex.compute_mdd_7d_pct(),
                                    runtime_sec=time.time() - ctx.created_at_ts,
                                )
                            )

                        # 2. ¿Alcanzó la Meta de $18.00 USD (+200% ROI) o tocó el Piso Ratchet?
                        elif action in (
                            BotEquityAction.TAKE_PROFIT_18_REACHED.value,
                            BotEquityAction.RATCHET_STOP_TRIGGERED.value,
                        ):
                            ex.plan = new_plan
                            await self.m5_broker.atomic_close_and_purge_bot(
                                symbol=sym, reason=action, session=session
                            )
                            t_24h, t_tot = ex.get_matched_trades_24h_and_total()
                            print(
                                self.format_binance_bot_details_card(
                                    plan=new_plan,
                                    matched_24h=t_24h,
                                    matched_total=t_tot,
                                    mdd_7d_pct=ex.compute_mdd_7d_pct(),
                                    runtime_sec=time.time() - ctx.created_at_ts,
                                )
                            )

                            self.m4_treasury.register_bot_session_closed(
                                symbol=sym, net_realized_pnl_usdt=ex.realized_pnl_usdt
                            )
                            self.active_bots.pop(sym, None)
                            self.perform_memory_and_wal_maintenance(truncate_wal=True)

                            new_phase = (
                                self.m4_treasury.resolve_phase_behavior_profile().phase.value
                            )
                            logger.info(
                                "🏆 [BOT %s FINALIZADO: %s] PnL Neto: %+.2f USD | Nuevo Balance Billetera: $%.2f USDT | Fase: %s",
                                sym,
                                action,
                                ex.realized_pnl_usdt,
                                self.m4_treasury.wallet_balance_usdt,
                                new_phase,
                            )

                            # En DRY_RUN detenemos el ciclo demostrativo cuando todos los bots cumplen su meta $18 USD
                            if self.m5_broker.dry_run and not self.active_bots:
                                logger.info(
                                    "✅ [CICLO DEMOSTRATIVO $6 -> $18 USD COMPLETADO EN TODOS LOS BOTS ACTIVOS]"
                                )
                                self.stop()
                                break
                            elif not self.m5_broker.dry_run:
                                # En operación continua, re-escanear y desplegar el siguiente Bot Grid elegible
                                await self.sync_universe_and_telemetry(session)
                                await self.deploy_eligible_grid_bots(session)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("⚠️ [Aviso en Bucle de Tesorería M4]: %s", exc)

    async def run(self) -> None:
        if aiohttp is not None:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                await self._run_with_session(session)
        else:
            await self._run_with_session(None)

    async def _run_with_session(self, session: Optional[Any]) -> None:
        await self.pre_flight_check(session)
        tasks = [
            asyncio.create_task(self.grid_hot_path_matching_loop(session)),
            asyncio.create_task(self.treasury_ratchet_and_audit_loop(session)),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                t.cancel()
            self.perform_memory_and_wal_maintenance(truncate_wal=True)
            logger.info("🛑 [PROSPERITY MARK-2] Apagado limpio y compactación WAL completados.")

    def stop(self) -> None:
        self._running = False


def main() -> None:
    orchestrator = ProsperityMark2Orchestrator(env_path=".env")

    def _sig_handler(*_: object) -> None:
        orchestrator.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()