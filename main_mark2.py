# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — main_mark2.py
Orquestador Principal Asíncrono End-to-End de Bots y Grids Geométricos de Futuros USDⓈ-M.

Flujo Operativo Integral (100% Basado en Datos Reales de la API de Binance):
1. Higiene y Pre-Flight Check: Carga .env, inicializa SQLite WAL, sincroniza el reloj con Binance
   (/fapi/v1/time), consulta comisiones reales (/fapi/v1/commissionRate) y balance (/fapi/v2/balance).
2. Escáner Universal Concurrente USDⓈ-M (M1 + M2): Descarga en paralelo con TCP Keep-Alive y caché DNS
   /fapi/v1/exchangeInfo, /fapi/v1/ticker/24hr, /fapi/v1/ticker/bookTicker, /fapi/v1/premiumIndex y
   /fapi/v1/klines (incluyendo volumen Taker real k[9]).
3. Mitigación de Incertidumbre y Gobernanza en 5 Fases (M3 + M4):
   - Fase 1 (4 a 100 USDT): Mitigar incertidumbre con datos de Binance + Triplicar cuenta (3x) en 30 días.
   - Fase 2 (100 a 1,000 USDT): Expansión multiactivo + Triplicar cuenta (3x) en 30 días.
   - Fase 3 (1,000 a 10,000 USDT): Aceleración estructural + Duplicar cuenta (2x) en 30 días.
   - Fase 4 (10,000 a 100,000 USDT): Consolidación institucional + Retiro mensual 20%.
   - Fase 5 (100,000 a 1,000,000 USDT): Preservación patrimonial soberana.
   Calcula Certidumbre (%), Apalancamiento/Riesgo de la Fase, ROI/PnL Esperado Diario y a 30 Días de cada
   mercado y los Días Estimados (3x en Fases 1 y 2; 2x en Fase 3).
4. Ejecución de Ventana Deslizante y Rebalanceo Infinito (M5): Monitorea en tiempo real las cotizaciones
   L1 y el Mark Price de Binance para emparejar compras y ventas reales, aplicar Trailing Grid, escalar
   por Tramos y cerrar atómicamente al cumplir el ROI/PnL esperado del mercado o activar el Escudo Anti-Liq.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

from collections import deque
import dataclasses

from core_mark2_contracts import (
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    HighRiskDDMStatus,
    Market3VTelemetryPOD,
    MarketSafetyTier,
    RankedMarketCandidatePOD,
)
from m1_historical_sqlite_warehouse import (
    HistoricalCandleRecordM2,
    HistoricalDerivativeRecordM2,
    Mark2BinanceHistoricalETLService,
    Mark2SQLiteWarehouseRepository,
)
from m1_universe_3v_ingestion import Mark2UniverseIngestionEngine
from m2_intelligence_regime_ml import Mark2IntelligenceEngine
from m3_alpha_high_roi_grid_engine import AlphaHighROIGridCalibrator
from m3_grid_decision_orchestrator import Mark2GridDecisionOrchestrator
from m4_risk_treasury_6to18 import (
    BotEquityAction,
    Mark2RiskAndTreasuryController,
)
from m5_grid_execution_audit import (
    BinanceMark2LiveGridBroker,
    Mark2SlidingWindowGridExecutor,
)
from m6_binance_bot_sheet_inspector import Mark2BotSheetMonitor
from m7_emergency_red_button import Mark2SurgicalRedButtonService
from m8_high_risk_alpha_accelerator import (
    ExactExchangeQuantizer,
    Mark2DualDDMController,
    Mark2HighRiskAlphaAccelerator,
)

try:
    import m0_workspace_hygiene  # type: ignore
except ImportError:
    m0_workspace_hygiene = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MAIN_MARK2")


# ============================================================================
# 1. CONTENEDOR DE CONTEXTO EN VIVO POR CADA BOT GRID ACTIVO
# ============================================================================

class LiveGridBotContext:
    """Agrupa en RAM el estado operativo, el ejecutor de ventana y los retornos reales de un Bot Grid."""

    __slots__ = (
        "symbol",
        "rules",
        "telemetry",
        "executor",
        "log_returns",
        "created_at_ts",
        "sheet_monitor",
        "ddm_controller",
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
        # Preserva el timestamp de inicio rehidratado desde SQLite WAL (o time.time() si es sesión nueva)
        self.created_at_ts = float(executor.start_timestamp_utc or time.time())
        self.sheet_monitor = Mark2BotSheetMonitor(
            plan=executor.plan, start_ts=self.created_at_ts
        )
        self.sheet_monitor.total_matched_trades = int(executor.completed_crosses)
        self.sheet_monitor.fill_timestamps = deque(executor.matched_timestamps_24h)
        self.ddm_controller = Mark2DualDDMController(
            initial_equity_usdt=executor.plan.initial_capital_usdt
        )


# ============================================================================
# 2. ORQUESTADOR ASÍNCRONO PRINCIPAL PROSPERITY MARK-2
# ============================================================================

class ProsperityMark2Orchestrator:
    """
    Coordina los módulos de MARK-2 para operar Bots Grid Geométricos de Futuros
    gobernados por las 5 Fases de Capital ($4 a $1,000,000 USDT), mitigando la
    incertidumbre con datos de Binance y proyectando el ROI/PnL esperado de cada mercado.
    """

    PUBLIC_MAINNET_FAPI = "https://fapi.binance.com"

    INTERVAL_BARS_PER_24H = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
    }

    def __init__(self, env_path: str = ".env"):
        if m0_workspace_hygiene is not None and hasattr(
            m0_workspace_hygiene, "run_full_hygiene"
        ):
            try:
                m0_workspace_hygiene.run_full_hygiene(checkpoint_wal=False)
            except Exception:
                pass

        # 1. Inicializar Módulo 5 (Adaptador Binance USDⓈ-M + Lector .env + SQLite WAL)
        self.m5_broker = BinanceMark2LiveGridBroker(env_path=env_path)

        # 1.B Inicializar Módulo 1 Warehouse (Almacén Histórico SQLite WAL + Servicio ETL)
        backtest_db_name = self.m5_broker.get_str(
            "BACKTEST_DB_PATH", "prosperity_backtest_warehouse.db"
        )
        self.m1_warehouse = Mark2SQLiteWarehouseRepository(db_path=backtest_db_name)
        self.m1_etl = Mark2BinanceHistoricalETLService(self.m1_warehouse)

        # 1.C Inicializar Módulo 7 (Botón Rojo Quirúrgico por Símbolo / Parada de Emergencia +3% en SQLite WAL)
        self.m7_red_button = Mark2SurgicalRedButtonService(
            broker=self.m5_broker, db_path=self.m5_broker.db_path
        )

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
        self.max_scan_limit = int(self.m5_broker.get_float("MAX_SCAN_SYMBOLS_LIMIT", 30))

        self.initial_wallet_usdt = self.m5_broker.get_float("INITIAL_CAPITAL_USDT", 18.00)
        self.mxn_capital_ref = self.m5_broker.get_float("INITIAL_CAPITAL_MXN", 600.00)
        self.bot_min_usd = self.m5_broker.get_float("BOT_MIN_INVESTMENT_USDT", 4.00)
        self.bot_target_usd = self.m5_broker.get_float("BOT_TARGET_EQUITY_USDT", 0.0)
        self.max_concurrent_bots = int(
            self.m5_broker.get_float("MAX_CONCURRENT_GRID_BOTS", 3)
        )
        self.max_leverage_cap = int(self.m5_broker.get_float("MAX_LEVERAGE_CAP", 25))
        self.open_pos_on_create = (
            self.m5_broker.get_str("OPEN_POSITION_ON_CREATE", "TRUE").upper() == "TRUE"
        )
        self.enable_super_roi_trail = (
            self.m5_broker.get_str("ENABLE_SUPER_ROI_TRAIL", "FALSE").upper() == "TRUE"
        )

        self.maker_fee = self.m5_broker.get_float("MAKER_FEE_RATE", 0.00020)
        self.taker_fee = self.m5_broker.get_float("TAKER_FEE_RATE", 0.00050)
        self.enable_m8_high_risk = (
            self.m5_broker.get_str("ENABLE_M8_HIGH_RISK_ACCELERATOR", "TRUE").upper() == "TRUE"
        )
        self.m8_max_capital_usdt = self.m5_broker.get_float("M8_SMALL_CAPITAL_MAX_USDT", 25.00)
        self.reserve_ratio = self.m5_broker.get_float("RESERVE_MARGIN_RATIO", 0.20)
        max_corr_rho = self.m5_broker.get_float("MAX_CORRELATION_RHO", 0.70)
        max_daily_dd = self.m5_broker.get_float("MAX_DAILY_DRAWDOWN_PCT", 0.05)
        max_p99_ms = self.m5_broker.get_float("MAX_P99_LATENCY_MS", 180.0)

        # 2. Inicializar Módulo 3 y Módulo 8 (Acelerador de Alto Riesgo sin Heurísticos para Capital Pequeño)
        self.m3_brain = Mark2GridDecisionOrchestrator(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            reserve_ratio=self.reserve_ratio,
        )
        self.m3_alpha_calibrator = AlphaHighROIGridCalibrator(maker_fee=self.maker_fee)
        self.m8_accelerator = Mark2HighRiskAlphaAccelerator(
            maker_fee=self.maker_fee, taker_fee=self.taker_fee
        )

        # 3. Inicializar Módulo 4 (Tesorería en 5 Fases, Meta Duplicar en 30d y Mitigación de Incertidumbre)
        self.m4_treasury = Mark2RiskAndTreasuryController(
            initial_wallet_balance_usdt=self.initial_wallet_usdt,
            min_bot_capital_usdt=self.bot_min_usd,
            target_bot_capital_usdt=self.bot_target_usd if self.bot_target_usd > 0 else None,
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
        self._http_semaphore = asyncio.Semaphore(10)

    # ------------------------------------------------------------------------
    # A. CLIENTE HTTP CONCURRENTE CON KEEP-ALIVE Y MEDICIÓN DE LATENCIA LIMPIA
    # ------------------------------------------------------------------------

    async def _safe_http_get_json(
        self,
        session: Optional[Any],
        url: str,
        params: Optional[Dict[str, Any]] = None,
        record_rtt: bool = True,
    ) -> Optional[Any]:
        """
        Realiza peticiones GET hacia Binance reutilizando sockets TCP Keep-Alive
        con semáforo de concurrencia para disminuir la latencia real sin provocar HTTP 429.
        """
        if session is None:
            return None
        async with self._http_semaphore:
            for attempt in range(1, 3):
                t0 = time.perf_counter()
                try:
                    async with session.get(url, params=params, timeout=8) as resp:
                        rtt_ms = (time.perf_counter() - t0) * 1000.0
                        if record_rtt:
                            self.m4_treasury.record_network_rtt_ms(rtt_ms)
                        if resp.status == 200:
                            return await resp.json()
                        if resp.status in (429, 418):
                            await asyncio.sleep(2.0)
                except Exception as exc:
                    if attempt == 2:
                        logger.warning(
                            "🌐 [Error de conexión a Binance API] %s no disponible: %s",
                            url,
                            exc,
                        )
                    await asyncio.sleep(0.5)
        return None

    # ------------------------------------------------------------------------
    # B. ESCÁNER UNIVERSAL CONCURRENTE USDⓈ-M (100% DATOS REALES DE BINANCE)
    # ------------------------------------------------------------------------

    async def sync_universe_and_telemetry(self, session: Optional[Any]) -> None:
        """
        Descarga en paralelo /fapi/v1/exchangeInfo, /fapi/v1/ticker/24hr,
        /fapi/v1/ticker/bookTicker, /fapi/v1/premiumIndex y /fapi/v1/klines de Binance.
        """
        t_start = time.perf_counter()

        ex_info, tickers_24h, book_tickers, premium_list = await asyncio.gather(
            self._safe_http_get_json(
                session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/exchangeInfo", record_rtt=False
            ),
            self._safe_http_get_json(
                session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/24hr", record_rtt=False
            ),
            self._safe_http_get_json(
                session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/bookTicker", record_rtt=True
            ),
            self._safe_http_get_json(
                session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/premiumIndex", record_rtt=True
            ),
        )

        if not ex_info or not isinstance(ex_info, dict) or "symbols" not in ex_info:
            raise RuntimeError(
                "No se pudo obtener /fapi/v1/exchangeInfo desde la API de Binance."
            )

        self.rules_catalog = Mark2UniverseIngestionEngine.parse_binance_exchange_info(ex_info)
        self.m5_broker.contract_rules = self.rules_catalog
        try:
            self.m1_warehouse.upsert_contract_rules(self.rules_catalog)
        except Exception as exc:
            logger.debug("⚠️ [M1 Warehouse Contract Rules]: %s", exc)

        vol_map: Dict[str, float] = {}
        if isinstance(tickers_24h, list):
            for item in tickers_24h:
                sym = item.get("symbol", "")
                if sym in self.rules_catalog:
                    q_vol = float(item.get("quoteVolume", 0.0))
                    if q_vol > 0.0:
                        vol_map[sym] = q_vol

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

        premium_map: Dict[str, Tuple[float, float]] = {}
        if isinstance(premium_list, list):
            for p in premium_list:
                sym = p.get("symbol", "")
                if sym in self.rules_catalog:
                    mp = float(p.get("markPrice", 0.0))
                    fr = float(p.get("lastFundingRate", 0.0))
                    if mp > 0.0:
                        premium_map[sym] = (mp, fr)

        valid_symbols = [
            s for s in self.rules_catalog if s in l1_map and s in vol_map and s in premium_map
        ]

        # Sincronizar comisiones reales Maker/Taker de la cuenta en vivo (/fapi/v1/commissionRate)
        if not self.m5_broker.dry_run and valid_symbols:
            real_mk, real_tk = await self.m5_broker.fetch_symbol_commission_rates(
                valid_symbols[0], session=session
            )
            self.maker_fee, self.taker_fee = real_mk, real_tk
            self.m3_brain.maker_fee = real_mk
            self.m3_brain.taker_fee = real_tk
            self.m8_accelerator.update_real_commission_rates(real_mk, real_tk)

        # Filtro Gauss-Tukey MAD 99% de sangría de funding y factibilidad de lote mínimo ($5 USDT) para capital pequeño
        wallet_ref = max(self.bot_min_usd, self.m4_treasury.wallet_balance_usdt)
        small_cap_m8_active = self.enable_m8_high_risk and (wallet_ref <= self.m8_max_capital_usdt)
        fund_cutoff = Mark2HighRiskAlphaAccelerator.compute_mad_99_funding_cutoff(
            [abs(premium_map[s][1]) for s in valid_symbols]
        )

        range_24h_map: Dict[str, float] = {}
        if isinstance(tickers_24h, list):
            for item in tickers_24h:
                sym_t = item.get("symbol", "")
                if sym_t in self.rules_catalog:
                    hi_24 = float(item.get("highPrice", 0.0))
                    lo_24 = float(item.get("lowPrice", 0.0))
                    if lo_24 > 0.0 and hi_24 >= lo_24:
                        range_24h_map[sym_t] = ((hi_24 - lo_24) / lo_24) * 100.0

        candidate_pool: List[str] = [
            s
            for s in self.curated_symbols
            if s in valid_symbols
            and (
                not small_cap_m8_active
                or ExactExchangeQuantizer.min_order_qty_and_notional(
                    premium_map[s][0], self.rules_catalog[s]
                )[1]
                <= wallet_ref
            )
        ]
        if self.scan_mode == "ALL_USDM_PERPETUALS":
            if small_cap_m8_active:
                # En Modo M8 ($10 USDT): incluir contratos líquidos de $5 min_notional ordenados por amplitud 24h real
                m8_eligible = [
                    s
                    for s in valid_symbols
                    if vol_map.get(s, 0.0) >= 15_000_000.0
                    and abs(premium_map[s][1]) <= fund_cutoff
                    and ExactExchangeQuantizer.min_order_qty_and_notional(
                        premium_map[s][0], self.rules_catalog[s]
                    )[1]
                    <= wallet_ref
                ]
                m8_sorted = sorted(
                    m8_eligible,
                    key=lambda k: (range_24h_map.get(k, 0.0), vol_map.get(k, 0.0)),
                    reverse=True,
                )
                target_scan_cap = max(self.max_scan_limit, 45)
                for s in m8_sorted:
                    if s not in candidate_pool and len(candidate_pool) < target_scan_cap:
                        candidate_pool.append(s)
            else:
                sorted_by_vol = sorted(
                    valid_symbols, key=lambda k: vol_map[k], reverse=True
                )
                for s in sorted_by_vol:
                    if s not in candidate_pool and len(candidate_pool) < self.max_scan_limit:
                        candidate_pool.append(s)

        bars_24h = self.INTERVAL_BARS_PER_24H.get(self.interval, 288)

        async def _fetch_and_build_symbol_telemetry(sym: str) -> None:
            klines = await self._safe_http_get_json(
                session,
                f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/klines",
                {"symbol": sym, "interval": self.interval, "limit": self.warmup_candles},
                record_rtt=True,
            )
            if isinstance(klines, list) and len(klines) >= 30:
                closes = np.array([float(k[4]) for k in klines], dtype=np.float64)
                highs = np.array([float(k[2]) for k in klines], dtype=np.float64)
                lows = np.array([float(k[3]) for k in klines], dtype=np.float64)
                vols = np.array([float(k[5]) for k in klines], dtype=np.float64)
                taker_buy_vols = np.array([float(k[9]) for k in klines], dtype=np.float64)
                last_kline_ts_ms = int(klines[-1][6])

                # Persistir velas e indicadores derivados en el Almacén Histórico SQLite WAL (M1 Warehouse)
                try:
                    candle_records = [
                        HistoricalCandleRecordM2(
                            symbol=sym,
                            timeframe=self.interval,
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
                        for k in klines
                    ]
                    self.m1_warehouse.upsert_candles(candle_records, is_synthetic=0)
                    mp_val, fr_val = premium_map[sym]
                    self.m1_warehouse.upsert_derivatives(
                        [
                            HistoricalDerivativeRecordM2(
                                symbol=sym,
                                timestamp_ms=last_kline_ts_ms,
                                funding_rate=fr_val,
                                open_interest_usdt=vol_map[sym],
                                top_trader_ls_ratio=1.0,
                            )
                        ]
                    )
                except Exception as exc:
                    logger.debug("⚠️ [M1 Warehouse Upsert %s]: %s", sym, exc)
            else:
                # Respaldo determinista hacia el Almacén Histórico SQLite WAL si la red falla temporalmente
                ohlcv_db = self.m1_warehouse.load_numpy_ohlcv_arrays(
                    sym, timeframe=self.interval
                )
                if len(ohlcv_db["closes"]) < 30:
                    return
                closes = ohlcv_db["closes"]
                highs = ohlcv_db["highs"]
                lows = ohlcv_db["lows"]
                vols = ohlcv_db["volumes"]
                taker_buy_vols = ohlcv_db["taker_buy_base"]
                last_kline_ts_ms = int(ohlcv_db["open_times"][-1])

            bid, ask, bid_q, ask_q = l1_map[sym]
            q_vol = vol_map[sym]
            mark_p, funding_r = premium_map[sym]

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
                mark_price=mark_p,
                quote_vol_24h=q_vol,
                funding_rate=funding_r,
                timestamp_ms=last_kline_ts_ms,
                taker_buy_volumes=taker_buy_vols,
                bars_per_24h=bars_24h,
            )
            self.telemetry_catalog[sym] = tel
            self.returns_catalog[sym] = np.diff(np.log(np.maximum(closes, 1e-12)))

        await asyncio.gather(
            *(_fetch_and_build_symbol_telemetry(sym) for sym in candidate_pool[: self.max_scan_limit])
        )

        scan_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "⚡ [Escaneo Concurrente Binance Completado] %d mercados analizados en %.0f ms (RTT p99=%.1f ms)",
            len(self.telemetry_catalog),
            scan_elapsed_ms,
            self.m4_treasury.get_p99_latency_ms(),
        )

    # ------------------------------------------------------------------------
    # C. IMPRESIÓN DE TABLA UNIVERSAL Y FICHA ESTILO APP DE BINANCE
    # ------------------------------------------------------------------------

    def print_universe_ranking_table(
        self, ranked_list: List[RankedMarketCandidatePOD]
    ) -> None:
        phase_prof = self.m4_treasury.resolve_phase_behavior_profile()
        mult_lbl = f"{phase_prof.phase_target_multiplier:.0f}x"
        grouped = self.m3_brain.group_markets_by_empirical_area(ranked_list)
        area_headers = (
            ("ESTABLES", "🟢 ÁREA 1: MERCADOS CRIPTO ESTABLES (Datos API: Menor Varianza NATR/EWMA, Menor Spread L1 y Mayor Liquidez 24h)"),
            ("INTERMEDIOS", "🟡 ÁREA 2: MERCADOS CRIPTO INTERMEDIOS (Datos API: Régimen Central de Volatilidad y Rotación Intradía)"),
            ("VOLATILES", "🔴 ÁREA 3: MERCADOS CRIPTO VOLÁTILES (Datos API: Mayor Excursión MAE/MFE, Recorrido 24h y Amplitud de Rango)"),
        )
        print("\n" + "=" * 144)
        print(
            f"🌍 ESCÁNER CRIPTO INTELIGENTE EN 3 ÁREAS (DATOS API BINANCE) | {phase_prof.phase.value} "
            f"(Billetera: ${self.m4_treasury.wallet_balance_usdt:.2f} USDT | Meta Mín {mult_lbl} 30d: "
            f"${phase_prof.min_30d_target_equity_usdt:.2f} USDT [+{phase_prof.target_roi_per_bot_pct:.0f}% ROI])"
        )
        print("=" * 144)
        global_idx = 1
        for area_key, area_title in area_headers:
            area_markets = grouped.get(area_key, [])
            print(f"\n{area_title} — [{len(area_markets)} mercados]")
            print("-" * 144)
            print(
                f"{'ORD':<4} | {'ÁREA API':<11} | {'MERCADO':<10} | {'CERTIDUMBRE':<11} | {'LEV / RIESGO':<14} | "
                f"{'LOTE MÍN':<9} | {'ESP. DIARIO (USDT/%)':<21} | {'ESP. 30 DÍAS (USDT/%)':<23} | "
                f"{'DÍAS EST.':<9} | {'ESTADO Y MITIGACIÓN'}"
            )
            print("-" * 144)
            for c in area_markets:
                c_mult = f"{c.phase_target_multiplier:.0f}x"
                badge = (
                    f"✅ APTO ({c_mult} en ~{c.est_days_to_double_capital:.1f}d)"
                    if c.viable_for_6usd_bot
                    else f"🔒 Min ${c.min_capital_for_grid_usdt:.2f}"
                )
                daily_str = f"+${c.expected_daily_pnl_usdt:.3f} (+{c.expected_daily_roi_pct:.2f}%)"
                m30_str = f"+${c.expected_30d_pnl_usdt:.2f} (+{c.expected_30d_roi_pct:.1f}%)"
                lev_risk_str = f"{c.recommended_phase_leverage:>2}x / {c.recommended_risk_pct:>4.2f}%"
                print(
                    f"#{global_idx:<3} | {c.safety_tier.value:<11} | {c.symbol:<10} | {c.confidence_pct:>9.1f}%  | {lev_risk_str:<14} | "
                    f"${c.min_line_notional_usdt:>7.2f} | {daily_str:<21} | {m30_str:<23} | "
                    f"{c.est_days_to_double_capital:>7.1f}d | {badge}"
                )
                global_idx += 1
        print("=" * 144)

    @staticmethod
    def format_binance_bot_details_card(
        plan: GeometricGridPlanPOD,
        matched_24h: int,
        matched_total: int,
        mdd_7d_pct: float,
        runtime_sec: float = 0.0,
    ) -> str:
        """
        Genera en consola la ficha 'Detalles del bot / Grid de futuros' con el Área Cripto (ESTABLES,
        INTERMEDIOS, VOLÁTILES), Certidumbre (%), Apalancamiento/Riesgo de la Fase, ROI/PnL Esperado
        Diario y a 30 Días de cada mercado y Días Estimados (3x en Fases 1 y 2; 2x en Fase 3).
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
        mult_lbl = f"{plan.phase_target_multiplier:.0f}x"

        return (
            f"\n┌──────────────────────────────────────────────────────────────────────────────────────────┐\n"
            f"│ 🤖 DETALLES DEL BOT GRID DE FUTUROS (MARK-2) — {plan.symbol:<10} [{plan.direction.value} {plan.leverage}x | {plan.safety_tier.value}] │\n"
            f"├──────────────────────────────────────────────────────────────────────────────────────────┤\n"
            f"│ • Área / Fase / Certid.  : Área: {plan.safety_tier.value} | {plan.active_phase} | Certidumbre API: {plan.confidence_pct:.1f}% | Riesgo Op: {plan.phase_operational_risk_pct:.2f}%\n"
            f"│ • PnL (USDT) / ROI Actual: {pnl_now:+.2f} USDT ({roi_now:+.2f}%) | Equidad Actual: ${plan.current_equity_usdt:.2f} USDT\n"
            f"│ • ROI/PnL Esp. Mercado   : Diario: +${plan.expected_daily_pnl_usdt:.3f} (+{plan.expected_daily_roi_pct:.2f}%/d) | 30 Días: +${plan.expected_30d_pnl_usdt:.2f} (+{plan.expected_30d_roi_pct:.1f}%)\n"
            f"│ • Capital / Meta Mercado : ${plan.initial_capital_usdt:.2f} USDT ──► Meta Ciclo: ${plan.target_equity_usdt:.2f} USDT (+{plan.tp_roi_pct:.1f}% ROI | Meta Mín {mult_lbl} 30d: ${plan.min_30d_doubling_target_usdt:.2f})\n"
            f"│ • Tiempo de ejecución    : {days}d {hours}h {mins}m | Tramo: {plan.compounding_stage.value}\n"
            f"│ • Total trades emparejados: {matched_24h}/{matched_total} (24h/Total) | MDD en 7D: {mdd_7d_pct:.2f}%\n"
            f"│ • Dirección / Apalancam. : {plan.direction.value} | ISOLATED {plan.leverage}x (Efectivo Inicial: {plan.effective_initial_leverage:.2f}x / Max Adverso: {plan.max_adverse_effective_leverage:.2f}x)\n"
            f"│ • Rango de precios (USDT): {plan.lower_price} - {plan.upper_price}\n"
            f"│ • Cant. de grids         : {plan.total_virtual_grids} niveles ({plan.active_window_orders} órdenes activas en libro)\n"
            f"│ • Modo / Lote por línea  : Geométrico | Qty={plan.qty_per_line} (${plan.notional_per_line_usdt:.2f} USDT)\n"
            f"│ • Ganancia/grid (neta)   : {plan.net_profit_per_cross_pct:.3f}% (+${plan.net_profit_per_cross_usdt:.4f} USDT/cruce desc. fee)\n"
            f"│ • Densidad de grid       : {density_grids_pct} grids/% | Paso geométrico: {plan.geometric_step_pct:.3f}%\n"
            f"│ • Trailing ascendente    : {'Activado' if plan.trailing_up else 'Desactivado'}\n"
            f"│ • Trailing descendente   : {'Activado' if plan.trailing_down else 'Desactivado'}\n"
            f"│ • Stop Loss / Piso Ratch.: {plan.sl_price} USDT | Piso Ratchet: ${plan.equity_ratchet_floor_usdt:.2f} USDT ({plan.sl_roi_pct:+.1f}% ROI) | P_liq: {plan.estimated_liquidation_price} USDT\n"
            f"│ • Take Profit (ROI Esp.) : {plan.tp_price} USDT | +${plan.tp_pnl_usdt:.2f} USDT (+{plan.tp_roi_pct:.1f}% ROI Esperado en Mercado)\n"
            f"│ • Abrir pos. en creación : {'Activado' if plan.direction != GridDirection.NEUTRAL else 'Neutral (Límites)'}\n"
            f"└──────────────────────────────────────────────────────────────────────────────────────────┘"
        )

    # ------------------------------------------------------------------------
    # D. DESPLIEGUE DE BOTS GRID GOBERNADOS POR FASE (1 A 5) Y ROI ESPERADO
    # ------------------------------------------------------------------------

    async def deploy_eligible_grid_bots(self, session: Optional[Any]) -> None:
        """
        Clasifica todos los mercados según sus datos reales de Binance, mitiga la incertidumbre,
        calcula el ROI y PnL esperado en cada mercado y despliega los Bots Grid según la Fase activa.
        """
        phase_profile = self.m4_treasury.resolve_phase_behavior_profile()
        bot_cap = phase_profile.capital_allocation_per_bot_usdt

        ranked_universe = self.m3_brain.rank_all_markets_safe_to_volatile(
            rules_map=self.rules_catalog,
            telemetry_map=self.telemetry_catalog,
            bot_capital_usdt=bot_cap,
            max_leverage=phase_profile.max_leverage,
            target_equity_usdt=None,  # Calibrado dinámicamente según el ROI/PnL esperado de cada mercado
            wallet_balance_usdt=self.m4_treasury.wallet_balance_usdt,
        )
        self.print_universe_ranking_table(ranked_universe)

        viable_candidates = [
            c
            for c in ranked_universe
            if c.viable_for_6usd_bot and c.symbol not in self.active_bots
        ]
        m8_small_cap_mode = (
            self.enable_m8_high_risk
            and self.m4_treasury.wallet_balance_usdt <= self.m8_max_capital_usdt
        )
        if m8_small_cap_mode:
            viable_candidates.sort(
                key=lambda c: (
                    c.safety_tier != MarketSafetyTier.VOLATILES,
                    -c.expected_30d_roi_pct,
                    -c.expected_daily_roi_pct,
                )
            )
        else:
            viable_candidates.sort(
                key=lambda c: (
                    c.est_days_to_double_capital > 30.0,
                    -c.expected_30d_roi_pct,
                )
            )

        max_slots = (
            1
            if (m8_small_cap_mode and self.m4_treasury.wallet_balance_usdt <= 12.50)
            else min(self.max_concurrent_bots, phase_profile.max_concurrent_bots)
        )
        live_pos_map = await self.m5_broker.fetch_live_positions_map(session)

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
                candidate_uncertainty_score=cand.uncertainty_score,
            )

            if not verdict.approved:
                logger.info("🛡️ [M4 Aduana en %s]: %s", cand.symbol, verdict.reason)
                continue

            rules = self.rules_catalog[cand.symbol]
            tel = self.telemetry_catalog[cand.symbol]

            existing_pos = live_pos_map.get(cand.symbol)
            existing_pos_amt = (
                float(existing_pos.get("positionAmt", 0.0)) if existing_pos else 0.0
            )
            forced_dir: Optional[GridDirection] = None
            if abs(existing_pos_amt) > 0.0:
                forced_dir = (
                    GridDirection.LONG if existing_pos_amt > 0.0 else GridDirection.SHORT
                )

            plan: Optional[GeometricGridPlanPOD] = None
            if m8_small_cap_mode and forced_dir is None:
                m8_plan = self.m8_accelerator.build_high_risk_plan_from_live_telemetry(
                    rule=rules,
                    tel=tel,
                    wallet_capital_usdt=verdict.assigned_capital_usdt,
                    critic_score=max(0.10, 1.0 - cand.uncertainty_score),
                )
                if m8_plan is not None:
                    plan = self.m8_accelerator.to_geometric_grid_plan_pod(m8_plan, rules)
                    logger.info(
                        "🔥 [M8 ACELERADOR ALTO RIESGO %s] Modo=%s | Aislado %dx (Efec %.2fx) | Anclas GTX=[%s / %s] | TP=%s (+%.2f%% Wallet) | SL=%s (%.2f%% Wallet) | P_liq=%s (< SL) | Cuota/Bruto=%.2f%%",
                        cand.symbol,
                        m8_plan.execution_mode,
                        m8_plan.isolated_leverage,
                        m8_plan.effective_wallet_leverage,
                        m8_plan.anchor_buy_gtx_price,
                        m8_plan.anchor_sell_gtx_price,
                        m8_plan.tp_limit_gtx_price,
                        m8_plan.expected_wallet_roi_tp_pct,
                        m8_plan.sl_stop_market_price,
                        m8_plan.expected_wallet_roi_sl_pct,
                        m8_plan.liquidation_price,
                        m8_plan.fee_to_gross_tp_pct,
                    )

            if plan is None:
                plan = self.m3_brain.build_geometric_grid_6_to_18(
                    rules=rules,
                    tel=tel,
                    safety_tier=cand.safety_tier,
                    initial_capital_usdt=verdict.assigned_capital_usdt,
                    current_equity_usdt=verdict.assigned_capital_usdt,
                    target_equity_usdt=None,  # Usa el ROI/PnL esperado del mercado + meta 30d de la Fase
                    forced_direction=forced_dir,
                    max_leverage=verdict.approved_leverage,
                    wallet_balance_usdt=self.m4_treasury.wallet_balance_usdt,
                )

            if not self.m5_broker.dry_run:
                await self.m5_broker.cancel_all_symbol_orders(
                    cand.symbol, include_algo=True, session=session
                )

            await self.m5_broker.configure_isolated_and_leverage(
                cand.symbol, plan.leverage, session=session
            )

            reused_existing_pos = False
            initial_pos_amt = 0.0
            if abs(existing_pos_amt) > 0.0 and existing_pos is not None:
                reused_existing_pos = True
                initial_pos_amt = existing_pos_amt
                avg_p = float(existing_pos.get("entryPrice", tel.last_price))
                logger.info(
                    "♻️ [M5 POSICIÓN EXISTENTE EN %s] Reutilizando inventario abierto Qty=%s @ Entry=%.5f",
                    cand.symbol,
                    existing_pos_amt,
                    avg_p,
                )
            elif self.open_pos_on_create and plan.direction != GridDirection.NEUTRAL:
                avg_p = await self.m5_broker.open_initial_base_position(
                    symbol=cand.symbol,
                    direction=plan.direction,
                    base_qty=plan.qty_per_line,
                    rules=rules,
                    session=session,
                )
                initial_pos_amt = self.m5_broker.live_position_amt_cache.get(
                    cand.symbol,
                    plan.qty_per_line
                    if plan.direction == GridDirection.LONG
                    else -plan.qty_per_line,
                )
            else:
                avg_p = 0.0
                self.m5_broker.live_position_amt_cache[cand.symbol] = 0.0

            await self.m5_broker.sync_active_limit_window(
                symbol=cand.symbol,
                buy_levels=plan.active_buy_window,
                sell_levels=plan.active_sell_window,
                qty=plan.qty_per_line,
                rules=rules,
                current_position_amt=initial_pos_amt,
                session=session,
            )

            # Brackets asíncronos vinculados a Fills: solo despachar si ya existe una posición FILLED activa
            if abs(initial_pos_amt) > 1e-12:
                try:
                    await self.m5_broker.place_global_protection_brackets(
                        symbol=cand.symbol,
                        direction=plan.direction,
                        sl_price=plan.sl_price,
                        tp_price=plan.tp_price,
                        current_mark_price=tel.mark_price,
                        rules=rules,
                        current_position_amt=initial_pos_amt,
                        order_filled=not reused_existing_pos,
                        verified_exchange_pos_amt=existing_pos_amt if reused_existing_pos else None,
                        lower_price=plan.lower_price,
                        upper_price=plan.upper_price,
                        session=session,
                    )
                except Exception as exc:
                    logger.info(
                        "ℹ️ [M5 Brackets en %s] Supervisión de SL/TP delegada al bucle M4 en RAM (%s)",
                        cand.symbol,
                        exc,
                    )
            else:
                logger.info(
                    "⏳ [M5 Brackets Asíncronos en %s] Posición inicial en 0. SL/TP se enviarán al confirmarse el primer FILL.",
                    cand.symbol,
                )

            executor = Mark2SlidingWindowGridExecutor(
                plan=plan,
                rules=rules,
                journal_db_path=self.m5_broker.db_path,
                maker_fee=self.m3_brain.maker_fee,
            )
            if reused_existing_pos:
                entry_ref = avg_p if avg_p > 0.0 else tel.last_price
                executor.net_inventory_qty = existing_pos_amt
                executor.avg_entry_price = entry_ref
                if existing_pos_amt > 0.0:
                    for sp in plan.active_sell_window:
                        executor.open_lots_entry_map[sp] = entry_ref
                else:
                    for bp in plan.active_buy_window:
                        executor.open_lots_entry_map[bp] = entry_ref
                if not self.m5_broker.dry_run and existing_pos is not None:
                    executor.unrealized_pnl_usdt = round(
                        float(existing_pos.get("unRealizedProfit", 0.0)), 4
                    )
            elif self.open_pos_on_create and plan.direction != GridDirection.NEUTRAL:
                entry_ref = avg_p if avg_p > 0.0 else tel.last_price
                executor.net_inventory_qty = (
                    plan.qty_per_line
                    if plan.direction == GridDirection.LONG
                    else -plan.qty_per_line
                )
                executor.avg_entry_price = entry_ref
                if plan.direction == GridDirection.LONG:
                    for sp in plan.active_sell_window:
                        executor.open_lots_entry_map[sp] = entry_ref
                else:
                    for bp in plan.active_buy_window:
                        executor.open_lots_entry_map[bp] = entry_ref

            # Si el ejecutor rehidrató PnL previo desde SQLite WAL (mark2_grid_cycles),
            # sincronizar la equidad y el tramo compuesto del plan antes de mostrar la ficha.
            if abs(executor.realized_pnl_usdt) > 1e-12 or abs(executor.unrealized_pnl_usdt) > 1e-12:
                _, updated_plan = self.m4_treasury.evaluate_bot_equity_transition(
                    current_plan=executor.plan,
                    realized_pnl_usdt=executor.realized_pnl_usdt,
                    unrealized_pnl_usdt=executor.unrealized_pnl_usdt,
                    rules=rules,
                    tel=tel,
                    orchestrator=self.m3_brain,
                )
                executor.plan = updated_plan
                plan = updated_plan

            ctx = LiveGridBotContext(
                symbol=cand.symbol,
                rules=rules,
                telemetry=tel,
                executor=executor,
                log_returns=cand_returns,
            )
            self.active_bots[cand.symbol] = ctx
            self.m4_treasury.correlation_guard.register_active_bot(
                cand.symbol, plan.direction
            )

            t_24h, t_tot = executor.get_matched_trades_24h_and_total()
            mdd_7d = executor.compute_mdd_7d_pct()
            runtime_elapsed = max(0.0, time.time() - ctx.created_at_ts)

            print(
                self.format_binance_bot_details_card(
                    plan=plan,
                    matched_24h=t_24h,
                    matched_total=t_tot,
                    mdd_7d_pct=mdd_7d,
                    runtime_sec=runtime_elapsed,
                )
            )

    # ------------------------------------------------------------------------
    # E. CORRUTINAS ASÍNCRONAS DE EMPAREJAMIENTO EN TIEMPO REAL Y TRINQUETE M4
    # ------------------------------------------------------------------------

    async def pre_flight_check(self, session: Optional[Any]) -> None:
        logger.info("==========================================================================")
        logger.info(
            "🚀 INICIANDO PROSPERITY MARK-2 | Modo: %s (DRY_RUN=%s) | Gobernanza en 5 Fases (Meta 30d: Duplicar en Fases 1-3)",
            self.m5_broker.env_mode,
            self.m5_broker.dry_run,
        )
        logger.info("==========================================================================")

        if session is not None:
            await self.m5_broker.sync_server_time(session)
            await self.m5_broker.sync_multi_assets_mode(session)
            real_maker, real_taker = await self.m5_broker.fetch_commission_rates(
                symbol="BTCUSDT", session=session
            )
            self.m3_brain.maker_fee = real_maker
            self.m3_brain.taker_fee = real_taker
            real_bal = await self.m5_broker.fetch_usdt_wallet_balance(session)
            self.m4_treasury.reconcile_wallet_with_exchange(real_bal)

        phase_prof = self.m4_treasury.resolve_phase_behavior_profile()
        logger.info(
            "💰 [TESORERÍA M4] Balance Billetera: $%.2f USDT (~$%.2f MXN) | Fase Activa: %s | "
            "Apalancamiento Fase: %dx (Max %dx) | Riesgo Op: %.1f%% | Meta Mín 30d: $%.2f USDT (+%.0f%% ROI)",
            self.m4_treasury.wallet_balance_usdt,
            self.m4_treasury.wallet_balance_usdt
            * (self.mxn_capital_ref / max(self.initial_wallet_usdt, 1.0)),
            phase_prof.phase.value,
            phase_prof.recommended_leverage,
            phase_prof.max_leverage,
            phase_prof.max_account_risk_per_bot_pct,
            phase_prof.min_30d_target_equity_usdt,
            phase_prof.target_roi_per_bot_pct,
        )

        await self.sync_universe_and_telemetry(session)
        await self.deploy_eligible_grid_bots(session)

    async def grid_hot_path_matching_loop(self, session: Optional[Any]) -> None:
        """
        Bucle de Ruta Caliente Basado 100% en Precios Reales de Binance:
        Consulta en lote /fapi/v1/ticker/bookTicker y /fapi/v1/premiumIndex en cada ciclo,
        actualiza el Mark-to-Market real de cada Bot Grid y ejecuta los cruces BUY/SELL
        únicamente cuando el bidPrice / askPrice real de Binance cruza el nivel del Grid.
        """
        while self._running:
            try:
                await asyncio.sleep(2.0)
                if not self.active_bots:
                    continue

                bt_all, prem_all = await asyncio.gather(
                    self._safe_http_get_json(
                        session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/ticker/bookTicker"
                    ),
                    self._safe_http_get_json(
                        session, f"{self.PUBLIC_MAINNET_FAPI}/fapi/v1/premiumIndex"
                    ),
                )

                bt_map: Dict[str, Dict[str, Any]] = {}
                if isinstance(bt_all, list):
                    for item in bt_all:
                        s = item.get("symbol", "")
                        if s in self.active_bots:
                            bt_map[s] = item

                prem_map: Dict[str, float] = {}
                if isinstance(prem_all, list):
                    for item in prem_all:
                        s = item.get("symbol", "")
                        if s in self.active_bots:
                            prem_map[s] = float(item.get("markPrice", 0.0))

                live_pos_map = await self.m5_broker.fetch_live_positions_map(session)

                async with self._execution_lock:
                    for sym, ctx in list(self.active_bots.items()):
                        try:
                            ex = ctx.executor
                            bt = bt_map.get(sym)
                            if not bt:
                                continue

                            live_bid = float(bt.get("bidPrice", 0.0))
                            live_ask = float(bt.get("askPrice", 0.0))
                            live_bid_q = float(bt.get("bidQty", 0.0))
                            live_ask_q = float(bt.get("askQty", 0.0))
                            if not Mark2UniverseIngestionEngine.validate_l1_quote_o1(
                                live_bid, live_ask, live_bid_q, live_ask_q
                            ):
                                continue

                            live_mid = (live_bid + live_ask) * 0.5
                            live_mark = prem_map.get(sym, live_mid)

                            real_binance_pos_amt = 0.0
                            if not self.m5_broker.dry_run and sym in live_pos_map:
                                real_binance_pos_amt = float(
                                    live_pos_map[sym].get("positionAmt", 0.0)
                                )
                                ex.unrealized_pnl_usdt = round(
                                    live_pos_map[sym]["unRealizedProfit"], 4
                                )
                                ex.net_inventory_qty = real_binance_pos_amt
                                self.m5_broker.live_position_amt_cache[sym] = ex.net_inventory_qty
                            else:
                                ex.update_mark_to_market(live_mark)

                            # Si Binance confirma posición real != 0 y aún no están activos los brackets SL/TP, colocarlos
                            if (
                                not self.m5_broker.dry_run
                                and abs(real_binance_pos_amt) > 1e-12
                                and not self.m5_broker.active_brackets_placed.get(sym, False)
                            ):
                                try:
                                    await self.m5_broker.place_global_protection_brackets(
                                        symbol=sym,
                                        direction=ex.plan.direction,
                                        sl_price=ex.plan.sl_price,
                                        tp_price=ex.plan.tp_price,
                                        current_mark_price=live_mark,
                                        rules=ctx.rules,
                                        current_position_amt=real_binance_pos_amt,
                                        order_filled=False,
                                        verified_exchange_pos_amt=real_binance_pos_amt,
                                        lower_price=ex.plan.lower_price,
                                        upper_price=ex.plan.upper_price,
                                        session=session,
                                    )
                                except Exception as exc:
                                    logger.info(
                                        "ℹ️ [M5 Brackets Post-Fill %s] Supervisión delegada a M4 (%s)",
                                        sym,
                                        exc,
                                    )

                            denom_obi = (live_bid * live_bid_q) + (live_ask * live_ask_q)
                            live_obi = (
                                ((live_bid * live_bid_q) - (live_ask * live_ask_q)) / denom_obi
                                if denom_obi > 0.0
                                else 0.0
                            )

                            # En modo LIVE: consultar órdenes abiertas en Binance para no disparar fills por simple roce de bookTicker si la orden sigue abierta
                            still_open_buy_prices = set()
                            still_open_sell_prices = set()
                            if not self.m5_broker.dry_run and (
                                any(live_bid <= bp for bp in ex.active_buys)
                                or any(live_ask >= sp for sp in ex.active_sells)
                            ):
                                open_reg = await self.m5_broker.fetch_open_regular_orders(
                                    sym, session=session
                                )
                                for o_item in open_reg:
                                    o_side = str(o_item.get("side", "")).upper()
                                    o_px = ExactExchangeQuantizer.quantize_price(
                                        float(o_item.get("price", 0.0)), ctx.rules.tick_size
                                    )
                                    if o_side == "BUY":
                                        still_open_buy_prices.add(o_px)
                                    elif o_side == "SELL":
                                        still_open_sell_prices.add(o_px)

                            # 1. Evaluar si el bidPrice real de Binance ejecutó algún escalón BUY activo
                            for b_price in list(ex.active_buys):
                                if live_bid <= b_price and b_price not in still_open_buy_prices:
                                    flip = ex.on_limit_order_filled("BUY", b_price)
                                    sell_target = float(flip["new_order_price"])
                                    self.m5_broker.live_position_amt_cache[sym] = ex.net_inventory_qty
                                    if not self.m5_broker.dry_run:
                                        await self.m5_broker.place_single_counter_limit_order(
                                            symbol=sym,
                                            side="SELL",
                                            price=sell_target,
                                            qty=ex.plan.qty_per_line,
                                            rules=ctx.rules,
                                            reduce_only=real_binance_pos_amt > 0.0,
                                            current_position_amt=real_binance_pos_amt,
                                            session=session,
                                        )
                                        if (
                                            abs(ex.net_inventory_qty) > 1e-12
                                            and not self.m5_broker.active_brackets_placed.get(sym, False)
                                        ):
                                            try:
                                                await self.m5_broker.place_global_protection_brackets(
                                                    symbol=sym,
                                                    direction=ex.plan.direction,
                                                    sl_price=ex.plan.sl_price,
                                                    tp_price=ex.plan.tp_price,
                                                    current_mark_price=live_mark,
                                                    rules=ctx.rules,
                                                    current_position_amt=ex.net_inventory_qty,
                                                    order_filled=True,
                                                    verified_exchange_pos_amt=None,
                                                    lower_price=ex.plan.lower_price,
                                                    upper_price=ex.plan.upper_price,
                                                    session=session,
                                                )
                                            except Exception as exc:
                                                logger.info(
                                                    "ℹ️ [M5 Brackets Post-Fill %s] Supervisión delegada a M4 (%s)",
                                                    sym,
                                                    exc,
                                                )
                                    logger.info(
                                        "🟢 [ORDEN BUY EJECUTADA %s] Fill @ %s (Bid Real=%s) -> Contra-Orden SELL @ %s (PosAmt=%.4f)",
                                        sym,
                                        b_price,
                                        live_bid,
                                        sell_target,
                                        ex.net_inventory_qty,
                                    )

                            # 2. Evaluar si el askPrice real de Binance ejecutó algún escalón SELL activo
                            for s_price in list(ex.active_sells):
                                if live_ask >= s_price and s_price not in still_open_sell_prices:
                                    harvest = ex.on_limit_order_filled("SELL", s_price)
                                    buy_target = float(harvest["new_order_price"])
                                    net_harv = float(harvest["net_harvest_usdt"])
                                    cum_eq = float(harvest["cumulative_equity_usdt"])
                                    self.m5_broker.live_position_amt_cache[sym] = ex.net_inventory_qty
                                    ctx.sheet_monitor.register_matched_trade(
                                        current_equity_usdt=cum_eq,
                                        net_pnl_usdt=net_harv,
                                    )
                                    ddm_snap = ctx.ddm_controller.register_closed_cycle(
                                        net_pnl_usdt=net_harv,
                                        current_equity_usdt=cum_eq,
                                    )
                                    if ddm_snap.regime_status == HighRiskDDMStatus.DRIFT_HALT_3SIGMA:
                                        logger.warning(
                                            "🚨 [M8 DDM 3-SIGMA DRIFT HALT EN %s] p=%.3f + s=%.3f >= p_min+3*s_min | Purgando y rotando mercado.",
                                            sym,
                                            ddm_snap.error_rate_p,
                                            ddm_snap.std_error_s,
                                        )
                                        await self.m5_broker.atomic_close_and_purge_bot(
                                            sym, reason="M8_GAMA_DDM_3SIGMA_DRIFT_HALT", session=session
                                        )
                                        self.active_bots.pop(sym, None)
                                        break
                                    if not self.m5_broker.dry_run:
                                        await self.m5_broker.place_single_counter_limit_order(
                                            symbol=sym,
                                            side="BUY",
                                            price=buy_target,
                                            qty=ex.plan.qty_per_line,
                                            rules=ctx.rules,
                                            reduce_only=real_binance_pos_amt < 0.0,
                                            current_position_amt=real_binance_pos_amt,
                                            session=session,
                                        )
                                        if (
                                            abs(ex.net_inventory_qty) > 1e-12
                                            and not self.m5_broker.active_brackets_placed.get(sym, False)
                                        ):
                                            try:
                                                await self.m5_broker.place_global_protection_brackets(
                                                    symbol=sym,
                                                    direction=ex.plan.direction,
                                                    sl_price=ex.plan.sl_price,
                                                    tp_price=ex.plan.tp_price,
                                                    current_mark_price=live_mark,
                                                    rules=ctx.rules,
                                                    current_position_amt=ex.net_inventory_qty,
                                                    order_filled=True,
                                                    verified_exchange_pos_amt=None,
                                                    lower_price=ex.plan.lower_price,
                                                    upper_price=ex.plan.upper_price,
                                                    session=session,
                                                )
                                            except Exception as exc:
                                                logger.info(
                                                    "ℹ️ [M5 Brackets Post-Fill %s] Supervisión delegada a M4 (%s)",
                                                    sym,
                                                    exc,
                                                )
                                    logger.info(
                                        "⚡ [CRUCE EMPAREJADO %s | %s] SELL @ %s (Ask Real=%s) -> Contra-Orden BUY @ %s | Neto: +$%.4f USDT | Equidad Bot: $%.2f USDT | DDM: %s",
                                        sym,
                                        ex.plan.compounding_stage.value,
                                        s_price,
                                        live_ask,
                                        buy_target,
                                        net_harv,
                                        cum_eq,
                                        ddm_snap.regime_status.value,
                                    )

                            # 2.B Evaluar Ratchet Breakeven + Escudo de Cuotas para planes M8 con inventario activo
                            if (
                                sym in self.active_bots
                                and ex.plan.m8_high_risk_plan is not None
                                and abs(ex.net_inventory_qty) > 1e-12
                            ):
                                new_be_sl = Mark2HighRiskAlphaAccelerator.evaluate_breakeven_fee_shield_ratchet(
                                    m8_plan=ex.plan.m8_high_risk_plan,
                                    position_amt=ex.net_inventory_qty,
                                    entry_price=ex.avg_entry_price,
                                    live_mark_price=live_mark,
                                    current_sl_price=ex.plan.sl_price,
                                    tick_size=ctx.rules.tick_size,
                                    maker_fee=self.maker_fee,
                                    taker_fee=self.taker_fee,
                                )
                                if new_be_sl is not None:
                                    ex.plan = dataclasses.replace(ex.plan, sl_price=new_be_sl)
                                    logger.info(
                                        "🛡️ [M8 RATCHET BREAKEVEN + CUOTAS %s] Mark=%.5f alcanzó 0.60*TP -> Nuevo SL asegurado en %s",
                                        sym,
                                        live_mark,
                                        new_be_sl,
                                    )

                            # 3. Evaluar Rebalanceo Infinito (Trailing Grid Up / Down) con precio y OBI reales
                            trail_evt = ex.check_and_apply_trailing_grid(
                                current_price=live_mid,
                                quote_obi_l1=live_obi,
                            )
                            if trail_evt is not None:
                                if not self.m5_broker.dry_run:
                                    await self.m5_broker.sync_active_limit_window(
                                        symbol=sym,
                                        buy_levels=tuple(ex.active_buys),
                                        sell_levels=tuple(ex.active_sells),
                                        qty=ex.plan.qty_per_line,
                                        rules=ctx.rules,
                                        cancel_existing=True,
                                        current_position_amt=ex.net_inventory_qty,
                                        session=session,
                                    )
                                logger.info(
                                    "🔄 [TRAILING GRID %s] %s | Nuevo Rango: [%s - %s]",
                                    sym,
                                    trail_evt["trailing_action"],
                                    trail_evt["new_range_lower"],
                                    trail_evt["new_range_upper"],
                                )
                        except asyncio.CancelledError:
                            raise
                        except Exception as sym_exc:
                            if "-4509" in str(sym_exc):
                                logger.info(
                                    "⏳ [Hot-Path %s] Estado AlgoOrder en transición (-4509 controlado); continúa supervisión.",
                                    sym,
                                )
                            else:
                                logger.warning(
                                    "⚠️ [Hot-Path Aislado %s] Excepción capturada sin afectar otros bots: %s",
                                    sym,
                                    sym_exc,
                                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("❌ [Error en Grid Hot-Path Loop]: %s", exc)
                await asyncio.sleep(2.0)

    async def treasury_ratchet_and_audit_loop(self, session: Optional[Any]) -> None:
        """
        Bucle Soberano de Tesorería M4:
        Vigila cada 4 segundos la transición por los Tramos de Interés Compuesto hacia el
        ROI/PnL esperado del mercado y ejecuta el cierre atómico al alcanzar la meta.
        """
        while self._running:
            try:
                await asyncio.sleep(4.0)

                async with self._execution_lock:
                    # 0. Supervisión del Botón Rojo Quirúrgico M7 (Comandos SQLite WAL + Monitoreo +3% ROI)
                    try:
                        pending_cmds = self.m7_red_button.fetch_and_ack_pending_commands()
                        for cmd in pending_cmds:
                            cmd_sym = str(
                                cmd.get("target_symbol") or cmd.get("symbol") or ""
                            ).upper()
                            cmd_mode = str(
                                cmd.get("mode", "RED_BUTTON_3PCT_PROFIT_CLOSE")
                            ).upper()
                            cmd_reason = str(cmd.get("reason", "WAL_QUEUE"))
                            cmd_tp_pct = float(cmd.get("target_profit_pct", 3.0))
                            if cmd_sym and cmd_sym != "ALL":
                                self.active_bots.pop(cmd_sym, None)
                                self.m4_treasury.correlation_guard.unregister_bot(cmd_sym)
                                await self.m7_red_button.surgical_close_symbol(
                                    symbol=cmd_sym,
                                    reason=cmd_reason,
                                    mode=cmd_mode,
                                    target_profit_pct=cmd_tp_pct,
                                    force_immediate="IMMEDIATE" in cmd_mode,
                                    session=session,
                                    notify_orchestrator=False,
                                )
                            elif cmd_sym == "ALL":
                                for s_all in list(self.active_bots.keys()):
                                    self.active_bots.pop(s_all, None)
                                    self.m4_treasury.correlation_guard.unregister_bot(s_all)
                                    await self.m7_red_button.surgical_close_symbol(
                                        symbol=s_all,
                                        reason=cmd_reason,
                                        mode=cmd_mode,
                                        target_profit_pct=cmd_tp_pct,
                                        force_immediate="IMMEDIATE" in cmd_mode,
                                        session=session,
                                        notify_orchestrator=False,
                                    )
                        closed_by_m7 = await self.m7_red_button.poll_and_close_armed_positions_at_3pct(
                            session=session
                        )
                        for m7_res in closed_by_m7:
                            c_sym = str(getattr(m7_res, "symbol", "")).upper()
                            if c_sym:
                                self.active_bots.pop(c_sym, None)
                                self.m4_treasury.correlation_guard.unregister_bot(c_sym)
                    except Exception as exc:
                        logger.debug("⚠️ [M7 Red Button Poll]: %s", exc)

                    if not self.active_bots:
                        continue

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

                        # 1. ¿Subió de Tramo Compuesto?
                        if action == BotEquityAction.STAGE_UPGRADED_RESIZE.value:
                            ex.plan = new_plan
                            ctx.sheet_monitor.plan = new_plan
                            ex.active_buys = list(new_plan.active_buy_window)
                            ex.active_sells = list(new_plan.active_sell_window)
                            await self.m5_broker.sync_active_limit_window(
                                symbol=sym,
                                buy_levels=new_plan.active_buy_window,
                                sell_levels=new_plan.active_sell_window,
                                qty=new_plan.qty_per_line,
                                rules=ctx.rules,
                                cancel_existing=True,
                                current_position_amt=ex.net_inventory_qty,
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

                        # 2. ¿Alcanzó la Meta Esperada del Mercado, tocó el Piso Ratchet o Escudo Anti-Liq?
                        elif action in (
                            BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value,
                            BotEquityAction.TAKE_PROFIT_18_REACHED.value,
                            BotEquityAction.RATCHET_STOP_TRIGGERED.value,
                            BotEquityAction.PRE_LIQUIDATION_SHIELD_TRIGGERED.value,
                        ):
                            ex.plan = new_plan
                            ctx.sheet_monitor.plan = new_plan
                            audited_pnl = await self.m5_broker.atomic_close_and_purge_bot(
                                symbol=sym, reason=action, session=session
                            )
                            final_net_pnl = (
                                audited_pnl
                                if (not self.m5_broker.dry_run and abs(audited_pnl) > 0.0)
                                else round(ex.realized_pnl_usdt + ex.unrealized_pnl_usdt, 4)
                            )
                            try:
                                ex.record_session_closed_wal(
                                    close_reason=action,
                                    final_net_pnl=final_net_pnl,
                                    status="COMPLETED",
                                )
                            except Exception as exc:
                                logger.debug("⚠️ [M5 Session Close WAL %s]: %s", sym, exc)

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
                                symbol=sym, net_realized_pnl_usdt=final_net_pnl
                            )
                            self.active_bots.pop(sym, None)

                            new_phase = (
                                self.m4_treasury.resolve_phase_behavior_profile().phase.value
                            )
                            logger.info(
                                "🏆 [BOT %s FINALIZADO: %s] PnL Neto: %+.2f USDT | Nuevo Balance Billetera: $%.2f USDT | Fase: %s",
                                sym,
                                action,
                                final_net_pnl,
                                self.m4_treasury.wallet_balance_usdt,
                                new_phase,
                            )

                            if not self.active_bots:
                                await self.sync_universe_and_telemetry(session)
                                await self.deploy_eligible_grid_bots(session)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("⚠️ [Aviso en Bucle de Tesorería M4]: %s", exc)

    async def run(self) -> None:
        if aiohttp is not None:
            connector = aiohttp.TCPConnector(
                limit=30,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60.0,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=12, connect=5)
            headers = {
                "Connection": "keep-alive",
                "Accept-Encoding": "gzip, deflate",
            }
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers=headers
            ) as session:
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
            logger.info("🛑 [PROSPERITY MARK-2] Apagado limpio completado.")

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
