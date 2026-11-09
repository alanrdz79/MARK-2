# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python core_mark2_contracts.py
Contratos de datos inmutables en memoria RAM O(1) con __slots__ e Interfaces.
Incluye:
- Las 5 Fases de Capital basadas en el saldo disponible en billetera ($4 a $1,000,000 USDT):
  * Fase 1 (4 a 100 USDT): Mitigación de incertidumbre con datos de Binance + Meta 30d TRIPLICAR (3x).
  * Fase 2 (100 a 1,000 USDT): Expansión multiactivo + Meta 30d TRIPLICAR (3x).
  * Fase 3 (1,000 a 10,000 USDT): Aceleración estructural + Meta 30d DUPLICAR (2x).
  * Fase 4 (10,000 a 100,000 USDT): Consolidación institucional y cosecha mensual.
  * Fase 5 (100,000 a 1,000,000 USDT): Preservación patrimonial soberana.
- ROI y PnL Esperado específico en cada mercado (Diario, Ciclo y a 30 Días).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple


class GridDirection(str, Enum):
    NEUTRAL = "NEUTRAL"
    LONG = "LONG"
    SHORT = "SHORT"


class MarketSafetyTier(str, Enum):
    """
    División inteligente del escáner cripto en 3 Áreas basadas exclusivamente
    en la distribución empírica de los datos extraídos en vivo de la API de Binance
    (clustering K-Means 1D + ponderación estadística CRITIC sin constantes heurísticas):
    - ESTABLES: Menor varianza condicional (NATR/EWMA/BB), menor spread L1 y mayor liquidez 24h.
    - INTERMEDIOS: Volatilidad empírica y rotación intradía situadas en el régimen central de la API.
    - VOLATILES: Mayor excursión empírica (MAE/MFE, Path 24h y NATR) extraída de las velas de la API.
    """
    ESTABLES = "ESTABLES"
    INTERMEDIOS = "INTERMEDIOS"
    VOLATILES = "VOLATILES"

    # Alias de compatibilidad hacia atrás
    TIER_1_INSTITUTIONAL_SAFE = "ESTABLES"
    TIER_2_LIQUID_MAJOR = "ESTABLES"
    TIER_3_HIGH_ROTATION_MID = "INTERMEDIOS"
    TIER_4_VOLATILE_ALPHA = "VOLATILES"


class ProsperityCapitalPhase(str, Enum):
    """
    Cinco Fases Operativas basadas en el capital disponible en la billetera USDT:
    - Fase 1 (4 a 100 USDT): Mitigación de incertidumbre con datos de Binance + Triplicar capital (3x) en 30 días.
    - Fase 2 (100 a 1,000 USDT): Expansión multiactivo + Triplicar capital (3x) en 30 días.
    - Fase 3 (1,000 a 10,000 USDT): Aceleración estructural + Duplicar capital (2x) en 30 días.
    - Fase 4 (10,000 a 100,000 USDT): Consolidación institucional y cosecha mensual.
    - Fase 5 (100,000 a 1,000,000 USDT): Preservación patrimonial soberana.
    """
    PHASE_1_MITIGATION_4_TO_100 = "FASE_1_4_A_100_USDT"
    PHASE_2_EXPANSION_100_TO_1K = "FASE_2_100_A_1000_USDT"
    PHASE_3_ACCELERATION_1K_TO_10K = "FASE_3_1000_A_10000_USDT"
    PHASE_4_INSTITUTIONAL_10K_TO_100K = "FASE_4_10000_A_100000_USDT"
    PHASE_5_SOVEREIGN_100K_TO_1M = "FASE_5_100000_A_1000000_USDT"

    # Alias de compatibilidad hacia atrás
    PHASE_1_MICRO_IGNITION = "FASE_1_4_A_100_USDT"
    PHASE_2_MULTI_CONSOLIDATION = "FASE_2_100_A_1000_USDT"
    PHASE_3_INSTITUTIONAL_HARVEST = "FASE_3_1000_A_10000_USDT"
    PHASE_4_SOVEREIGN_PRESERVATION = "FASE_4_10000_A_100000_USDT"


class CompoundingStage(str, Enum):
    """
    Tramos progresivos de interés compuesto y protección de capital (Ratchet Lock)
    hacia el ROI y PnL esperado del mercado y la meta de 30 días de la Fase (3x en Fases 1-2, 2x en Fase 3).
    """
    STAGE_1_MITIGATION_0_TO_33PCT = "TRAMO_1_MITIGACION_0_A_33PCT_META"
    STAGE_2_EXPANSION_33_TO_66PCT = "TRAMO_2_EXPANSION_33_A_66PCT_META"
    STAGE_3_HARVEST_66_TO_100PCT = "TRAMO_3_COSECHA_66_A_100PCT_META"
    TARGET_REACHED_EXPECTED_ROI = "META_MERCADO_COMPLETADA"

    # Alias de compatibilidad
    STAGE_1_IGNITION_6_TO_9 = "TRAMO_1_MITIGACION_0_A_33PCT_META"
    STAGE_2_EXPANSION_9_TO_12 = "TRAMO_2_EXPANSION_33_A_66PCT_META"
    STAGE_3_HARVEST_12_TO_18 = "TRAMO_3_COSECHA_66_A_100PCT_META"
    TARGET_REACHED_18_USD = "META_MERCADO_COMPLETADA"


@dataclass(frozen=True, slots=True)
class ExchangeContractRulePOD:
    """Reglas físicas de /fapi/v1/exchangeInfo para validación pre-trade."""
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float
    max_num_orders: int
    maint_margin_rate: float = 0.005
    max_leverage: int = 125


@dataclass(frozen=True, slots=True)
class Market3VTelemetryPOD:
    """Instantánea O(1) fusionando Klines, BookTicker L1/L2, Derivados y Ciencia de Datos Cuantitativa."""
    symbol: str
    timestamp_ms: int
    last_price: float
    mark_price: float
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    quote_vol_24h: float
    atr_14: float
    natr_14_pct: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    hurst_exp: float
    ema50_1h_slope_pct: float
    quote_obi_l1: float
    cvd_norm_3: float
    funding_rate: float
    mae_q65_pct: float
    mfe_q85_pct: float
    mfe_q92_pct: float
    intraday_path_pct: float
    ewma_vol_pct: float = 0.0
    variance_ratio_q4: float = 1.0
    flow_toxicity_z: float = 0.0
    bayesian_grid_survival_prob: float = 0.85


@dataclass(frozen=True, slots=True)
class RankedMarketCandidatePOD:
    """
    Clasificación de cada mercado con Certidumbre (%), Apalancamiento/Riesgo de la Fase,
    ROI/PnL Esperado Diario y a 30 Días de cada mercado, y Días Estimados para cumplir
    el multiplicador de la Fase (3x en Fases 1 y 2, 2x en Fase 3).
    """
    symbol: str
    safety_tier: MarketSafetyTier
    risk_score_ivr: float
    min_line_notional_usdt: float
    min_capital_for_grid_usdt: float
    viable_for_6usd_bot: bool
    est_daily_crosses: float
    est_days_6_to_18_usd: float  # Días estimados a la meta esperada del mercado
    status_reason: str
    expected_daily_pnl_usdt: float = 0.0
    expected_daily_roi_pct: float = 0.0
    expected_market_pnl_usdt: float = 0.0
    expected_market_roi_pct: float = 100.0
    expected_30d_pnl_usdt: float = 0.0
    expected_30d_roi_pct: float = 100.0
    uncertainty_score: float = 0.20
    confidence_pct: float = 80.0
    recommended_phase_leverage: int = 15
    recommended_risk_pct: float = 55.0
    phase_target_multiplier: float = 3.0  # 3.0x en Fases 1 y 2 | 2.0x en Fase 3
    est_days_to_double_capital: float = 30.0  # Días estimados para cumplir la meta de la Fase (3x o 2x)


@dataclass(frozen=True, slots=True)
class GeometricGridPlanPOD:
    """
    Especificación matemática completa del Grid Geométrico autocalculado por MARK-2.
    Integra Certidumbre (%), Apalancamiento/Riesgo según la Fase (1 a 5), ROI/PnL Esperado
    Diario y a 30 Días de cada mercado, y Días Estimados (Meta 3x en Fases 1-2, 2x en Fase 3).
    """
    symbol: str
    safety_tier: MarketSafetyTier
    direction: GridDirection
    compounding_stage: CompoundingStage
    initial_capital_usdt: float
    current_equity_usdt: float
    target_equity_usdt: float
    equity_ratchet_floor_usdt: float
    leverage: int
    lower_price: float
    upper_price: float
    total_virtual_grids: int
    active_window_orders: int
    geometric_step_pct: float
    net_profit_per_cross_pct: float
    net_profit_per_cross_usdt: float
    qty_per_line: float
    notional_per_line_usdt: float
    all_virtual_levels: Tuple[float, ...]
    active_buy_window: Tuple[float, ...]
    active_sell_window: Tuple[float, ...]
    tp_price: float
    tp_pnl_usdt: float
    tp_roi_pct: float
    sl_price: float
    sl_pnl_usdt: float
    sl_roi_pct: float
    est_daily_crosses: float
    est_days_to_18_usd: float  # Días estimados para alcanzar la meta esperada del mercado
    trailing_up: bool
    trailing_down: bool
    effective_initial_leverage: float = 1.0
    max_adverse_effective_leverage: float = 3.0
    estimated_liquidation_price: float = 0.0
    liquidation_safety_buffer_pct: float = 100.0
    anti_liquidation_reserve_usdt: float = 1.20
    active_phase: str = "FASE_1_4_A_100_USDT"
    phase_operational_risk_pct: float = 55.0
    phase_target_multiplier: float = 3.0  # 3.0x en Fases 1 y 2 | 2.0x en Fase 3
    uncertainty_score: float = 0.20
    confidence_pct: float = 80.0
    expected_daily_pnl_usdt: float = 0.0
    expected_daily_roi_pct: float = 0.0
    expected_30d_pnl_usdt: float = 0.0
    expected_30d_roi_pct: float = 100.0
    min_30d_doubling_target_usdt: float = 0.0  # Equidad mínima objetivo en 30d según Fase (3x en F1-F2, 2x en F3)
    m8_high_risk_plan: Optional["HighRiskAcceleratedOrderPlanPOD"] = None


class HighRiskDDMStatus(str, Enum):
    """Estado del Detector Estadístico Binomial de Deriva (Gama et al., 2004)."""
    IN_CONTROL = "IN_CONTROL"
    WARNING_2SIGMA = "WARNING_2SIGMA"
    DRIFT_HALT_3SIGMA = "DRIFT_HALT_3SIGMA"


@dataclass(frozen=True, slots=True)
class BinomialDDMStatePOD:
    """Instantánea auditable del control dual DDM (Deriva Binomial + MDD 7D)."""
    n_closed_cycles: int
    error_count: int
    error_rate_p: float
    std_error_s: float
    p_min: float
    s_min: float
    regime_status: HighRiskDDMStatus
    mdd_7d_pct: float


@dataclass(frozen=True, slots=True)
class HighRiskMarketProfilePOD:
    """
    Perfil empírico de microestructura (0% constantes heurísticas) extraído de datos
    reales de Binance / SQLite WAL con Winsorización Tukey 1.5*IQR, K-Means Silhouette
    y pesos objetivos CRITIC (Diakoulaki et al., 1995).
    """
    symbol: str
    rv_24h_pct: float
    natr_5m_pct: float
    variance_ratio_1h: float
    winsorized_range_1h_q50_pct: float
    winsorized_mfe_1h_q60_pct: float
    winsorized_mae_1h_q70_pct: float
    max_gap_winsorized_pct: float
    quote_vol_24h_usdt: float
    mean_abs_funding_rate: float
    effective_min_notional_usdt: float
    silhouette_cluster_id: int
    silhouette_score: float
    critic_weights: Tuple[float, float, float, float]
    critic_score: float
    real_maker_fee: float
    real_taker_fee: float
    eligible_for_10usdt: bool
    exclusion_reason: str = ""


@dataclass(frozen=True, slots=True)
class HighRiskAcceleratedOrderPlanPOD:
    """
    Especificación de orden/bot de Alto Riesgo para aceleración de capital pequeño ($10.00 USDT)
    cuantizada a tick_size y step_size reales, con garantía matemática P_liq < P_SL en ISOLATED.
    """
    symbol: str
    execution_mode: str
    margin_type: str
    wallet_capital_usdt: float
    isolated_leverage: int
    effective_wallet_leverage: float
    mark_price: float
    anchor_buy_gtx_price: float
    anchor_sell_gtx_price: float
    macro_step_pct: float
    order_qty: float
    notional_usdt: float
    isolated_margin_usdt: float
    tp_limit_gtx_price: float
    tp_pct: float
    sl_stop_market_price: float
    sl_pct: float
    breakeven_trigger_price: float
    breakeven_lock_price: float
    liquidation_price: float
    expected_net_tp_usdt: float
    expected_net_sl_usdt: float
    expected_wallet_roi_tp_pct: float
    expected_isolated_roi_tp_pct: float
    expected_wallet_roi_sl_pct: float
    roundtrip_fee_tp_usdt: float
    fee_to_gross_tp_pct: float
    open_position_on_creation: bool = False
    critic_score: float = 0.0
    rv_24h_pct: float = 0.0
    ddm_status: HighRiskDDMStatus = HighRiskDDMStatus.IN_CONTROL
    expected_mdd_7d_pct: float = 0.0


class IGridBrokerExecutionPort(Protocol):
    """Puerto abstracto hacia el adaptador de Binance Futures USDⓈ-M (Módulo 5)."""
    async def configure_isolated_and_leverage(self, symbol: str, leverage: int) -> bool: ...
    async def sync_active_limit_window(
        self, symbol: str, buy_levels: Tuple[float, ...], sell_levels: Tuple[float, ...], qty: float
    ) -> int: ...
    async def place_global_protection_brackets(
        self, symbol: str, direction: GridDirection, sl_price: float, tp_price: float
    ) -> bool: ...
    async def atomic_close_and_purge_bot(self, symbol: str, reason: str) -> float: ...
