# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python m4_risk_treasury_6to18.py
Módulo 4 Completo: El Soberano del Capital, Gobernanza de las 5 Fases por Capital Disponible
en Billetera, Meta a 30 Días (3x en Fases 1 y 2, 2x en Fase 3), Mitigación de Incertidumbre
con Datos de Binance, ROI/PnL Esperado por Mercado y Escudo Anti-Liquidación.

Fases de Capital en Billetera (USDT):
- Fase 1 (4 a 100 USDT): Fase actual. Mitigar/disminuir la incertidumbre mediante los datos
  reales de Binance y TRIPLICAR (3x) el capital inicial de la cuenta (+200% ROI mínimo) en 30 días.
- Fase 2 (100 a 1,000 USDT): Expansión multiactivo. TRIPLICAR (3x) el capital inicial de la cuenta
  en ese momento (+200% ROI mínimo) en 30 días.
- Fase 3 (1,000 a 10,000 USDT): Aceleración estructural. DUPLICAR (2x) el capital inicial de la cuenta
  en ese momento (+100% ROI mínimo) en 30 días.
- Fase 4 (10,000 a 100,000 USDT): Consolidación Institucional + Retiro mensual del 20% + Earn 15%.
- Fase 5 (100,000 a 1,000,000 USDT): Preservación Patrimonial Soberana + Retiro 20% + Earn 30%.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from core_mark2_contracts import (
    CompoundingStage,
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    Market3VTelemetryPOD,
    ProsperityCapitalPhase,
)
from m3_grid_decision_orchestrator import Mark2GridDecisionOrchestrator
from m4_phase_behavior_governor import Mark2PhaseBehaviorGovernor

logger = logging.getLogger("M4_RiskTreasury")


# ============================================================================
# 1. ENUMERACIONES Y CONTRATOS INMUTABLES DE TESORERÍA Y 5 FASES DE CAPITAL
# ============================================================================

class BotEquityAction(str, Enum):
    HOLD_ACTIVE_GRID = "HOLD_ACTIVE_GRID"
    STAGE_UPGRADED_RESIZE = "STAGE_UPGRADED_RESIZE"
    TAKE_PROFIT_18_REACHED = "TAKE_PROFIT_EXPECTED_REACHED"
    TAKE_PROFIT_EXPECTED_REACHED = "TAKE_PROFIT_EXPECTED_REACHED"
    TRAILING_SUPER_ROI_LOCKED = "TRAILING_SUPER_ROI_LOCKED"
    RATCHET_STOP_TRIGGERED = "RATCHET_STOP_TRIGGERED"
    PRE_LIQUIDATION_SHIELD_TRIGGERED = "PRE_LIQUIDATION_SHIELD_TRIGGERED"


@dataclass(frozen=True, slots=True)
class PhaseBehaviorProfile:
    """
    Reglas operativas inmutables que gobiernan el riesgo en operaciones, el apalancamiento
    y el objetivo a 30 días (3x en Fases 1-2, 2x en Fase 3) según la Fase de Capital.
    """
    phase: ProsperityCapitalPhase
    phase_min_capital_usdt: float
    phase_max_capital_usdt: float
    min_leverage: int
    recommended_leverage: int
    max_leverage: int
    min_grids_per_bot: int
    max_grids_per_bot: int
    max_concurrent_bots: int
    capital_allocation_per_bot_usdt: float
    target_roi_per_bot_pct: float  # +200.0% (3x) en Fases 1-2 | +100.0% (2x) en Fase 3 en 30 días
    max_account_risk_per_bot_pct: float
    allowed_safety_tiers: Tuple[str, ...]
    unlocked_major_symbols: Tuple[str, ...]
    allow_alpha_high_roi_bots: bool
    withdrawal_unlocked: bool
    monthly_profit_dividend_pct: float
    fixed_income_earn_migration_pct: float
    max_effective_inventory_leverage: float = 3.5
    horizon_days: int = 30
    phase_target_multiplier: float = 3.0  # 3.0x en Fases 1-2, 2.0x en Fase 3
    must_double_in_30d: bool = True
    min_30d_target_equity_usdt: float = 12.0
    min_30d_target_pnl_usdt: float = 8.0
    required_daily_compound_roi_pct: float = 3.73
    min_confidence_pct: float = 68.0


@dataclass(frozen=True, slots=True)
class MonthlyTreasurySettlementPOD:
    """Reporte inmutable del corte contable mensual de tesorería (30 días)."""
    phase: str
    phase_initial_capital_usdt: float
    gross_monthly_profit_usdt: float
    achieved_30d_roi_pct: float
    doubling_goal_met: bool
    dividend_withdrawn_20pct_usdt: float
    migrated_to_earn_cetes_usdt: float
    reinvested_compounding_usdt: float
    new_trading_wallet_balance_usdt: float
    summary_message: str


@dataclass(frozen=True, slots=True)
class PreTradeSovereignVerdictPOD:
    """Dictamen de la Aduana Soberana M4 antes de desplegar un nuevo Bot Grid."""
    approved: bool
    assigned_capital_usdt: float
    approved_leverage: int
    approved_risk_pct: float
    active_phase: ProsperityCapitalPhase
    anti_martingale_applied: bool
    uncertainty_mitigated: bool
    reason: str


# ============================================================================
# 2. GUARDIÁN DE CORRELACIÓN CRUZADA EN MEMORIA RAM O(1)
# ============================================================================

class RollingCorrelationGuardRAM:
    """
    Supervisa en RAM la correlación de Pearson sobre retornos logarítmicos
    entre múltiples Bots Grid activos para evitar exposición colineal (rho >= 0.70).
    """

    def __init__(self, max_correlation_rho: float = 0.70, window_bars: int = 30):
        self.max_correlation_rho = max_correlation_rho
        self.window_bars = window_bars
        self.price_series_ram: Dict[str, Deque[float]] = {}
        self.active_bot_directions: Dict[str, GridDirection] = {}

    def ingest_close_price(self, symbol: str, close_price: float) -> None:
        if close_price <= 0.0 or math.isnan(close_price):
            return
        if symbol not in self.price_series_ram:
            self.price_series_ram[symbol] = deque(maxlen=self.window_bars + 2)
        self.price_series_ram[symbol].append(close_price)

    def register_active_bot(self, symbol: str, direction: GridDirection) -> None:
        self.active_bot_directions[symbol] = direction

    def unregister_active_bot(self, symbol: str) -> None:
        self.active_bot_directions.pop(symbol, None)

    @staticmethod
    def compute_log_returns(prices: List[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(prices, dtype=np.float64)
        arr = np.maximum(arr, 1e-12)
        return np.diff(np.log(arr))

    def verify_pair_correlation(
        self,
        candidate_returns: np.ndarray,
        active_bots_returns: Dict[str, np.ndarray],
    ) -> Tuple[bool, str]:
        for active_sym, active_ret in active_bots_returns.items():
            m = min(len(candidate_returns), len(active_ret))
            if m < 20:
                continue
            r_cand = candidate_returns[-m:]
            r_act = active_ret[-m:]
            if np.std(r_cand) <= 1e-12 or np.std(r_act) <= 1e-12:
                continue
            corr_matrix = np.corrcoef(r_cand, r_act)
            rho = float(corr_matrix[0, 1])
            if not math.isnan(rho) and abs(rho) >= self.max_correlation_rho:
                return (
                    False,
                    f"VETO_CORRELACION_ALTA con {active_sym} (rho={rho:.2f} >= {self.max_correlation_rho:.2f})",
                )
        return (True, "CORRELACION_APROBADA")


# ============================================================================
# 3. SOBERANO DEL CAPITAL: 5 FASES, MITIGACIÓN DE INCERTIDUMBRE Y META 30 DÍAS
# ============================================================================

class Mark2RiskAndTreasuryController:
    """
    Controlador Soberano de Tesorería y Riesgo para PROSPERITY MARK-2:
    1. Gobierna las 5 Fases Financieras basadas en el capital disponible en billetera:
       - Fase 1: 4 a 100 USDT (Mitigar incertidumbre con datos de Binance + TRIPLICAR 3x en 30d).
       - Fase 2: 100 a 1,000 USDT (TRIPLICAR 3x en 30d con riesgo y apalancamiento reducidos).
       - Fase 3: 1,000 a 10,000 USDT (DUPLICAR 2x en 30d con apalancamiento moderado).
       - Fase 4: 10,000 a 100,000 USDT (Cosecha institucional + retiro 20% mensual).
       - Fase 5: 100,000 a 1,000,000 USDT (Preservación soberana + retiro 20% + Earn 30%).
    2. Ajusta el riesgo operativo y el apalancamiento en cada operación según la Fase activa y
       la certidumbre empírica del mercado en Binance.
    3. Calcula y persigue el ROI y PnL Esperado específico de cada mercado, aplicando el
       Trinquete de Equidad (Ratchet Lock) y el Escudo Anti-Liquidación.
    """

    DAILY_COMPOUND_FOR_30D_TRIPLING_PCT = round(((3.0 ** (1.0 / 30.0)) - 1.0) * 100.0, 4)  # ~3.7299%/día (3x en 30d)
    DAILY_COMPOUND_FOR_30D_DOUBLING_PCT = round(((2.0 ** (1.0 / 30.0)) - 1.0) * 100.0, 4)  # ~2.3374%/día (2x en 30d)

    def __init__(
        self,
        initial_wallet_balance_usdt: float = 18.00,
        min_bot_capital_usdt: float = 4.00,
        target_bot_capital_usdt: Optional[float] = None,
        max_correlation_rho: float = 0.70,
        max_daily_global_dd_pct: float = 0.05,
        max_p99_latency_ms: float = 150.0,
        enable_super_roi_trailing: bool = False,
    ):
        self.wallet_balance_usdt: float = max(4.0, round(initial_wallet_balance_usdt, 4))
        self.phase_initial_capital_usdt: float = self.wallet_balance_usdt
        self.daily_start_balance_usdt: float = self.wallet_balance_usdt
        self.peak_wallet_balance_usdt: float = self.wallet_balance_usdt

        self.min_bot_capital_usdt: float = max(4.00, min_bot_capital_usdt)
        self.target_bot_capital_usdt: Optional[float] = target_bot_capital_usdt
        self.max_correlation_rho: float = max_correlation_rho
        self.max_daily_global_dd_pct: float = max_daily_global_dd_pct
        self.max_p99_latency_ms: float = max_p99_latency_ms
        self.enable_super_roi_trailing: bool = enable_super_roi_trailing

        self.daily_trading_pnl_usdt: float = 0.0
        self.monthly_net_profit_usdt: float = 0.0
        self.total_external_deposits_usdt: float = 0.0
        self.total_withdrawn_dividends_usdt: float = 0.0
        self.total_migrated_to_earn_usdt: float = 0.0

        self.consecutive_bot_losses: int = 0
        self.consecutive_api_errors: int = 0
        self.kill_switch_engaged: bool = False
        self.rtt_latency_ms_window: Deque[float] = deque(maxlen=50)

        self.bot_peak_equity_map: Dict[str, float] = {}

        self.correlation_guard = RollingCorrelationGuardRAM(
            max_correlation_rho=max_correlation_rho
        )
        self.phase_governor = Mark2PhaseBehaviorGovernor(
            initial_balance_usdt=self.wallet_balance_usdt
        )

    # ------------------------------------------------------------------------
    # A. GOBERNANZA DE LAS 5 FASES FINANCIERAS POR CAPITAL EN BILLETERA
    # ------------------------------------------------------------------------

    def resolve_phase_behavior_profile(
        self, balance_override_usdt: Optional[float] = None
    ) -> PhaseBehaviorProfile:
        bal = max(
            4.0,
            balance_override_usdt
            if balance_override_usdt is not None
            else self.wallet_balance_usdt,
        )

        # FASE 1: 4 a 100 USDT (Fase actual -> Mitigar incertidumbre + TRIPLICAR 3x en 30 días)
        if bal < 100.0:
            base_min = max(4.00, self.min_bot_capital_usdt)
            if bal < 12.0:
                max_bots = 1
                bot_cap = round(bal, 2)
            elif bal < 30.0:
                max_bots = min(3, max(1, int(bal // base_min)))
                bot_cap = round(max(base_min, bal / max_bots), 2)
            else:
                max_bots = min(4, int(bal // 10.0))
                bot_cap = round(max(base_min, bal * 0.25), 2)

            rec_lev = 20 if bal < 15.0 else (15 if bal < 50.0 else 12)
            max_lev = 25 if bal < 25.0 else 20

            return PhaseBehaviorProfile(
                phase=ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100,
                phase_min_capital_usdt=4.0,
                phase_max_capital_usdt=100.0,
                min_leverage=5,
                recommended_leverage=rec_lev,
                max_leverage=max_lev,
                min_grids_per_bot=8,
                max_grids_per_bot=45,
                max_concurrent_bots=max(1, max_bots),
                capital_allocation_per_bot_usdt=bot_cap,
                target_roi_per_bot_pct=200.0,  # Triplicar capital inicial (3x = +200% ROI mínimo en 30d)
                max_account_risk_per_bot_pct=55.0,
                allowed_safety_tiers=("ESTABLES", "INTERMEDIOS", "VOLATILES"),
                unlocked_major_symbols=("SOLUSDT", "BNBUSDT", "XRPUSDT", "SUIUSDT", "DOGEUSDT", "ADAUSDT"),
                allow_alpha_high_roi_bots=True,
                withdrawal_unlocked=False,
                monthly_profit_dividend_pct=0.0,
                fixed_income_earn_migration_pct=0.0,
                max_effective_inventory_leverage=3.5,
                horizon_days=30,
                phase_target_multiplier=3.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 3.0, 2),
                min_30d_target_pnl_usdt=round(bal * 2.0, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_TRIPLING_PCT,
                min_confidence_pct=68.0,
            )

        # FASE 2: 100 a 1,000 USDT (Expansión Multiactivo -> TRIPLICAR 3x en 30 días)
        if bal < 1_000.0:
            bot_cap = round(max(25.0, bal * 0.25), 2)
            return PhaseBehaviorProfile(
                phase=ProsperityCapitalPhase.PHASE_2_EXPANSION_100_TO_1K,
                phase_min_capital_usdt=100.0,
                phase_max_capital_usdt=1_000.0,
                min_leverage=4,
                recommended_leverage=10,
                max_leverage=15,
                min_grids_per_bot=20,
                max_grids_per_bot=75,
                max_concurrent_bots=4,
                capital_allocation_per_bot_usdt=bot_cap,
                target_roi_per_bot_pct=200.0,  # Triplicar capital inicial (3x = +200% ROI mínimo en 30d)
                max_account_risk_per_bot_pct=3.5,
                allowed_safety_tiers=("ESTABLES", "INTERMEDIOS", "VOLATILES"),
                unlocked_major_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "SUIUSDT"),
                allow_alpha_high_roi_bots=True,
                withdrawal_unlocked=False,
                monthly_profit_dividend_pct=0.0,
                fixed_income_earn_migration_pct=0.0,
                max_effective_inventory_leverage=2.8,
                horizon_days=30,
                phase_target_multiplier=3.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 3.0, 2),
                min_30d_target_pnl_usdt=round(bal * 2.0, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_TRIPLING_PCT,
                min_confidence_pct=70.0,
            )

        # FASE 3: 1,000 a 10,000 USDT (Aceleración Estructural -> DUPLICAR 2x en 30 días)
        if bal < 10_000.0:
            bot_cap = round(max(200.0, bal * 0.20), 2)
            return PhaseBehaviorProfile(
                phase=ProsperityCapitalPhase.PHASE_3_ACCELERATION_1K_TO_10K,
                phase_min_capital_usdt=1_000.0,
                phase_max_capital_usdt=10_000.0,
                min_leverage=3,
                recommended_leverage=6,
                max_leverage=10,
                min_grids_per_bot=40,
                max_grids_per_bot=110,
                max_concurrent_bots=4,
                capital_allocation_per_bot_usdt=bot_cap,
                target_roi_per_bot_pct=100.0,  # Duplicar capital inicial (2x = +100% ROI mínimo en 30d)
                max_account_risk_per_bot_pct=2.0,
                allowed_safety_tiers=("ESTABLES", "INTERMEDIOS"),
                unlocked_major_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
                allow_alpha_high_roi_bots=True,
                withdrawal_unlocked=False,
                monthly_profit_dividend_pct=0.0,
                fixed_income_earn_migration_pct=0.0,
                max_effective_inventory_leverage=2.0,
                horizon_days=30,
                phase_target_multiplier=2.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 2.0, 2),
                min_30d_target_pnl_usdt=round(bal, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_DOUBLING_PCT,
                min_confidence_pct=72.0,
            )

        # FASE 4: 10,000 a 100,000 USDT (Cosecha Institucional -> Retiro 20%)
        if bal < 100_000.0:
            bot_cap = round(bal * 0.20, 2)
            return PhaseBehaviorProfile(
                phase=ProsperityCapitalPhase.PHASE_4_INSTITUTIONAL_10K_TO_100K,
                phase_min_capital_usdt=10_000.0,
                phase_max_capital_usdt=100_000.0,
                min_leverage=2,
                recommended_leverage=4,
                max_leverage=5,
                min_grids_per_bot=70,
                max_grids_per_bot=150,
                max_concurrent_bots=4,
                capital_allocation_per_bot_usdt=bot_cap,
                target_roi_per_bot_pct=35.0,
                max_account_risk_per_bot_pct=1.0,
                allowed_safety_tiers=("ESTABLES", "INTERMEDIOS"),
                unlocked_major_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
                allow_alpha_high_roi_bots=False,
                withdrawal_unlocked=True,
                monthly_profit_dividend_pct=0.20,
                fixed_income_earn_migration_pct=0.15,
                max_effective_inventory_leverage=1.5,
                horizon_days=30,
                phase_target_multiplier=1.35,
                must_double_in_30d=False,
                min_30d_target_equity_usdt=round(bal * 1.35, 2),
                min_30d_target_pnl_usdt=round(bal * 0.35, 2),
                required_daily_compound_roi_pct=round(((1.35 ** (1.0 / 30.0)) - 1.0) * 100.0, 4),
                min_confidence_pct=75.0,
            )

        # FASE 5: 100,000 a 1,000,000 USDT (Preservación Soberana)
        bot_cap = round(bal * 0.15, 2)
        return PhaseBehaviorProfile(
            phase=ProsperityCapitalPhase.PHASE_5_SOVEREIGN_100K_TO_1M,
            phase_min_capital_usdt=100_000.0,
            phase_max_capital_usdt=1_000_000.0,
            min_leverage=1,
            recommended_leverage=2,
            max_leverage=3,
            min_grids_per_bot=90,
            max_grids_per_bot=169,
            max_concurrent_bots=3,
            capital_allocation_per_bot_usdt=bot_cap,
            target_roi_per_bot_pct=15.0,
            max_account_risk_per_bot_pct=0.5,
            allowed_safety_tiers=("ESTABLES",),
            unlocked_major_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            allow_alpha_high_roi_bots=False,
            withdrawal_unlocked=True,
            monthly_profit_dividend_pct=0.20,
            fixed_income_earn_migration_pct=0.30,
            max_effective_inventory_leverage=1.0,
            horizon_days=30,
            phase_target_multiplier=1.15,
            must_double_in_30d=False,
            min_30d_target_equity_usdt=round(bal * 1.15, 2),
            min_30d_target_pnl_usdt=round(bal * 0.15, 2),
            required_daily_compound_roi_pct=round(((1.15 ** (1.0 / 30.0)) - 1.0) * 100.0, 4),
            min_confidence_pct=80.0,
        )

    # ------------------------------------------------------------------------
    # B. MÁQUINA DE ESTADOS DEL TRINQUETE HACIA LA META DEL MERCADO Y ESCUDO ANTI-LIQ
    # ------------------------------------------------------------------------

    def evaluate_bot_equity_transition(
        self,
        current_plan: GeometricGridPlanPOD,
        realized_pnl_usdt: float,
        unrealized_pnl_usdt: float,
        rules: ExchangeContractRulePOD,
        tel: Market3VTelemetryPOD,
        orchestrator: Mark2GridDecisionOrchestrator,
    ) -> Tuple[str, GeometricGridPlanPOD]:
        """
        Evalúa en tiempo constante O(1) la evolución patrimonial de un Bot Grid hacia el
        ROI y PnL Esperado de su mercado y la meta de 30 días de la Fase (3x en F1-F2, 2x en F3).
        """
        sym = current_plan.symbol
        live_equity = round(
            current_plan.initial_capital_usdt + realized_pnl_usdt + unrealized_pnl_usdt,
            4,
        )
        realized_equity = round(
            current_plan.initial_capital_usdt + realized_pnl_usdt,
            4,
        )

        prev_peak = self.bot_peak_equity_map.get(sym, current_plan.initial_capital_usdt)
        if live_equity > prev_peak:
            self.bot_peak_equity_map[sym] = live_equity
            prev_peak = live_equity

        # 1. ¿Alcanzó o superó el ROI y PnL Esperado del mercado (target_equity_usdt)?
        if live_equity >= current_plan.target_equity_usdt:
            if not self.enable_super_roi_trailing:
                completed_plan = orchestrator.build_geometric_grid_6_to_18(
                    rules=rules,
                    tel=tel,
                    safety_tier=current_plan.safety_tier,
                    initial_capital_usdt=current_plan.initial_capital_usdt,
                    current_equity_usdt=live_equity,
                    target_equity_usdt=current_plan.target_equity_usdt,
                    forced_direction=current_plan.direction,
                    max_leverage=current_plan.leverage,
                    wallet_balance_usdt=self.wallet_balance_usdt,
                )
                return (BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value, completed_plan)

            dynamic_super_floor = max(
                current_plan.target_equity_usdt, round(prev_peak * 0.92, 2)
            )
            if live_equity <= dynamic_super_floor and prev_peak > current_plan.target_equity_usdt * 1.05:
                completed_plan = orchestrator.build_geometric_grid_6_to_18(
                    rules=rules,
                    tel=tel,
                    safety_tier=current_plan.safety_tier,
                    initial_capital_usdt=current_plan.initial_capital_usdt,
                    current_equity_usdt=live_equity,
                    target_equity_usdt=current_plan.target_equity_usdt,
                    forced_direction=current_plan.direction,
                    max_leverage=current_plan.leverage,
                    wallet_balance_usdt=self.wallet_balance_usdt,
                )
                return (BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value, completed_plan)

        # 2. Escudo Preventivo Anti-Liquidación (Zero-Liquidation Shield)
        if current_plan.estimated_liquidation_price > 0.0 and tel.mark_price > 0.0:
            safety_margin_price = max(tel.atr_14 * 1.25, tel.mark_price * 0.01)
            if current_plan.direction in (GridDirection.LONG, GridDirection.NEUTRAL):
                if tel.mark_price <= (current_plan.estimated_liquidation_price + safety_margin_price):
                    return (BotEquityAction.PRE_LIQUIDATION_SHIELD_TRIGGERED.value, current_plan)
            else:
                if tel.mark_price >= (current_plan.estimated_liquidation_price - safety_margin_price):
                    return (BotEquityAction.PRE_LIQUIDATION_SHIELD_TRIGGERED.value, current_plan)

        # 3. ¿Tocó el Piso Protegido del Trinquete (Ratchet Floor) o Stop Loss Estructural?
        if live_equity <= current_plan.equity_ratchet_floor_usdt:
            return (BotEquityAction.RATCHET_STOP_TRIGGERED.value, current_plan)

        # 4. ¿Cruza con ganancia realizada hacia el Tramo 2 (25% de la meta) o Tramo 3 (50% de la meta)?
        total_target_pnl = max(0.01, current_plan.target_equity_usdt - current_plan.initial_capital_usdt)
        stage_2_threshold = round(current_plan.initial_capital_usdt + (total_target_pnl * 0.25), 2)
        stage_3_threshold = round(current_plan.initial_capital_usdt + (total_target_pnl * 0.50), 2)

        if (
            current_plan.compounding_stage == CompoundingStage.STAGE_1_MITIGATION_0_TO_33PCT
            and realized_equity >= stage_2_threshold
        ) or (
            current_plan.compounding_stage == CompoundingStage.STAGE_2_EXPANSION_33_TO_66PCT
            and realized_equity >= stage_3_threshold
        ):
            upgraded_plan = orchestrator.build_geometric_grid_6_to_18(
                rules=rules,
                tel=tel,
                safety_tier=current_plan.safety_tier,
                initial_capital_usdt=current_plan.initial_capital_usdt,
                current_equity_usdt=realized_equity,
                target_equity_usdt=current_plan.target_equity_usdt,
                forced_direction=current_plan.direction,
                max_leverage=current_plan.leverage,
                wallet_balance_usdt=self.wallet_balance_usdt,
            )
            logger.info(
                "📈 [M4 SALTO DE TRAMO EN %s | %s] Equidad Realizada=$%.2f USDT -> Nuevo Tramo=%s | Lote=%s ($%.2f USDT) | Nuevo Piso=$%.2f USDT",
                sym,
                upgraded_plan.active_phase,
                realized_equity,
                upgraded_plan.compounding_stage.value,
                upgraded_plan.qty_per_line,
                upgraded_plan.notional_per_line_usdt,
                upgraded_plan.equity_ratchet_floor_usdt,
            )
            return (BotEquityAction.STAGE_UPGRADED_RESIZE.value, upgraded_plan)

        return (BotEquityAction.HOLD_ACTIVE_GRID.value, current_plan)

    # ------------------------------------------------------------------------
    # C. ADUANA PRE-TRADE POR FASE, MITIGACIÓN DE INCERTIDUMBRE Y CORRELACIÓN
    # ------------------------------------------------------------------------

    def authorize_new_grid_bot_deployment(
        self,
        candidate_symbol: str,
        active_bots_count: int,
        candidate_returns: Optional[np.ndarray] = None,
        active_bots_returns: Optional[Dict[str, np.ndarray]] = None,
        candidate_uncertainty_score: float = 0.20,
    ) -> PreTradeSovereignVerdictPOD:
        profile = self.resolve_phase_behavior_profile()

        if self.kill_switch_engaged or self.consecutive_api_errors >= 3:
            return PreTradeSovereignVerdictPOD(
                approved=False,
                assigned_capital_usdt=0.0,
                approved_leverage=1,
                approved_risk_pct=0.0,
                active_phase=profile.phase,
                anti_martingale_applied=False,
                uncertainty_mitigated=False,
                reason="VETO_KILL_SWITCH_O_ERRORES_API",
            )

        if active_bots_count >= profile.max_concurrent_bots:
            return PreTradeSovereignVerdictPOD(
                approved=False,
                assigned_capital_usdt=0.0,
                approved_leverage=1,
                approved_risk_pct=0.0,
                active_phase=profile.phase,
                anti_martingale_applied=False,
                uncertainty_mitigated=False,
                reason=f"VETO_CUPO_MAX_BOTS_FASE ({active_bots_count}/{profile.max_concurrent_bots})",
            )

        if candidate_returns is not None and active_bots_returns:
            corr_ok, corr_msg = self.check_multi_bot_correlation_guard(
                candidate_returns, active_bots_returns
            )
            if not corr_ok:
                return PreTradeSovereignVerdictPOD(
                    approved=False,
                    assigned_capital_usdt=0.0,
                    approved_leverage=1,
                    approved_risk_pct=0.0,
                    active_phase=profile.phase,
                    anti_martingale_applied=False,
                    uncertainty_mitigated=False,
                    reason=corr_msg,
                )

        unc_clamped = max(0.05, min(0.95, candidate_uncertainty_score))
        uncertainty_lev_factor = max(0.65, 1.0 - (0.30 * unc_clamped))
        approved_lev = max(
            profile.min_leverage,
            min(
                profile.max_leverage,
                int(round(profile.recommended_leverage * uncertainty_lev_factor)),
            ),
        )
        approved_risk_pct = round(
            max(0.25, profile.max_account_risk_per_bot_pct * max(0.55, 1.0 - (0.40 * unc_clamped))),
            2,
        )

        assigned_cap = profile.capital_allocation_per_bot_usdt
        anti_martingale = False

        if self.consecutive_bot_losses > 0:
            anti_martingale = True
            assigned_cap = max(4.00, min(self.min_bot_capital_usdt, round(assigned_cap * 0.50, 2)))
            approved_risk_pct = round(approved_risk_pct * 0.50, 2)

        if assigned_cap > self.wallet_balance_usdt:
            if self.wallet_balance_usdt >= 4.00:
                assigned_cap = round(self.wallet_balance_usdt, 2)
            else:
                return PreTradeSovereignVerdictPOD(
                    approved=False,
                    assigned_capital_usdt=0.0,
                    approved_leverage=1,
                    approved_risk_pct=0.0,
                    active_phase=profile.phase,
                    anti_martingale_applied=anti_martingale,
                    uncertainty_mitigated=True,
                    reason=f"VETO_SALDO_INSUFICIENTE (${self.wallet_balance_usdt:.2f} < $4.00 USDT)",
                )

        return PreTradeSovereignVerdictPOD(
            approved=True,
            assigned_capital_usdt=assigned_cap,
            approved_leverage=approved_lev,
            approved_risk_pct=approved_risk_pct,
            active_phase=profile.phase,
            anti_martingale_applied=anti_martingale,
            uncertainty_mitigated=True,
            reason=(
                f"AUTORIZADO_{profile.phase.value} | Lev={approved_lev}x (Max Fase={profile.max_leverage}x) | "
                f"Riesgo Op={approved_risk_pct:.2f}% | Meta 30d={profile.phase_target_multiplier:.0f}x (+{profile.target_roi_per_bot_pct:.0f}%)"
            ),
        )

    def check_multi_bot_correlation_guard(
        self,
        candidate_returns: np.ndarray,
        active_bots_returns: Dict[str, np.ndarray],
    ) -> Tuple[bool, str]:
        return self.correlation_guard.verify_pair_correlation(
            candidate_returns, active_bots_returns
        )

    # ------------------------------------------------------------------------
    # D. CONTABILIDAD DE DEPÓSITOS, CORTE DE 30 DÍAS Y RETIROS (FASES 4 Y 5)
    # ------------------------------------------------------------------------

    def register_bot_session_closed(self, symbol: str, net_realized_pnl_usdt: float) -> None:
        prev_phase = self.resolve_phase_behavior_profile().phase
        self.wallet_balance_usdt = round(self.wallet_balance_usdt + net_realized_pnl_usdt, 4)
        self.phase_governor.wallet_balance_usdt = self.wallet_balance_usdt
        self.daily_trading_pnl_usdt = round(self.daily_trading_pnl_usdt + net_realized_pnl_usdt, 4)
        self.bot_peak_equity_map.pop(symbol, None)
        self.correlation_guard.unregister_active_bot(symbol)

        if self.wallet_balance_usdt > self.peak_wallet_balance_usdt:
            self.peak_wallet_balance_usdt = self.wallet_balance_usdt

        new_phase = self.resolve_phase_behavior_profile().phase
        if new_phase != prev_phase:
            self.phase_initial_capital_usdt = self.wallet_balance_usdt

        if net_realized_pnl_usdt < 0.0:
            self.consecutive_bot_losses += 1
        else:
            self.consecutive_bot_losses = 0
            self.monthly_net_profit_usdt = round(
                self.monthly_net_profit_usdt + net_realized_pnl_usdt, 4
            )
            self.phase_governor.monthly_realized_profit_usdt = self.monthly_net_profit_usdt

    def deposit_external_capital(self, amount_usdt: float) -> float:
        if amount_usdt <= 0.0:
            return self.wallet_balance_usdt
        self.wallet_balance_usdt = round(self.wallet_balance_usdt + amount_usdt, 4)
        self.phase_governor.record_external_deposit(amount_usdt)
        self.phase_initial_capital_usdt = round(self.phase_initial_capital_usdt + amount_usdt, 4)
        self.daily_start_balance_usdt = round(self.daily_start_balance_usdt + amount_usdt, 4)
        self.total_external_deposits_usdt = round(
            self.total_external_deposits_usdt + amount_usdt, 4
        )
        if self.wallet_balance_usdt > self.peak_wallet_balance_usdt:
            self.peak_wallet_balance_usdt = self.wallet_balance_usdt
        return self.wallet_balance_usdt

    def reconcile_wallet_with_exchange(self, exchange_reported_balance_usdt: float) -> None:
        delta = round(exchange_reported_balance_usdt - self.wallet_balance_usdt, 4)
        if delta > 0.05:
            self.deposit_external_capital(delta)
        elif delta < -0.05:
            self.wallet_balance_usdt = round(exchange_reported_balance_usdt, 4)
            self.phase_governor.wallet_balance_usdt = self.wallet_balance_usdt
            self.phase_initial_capital_usdt = max(4.0, self.wallet_balance_usdt)
            self.daily_start_balance_usdt = max(
                4.0, round(self.daily_start_balance_usdt + delta, 4)
            )

    def process_monthly_treasury_settlement(self) -> MonthlyTreasurySettlementPOD:
        profile = self.resolve_phase_behavior_profile()
        gross_profit = max(0.0, self.monthly_net_profit_usdt)
        base_cap = max(4.0, self.phase_initial_capital_usdt - gross_profit)
        achieved_roi_30d = round((gross_profit / base_cap) * 100.0, 2)
        goal_ok = achieved_roi_30d >= profile.target_roi_per_bot_pct

        if not profile.withdrawal_unlocked or gross_profit <= 0.0:
            self.monthly_net_profit_usdt = 0.0
            self.phase_initial_capital_usdt = self.wallet_balance_usdt
            return MonthlyTreasurySettlementPOD(
                phase=profile.phase.value,
                phase_initial_capital_usdt=round(base_cap, 2),
                gross_monthly_profit_usdt=round(gross_profit, 2),
                achieved_30d_roi_pct=achieved_roi_30d,
                doubling_goal_met=goal_ok,
                dividend_withdrawn_20pct_usdt=0.0,
                migrated_to_earn_cetes_usdt=0.0,
                reinvested_compounding_usdt=round(gross_profit, 2),
                new_trading_wallet_balance_usdt=round(self.wallet_balance_usdt, 2),
                summary_message=(
                    f"[{profile.phase.value}] Corte de 30 Días: Utilidad=${gross_profit:.2f} USDT (+{achieved_roi_30d:.1f}% ROI | "
                    f"Meta {profile.phase_target_multiplier:.0f}x={'CUMPLIDA' if goal_ok else 'EN PROGRESO'}). 100% reinvertido para escalar de Fase."
                ),
            )

        dividend_20pct = round(gross_profit * profile.monthly_profit_dividend_pct, 2)
        self.wallet_balance_usdt = round(self.wallet_balance_usdt - dividend_20pct, 4)
        self.total_withdrawn_dividends_usdt = round(
            self.total_withdrawn_dividends_usdt + dividend_20pct, 4
        )

        remaining_profit = max(0.0, gross_profit - dividend_20pct)
        earn_migration = round(
            remaining_profit * profile.fixed_income_earn_migration_pct, 2
        )
        self.wallet_balance_usdt = round(self.wallet_balance_usdt - earn_migration, 4)
        self.total_migrated_to_earn_usdt = round(
            self.total_migrated_to_earn_usdt + earn_migration, 4
        )

        reinvested = round(remaining_profit - earn_migration, 2)
        self.monthly_net_profit_usdt = 0.0
        self.phase_initial_capital_usdt = self.wallet_balance_usdt

        return MonthlyTreasurySettlementPOD(
            phase=profile.phase.value,
            phase_initial_capital_usdt=round(base_cap, 2),
            gross_monthly_profit_usdt=round(gross_profit, 2),
            achieved_30d_roi_pct=achieved_roi_30d,
            doubling_goal_met=goal_ok,
            dividend_withdrawn_20pct_usdt=dividend_20pct,
            migrated_to_earn_cetes_usdt=earn_migration,
            reinvested_compounding_usdt=reinvested,
            new_trading_wallet_balance_usdt=round(self.wallet_balance_usdt, 2),
            summary_message=(
                f"[{profile.phase.value}] Corte 30 Días Exitoso: Dividendo 20%=${dividend_20pct:.2f} USDT | "
                f"Renta Fija/Earn=${earn_migration:.2f} USDT | Reinvertido=${reinvested:.2f} USDT."
            ),
        )

    def record_network_rtt_ms(self, rtt_ms: float) -> None:
        if rtt_ms >= 0.0:
            self.rtt_latency_ms_window.append(rtt_ms)

    def get_p99_latency_ms(self) -> float:
        if not self.rtt_latency_ms_window:
            return 0.0
        return round(float(np.percentile(self.rtt_latency_ms_window, 99)), 1)


# ============================================================================
# 4. SUITE DE VERIFICACIÓN INDEPENDIENTE DEL MÓDULO 4 (5 FASES + 3X EN F1/F2, 2X EN F3)
# ============================================================================

def _self_test_m4_module() -> None:
    from core_mark2_contracts import MarketSafetyTier

    orchestrator = Mark2GridDecisionOrchestrator()
    treasury = Mark2RiskAndTreasuryController(initial_wallet_balance_usdt=18.00)

    # 1. Verificar las 5 Fases por capital disponible en billetera y su meta de 30 días
    p1 = treasury.resolve_phase_behavior_profile(18.0)     # Fase 1: 4 a 100 USDT -> 3x (+200%)
    p2 = treasury.resolve_phase_behavior_profile(250.0)    # Fase 2: 100 a 1,000 USDT -> 3x (+200%)
    p3 = treasury.resolve_phase_behavior_profile(2_500.0)  # Fase 3: 1,000 a 10,000 USDT -> 2x (+100%)
    p4 = treasury.resolve_phase_behavior_profile(25_000.0) # Fase 4: 10,000 a 100,000 USDT -> 1.35x (+35%)
    p5 = treasury.resolve_phase_behavior_profile(250_000.0)# Fase 5: 100,000 a 1,000,000 USDT -> 1.15x (+15%)

    assert p1.phase == ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100
    assert p2.phase == ProsperityCapitalPhase.PHASE_2_EXPANSION_100_TO_1K
    assert p3.phase == ProsperityCapitalPhase.PHASE_3_ACCELERATION_1K_TO_10K
    assert p4.phase == ProsperityCapitalPhase.PHASE_4_INSTITUTIONAL_10K_TO_100K
    assert p5.phase == ProsperityCapitalPhase.PHASE_5_SOVEREIGN_100K_TO_1M

    # Verificar que en Fase 1 y Fase 2 la meta en 30 días es TRIPLICAR (3x / +200% ROI)
    for p in (p1, p2):
        assert p.phase_target_multiplier == 3.0
        assert p.target_roi_per_bot_pct == 200.0
        assert p.horizon_days == 30

    # Verificar que en Fase 3 la meta en 30 días es DUPLICAR (2x / +100% ROI)
    assert p3.phase_target_multiplier == 2.0
    assert p3.target_roi_per_bot_pct == 100.0
    assert p3.horizon_days == 30

    # Verificar que el apalancamiento y el riesgo disminuyen progresivamente de Fase 1 a Fase 5
    assert p1.max_leverage > p2.max_leverage > p3.max_leverage > p4.max_leverage > p5.max_leverage
    assert (
        p1.max_account_risk_per_bot_pct
        > p2.max_account_risk_per_bot_pct
        > p3.max_account_risk_per_bot_pct
        > p4.max_account_risk_per_bot_pct
        > p5.max_account_risk_per_bot_pct
    )

    verdict_f1 = treasury.authorize_new_grid_bot_deployment(
        "SUIUSDT", active_bots_count=0, candidate_uncertainty_score=0.18
    )
    assert verdict_f1.approved is True
    assert verdict_f1.assigned_capital_usdt >= 4.00
    assert verdict_f1.active_phase == ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100

    rules_sui = ExchangeContractRulePOD("SUIUSDT", 0.0001, 0.1, 0.1, 5.0, 200, max_leverage=50)
    tel_sui = Market3VTelemetryPOD(
        symbol="SUIUSDT", timestamp_ms=1730000000000,
        last_price=3.2000, mark_price=3.2000, bid_price=3.1995, ask_price=3.2005,
        bid_qty=5000.0, ask_qty=4200.0, quote_vol_24h=520_000_000.0,
        atr_14=0.0280, natr_14_pct=0.875, bb_upper=3.3200, bb_mid=3.2000, bb_lower=3.0800,
        hurst_exp=0.46, ema50_1h_slope_pct=0.25, quote_obi_l1=0.20, cvd_norm_3=0.18,
        funding_rate=0.0001, mae_q65_pct=2.8, mfe_q85_pct=5.2, mfe_q92_pct=7.5,
        intraday_path_pct=22.0, ewma_vol_pct=0.82, variance_ratio_q4=0.84,
        flow_toxicity_z=0.35, bayesian_grid_survival_prob=0.91,
    )

    plan_s1 = orchestrator.build_geometric_grid_6_to_18(
        rules=rules_sui, tel=tel_sui, safety_tier=MarketSafetyTier.TIER_3_HIGH_ROTATION_MID,
        initial_capital_usdt=6.00, current_equity_usdt=6.00, target_equity_usdt=None,
        wallet_balance_usdt=18.00,
    )
    assert plan_s1.compounding_stage == CompoundingStage.STAGE_1_MITIGATION_0_TO_33PCT
    assert plan_s1.estimated_liquidation_price < plan_s1.sl_price
    assert plan_s1.phase_target_multiplier == 3.0
    assert plan_s1.tp_roi_pct >= 200.0  # En Fase 1 al menos triplica (3x) y se calibra por el mercado
    assert plan_s1.expected_daily_pnl_usdt > 0.0

    target_pnl = plan_s1.tp_pnl_usdt
    act_s2, plan_s2 = treasury.evaluate_bot_equity_transition(
        plan_s1, realized_pnl_usdt=round(target_pnl * 0.28, 2), unrealized_pnl_usdt=0.0,
        rules=rules_sui, tel=tel_sui, orchestrator=orchestrator,
    )
    assert act_s2 == "STAGE_UPGRADED_RESIZE"
    assert plan_s2.compounding_stage == CompoundingStage.STAGE_2_EXPANSION_33_TO_66PCT

    act_s3, plan_s3 = treasury.evaluate_bot_equity_transition(
        plan_s2, realized_pnl_usdt=round(target_pnl * 0.55, 2), unrealized_pnl_usdt=0.0,
        rules=rules_sui, tel=tel_sui, orchestrator=orchestrator,
    )
    assert act_s3 == "STAGE_UPGRADED_RESIZE"
    assert plan_s3.compounding_stage == CompoundingStage.STAGE_3_HARVEST_66_TO_100PCT

    act_tp, plan_tp = treasury.evaluate_bot_equity_transition(
        plan_s3, realized_pnl_usdt=round(target_pnl + 0.10, 2), unrealized_pnl_usdt=0.0,
        rules=rules_sui, tel=tel_sui, orchestrator=orchestrator,
    )
    assert act_tp == BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value
    assert plan_tp.compounding_stage == CompoundingStage.TARGET_REACHED_EXPECTED_ROI

    treasury.deposit_external_capital(11_982.00)
    treasury.register_bot_session_closed("SUIUSDT", net_realized_pnl_usdt=2_000.00)
    settlement = treasury.process_monthly_treasury_settlement()
    assert settlement.dividend_withdrawn_20pct_usdt == 400.00

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(
        f"✅ [M4 5 FASES + 3X EN FASES 1-2 y 2X EN FASE 3 OK] "
        f"Fase 1 ($4-$100: {p1.max_leverage}x, Riesgo {p1.max_account_risk_per_bot_pct}%, Meta {p1.phase_target_multiplier:.0f}x) | "
        f"Fase 2 ($100-$1k: {p2.max_leverage}x, Riesgo {p2.max_account_risk_per_bot_pct}%, Meta {p2.phase_target_multiplier:.0f}x) | "
        f"Fase 3 ($1k-$10k: {p3.max_leverage}x, Riesgo {p3.max_account_risk_per_bot_pct}%, Meta {p3.phase_target_multiplier:.0f}x) | "
        f"SUI ROI Esperado={plan_s1.tp_roi_pct:.1f}% (+${plan_s1.tp_pnl_usdt:.2f} USDT) | Certidumbre={plan_s1.confidence_pct:.1f}%"
    )


if __name__ == "__main__":
    _self_test_m4_module()
