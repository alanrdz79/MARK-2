# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python m4_phase_behavior_governor.py
Gobernador de Mutación Automática de Comportamiento por Fases de Capital Disponible en Billetera:
- Fase 1 (4 a 100 USDT): Fase actual. Objetivo: Mitigar/disminuir la incertidumbre mediante los datos
  reales de Binance y TRIPLICAR (3x) el capital inicial de la cuenta (+200% ROI mínimo) en un plazo de 30 días.
- Fase 2 (100 a 1,000 USDT): Expansión multiactivo. Objetivo: TRIPLICAR (3x) el capital inicial de la cuenta
  en ese momento (+200% ROI mínimo) en un plazo de 30 días con menor riesgo y menor apalancamiento.
- Fase 3 (1,000 a 10,000 USDT): Aceleración estructural. Objetivo: DUPLICAR (2x) el capital inicial de la cuenta
  en ese momento (+100% ROI mínimo) en un plazo de 30 días con apalancamiento moderado y liquidez profunda.
- Fase 4 (10,000 a 100,000 USDT): Cosecha Institucional + Retiro mensual del 20% + Renta Fija 15%.
- Fase 5 (100,000 a 1,000,000 USDT): Preservación Patrimonial Soberana + Retiro 20% + Renta Fija 30%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from core_mark2_contracts import ProsperityCapitalPhase


@dataclass(frozen=True, slots=True)
class PhaseBehaviorProfile:
    """
    Define el riesgo por operación, el apalancamiento nominal y efectivo, el filtro de incertidumbre
    y la meta de rendimiento a 30 días (3x en Fases 1 y 2, 2x en Fase 3) según la Fase de Capital.
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
    target_roi_per_bot_pct: float  # +200% (3x) en Fases 1 y 2 | +100% (2x) en Fase 3 en 30 días
    max_account_risk_per_bot_pct: float
    allowed_market_tiers: Tuple[str, ...]
    mandatory_core_symbols: Tuple[str, ...]
    allow_alpha_satellite_bots: bool
    monthly_profit_withdrawal_pct: float
    fixed_income_migration_pct: float
    require_correlation_filter: bool
    max_effective_inventory_leverage: float = 3.5
    horizon_days: int = 30
    phase_target_multiplier: float = 3.0  # 3.0x en Fases 1-2, 2.0x en Fase 3
    must_double_in_30d: bool = True
    min_30d_target_equity_usdt: float = 12.0
    min_30d_target_pnl_usdt: float = 8.0
    required_daily_compound_roi_pct: float = 3.73
    min_confidence_pct: float = 68.0


class Mark2PhaseBehaviorGovernor:
    """
    Conmutador determinista O(1) que adapta el riesgo en operaciones, el número de apalancamiento
    y la exigencia de certidumbre estadística conforme el capital disponible en billetera escala
    por las 5 Fases ($4 USDT hasta $1,000,000 USDT).
    """

    DAILY_COMPOUND_FOR_30D_TRIPLING_PCT = round(((3.0 ** (1.0 / 30.0)) - 1.0) * 100.0, 4)  # ~3.7299%/día (3x en 30d)
    DAILY_COMPOUND_FOR_30D_DOUBLING_PCT = round(((2.0 ** (1.0 / 30.0)) - 1.0) * 100.0, 4)  # ~2.3374%/día (2x en 30d)

    def __init__(self, initial_balance_usdt: float = 18.0):
        self.wallet_balance_usdt: float = max(4.0, round(initial_balance_usdt, 4))
        self.phase_initial_anchor_usdt: float = self.wallet_balance_usdt
        self.monthly_realized_profit_usdt: float = 0.0
        self.total_withdrawn_dividends_usdt: float = 0.0
        self.total_external_deposits_usdt: float = 0.0

    def record_external_deposit(self, amount_usdt: float) -> None:
        if amount_usdt <= 0.0:
            return
        self.wallet_balance_usdt = round(self.wallet_balance_usdt + amount_usdt, 2)
        self.phase_initial_anchor_usdt = round(self.phase_initial_anchor_usdt + amount_usdt, 2)
        self.total_external_deposits_usdt = round(
            self.total_external_deposits_usdt + amount_usdt, 2
        )

    def resolve_active_behavior_profile(
        self, current_balance_usdt: float | None = None
    ) -> PhaseBehaviorProfile:
        bal = max(
            4.0,
            current_balance_usdt if current_balance_usdt is not None else self.wallet_balance_usdt,
        )

        # =====================================================================
        # FASE 1: 4 a 100 USDT (Mitigación de Incertidumbre + TRIPLICAR 3x en 30d)
        # =====================================================================
        if bal < 100.0:
            if bal < 12.0:
                max_bots = 1
                bot_alloc = round(bal, 2)
            elif bal < 30.0:
                max_bots = min(3, int(bal // 6.0))
                bot_alloc = round(bal / max(1, max_bots), 2)
            else:
                max_bots = min(4, int(bal // 10.0))
                bot_alloc = round(bal * 0.25, 2)

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
                capital_allocation_per_bot_usdt=max(4.0, bot_alloc),
                target_roi_per_bot_pct=200.0,  # Triplicar capital inicial en 30 días (3x = +200% ROI mínimo)
                max_account_risk_per_bot_pct=55.0,
                allowed_market_tiers=("ESTABLES", "INTERMEDIOS", "VOLATILES"),
                mandatory_core_symbols=("XRPUSDT", "SUIUSDT", "DOGEUSDT", "SOLUSDT", "ADAUSDT"),
                allow_alpha_satellite_bots=True,
                monthly_profit_withdrawal_pct=0.0,
                fixed_income_migration_pct=0.0,
                require_correlation_filter=True,
                max_effective_inventory_leverage=3.5,
                horizon_days=30,
                phase_target_multiplier=3.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 3.0, 2),
                min_30d_target_pnl_usdt=round(bal * 2.0, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_TRIPLING_PCT,
                min_confidence_pct=68.0,
            )

        # =====================================================================
        # FASE 2: 100 a 1,000 USDT (Expansión + TRIPLICAR 3x en 30d)
        # =====================================================================
        if bal < 1_000.0:
            bot_alloc = round(bal * 0.25, 2)
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
                capital_allocation_per_bot_usdt=bot_alloc,
                target_roi_per_bot_pct=200.0,  # Triplicar capital inicial en 30 días (3x = +200% ROI mínimo)
                max_account_risk_per_bot_pct=3.5,
                allowed_market_tiers=("ESTABLES", "INTERMEDIOS", "VOLATILES"),
                mandatory_core_symbols=("SOLUSDT", "BNBUSDT", "XRPUSDT", "SUIUSDT", "ETHUSDT"),
                allow_alpha_satellite_bots=True,
                monthly_profit_withdrawal_pct=0.0,
                fixed_income_migration_pct=0.0,
                require_correlation_filter=True,
                max_effective_inventory_leverage=2.8,
                horizon_days=30,
                phase_target_multiplier=3.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 3.0, 2),
                min_30d_target_pnl_usdt=round(bal * 2.0, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_TRIPLING_PCT,
                min_confidence_pct=70.0,
            )

        # =====================================================================
        # FASE 3: 1,000 a 10,000 USDT (Aceleración + DUPLICAR 2x en 30d)
        # =====================================================================
        if bal < 10_000.0:
            bot_alloc = round(bal * 0.20, 2)
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
                capital_allocation_per_bot_usdt=bot_alloc,
                target_roi_per_bot_pct=100.0,  # Duplicar capital inicial en 30 días (2x = +100% ROI mínimo)
                max_account_risk_per_bot_pct=2.0,
                allowed_market_tiers=("ESTABLES", "INTERMEDIOS"),
                mandatory_core_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
                allow_alpha_satellite_bots=True,
                monthly_profit_withdrawal_pct=0.0,
                fixed_income_migration_pct=0.0,
                require_correlation_filter=True,
                max_effective_inventory_leverage=2.0,
                horizon_days=30,
                phase_target_multiplier=2.0,
                must_double_in_30d=True,
                min_30d_target_equity_usdt=round(bal * 2.0, 2),
                min_30d_target_pnl_usdt=round(bal, 2),
                required_daily_compound_roi_pct=self.DAILY_COMPOUND_FOR_30D_DOUBLING_PCT,
                min_confidence_pct=72.0,
            )

        # =====================================================================
        # FASE 4: 10,000 a 100,000 USDT (Cosecha Institucional)
        # =====================================================================
        if bal < 100_000.0:
            bot_alloc = round(bal * 0.20, 2)
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
                capital_allocation_per_bot_usdt=bot_alloc,
                target_roi_per_bot_pct=35.0,
                max_account_risk_per_bot_pct=1.0,
                allowed_market_tiers=("ESTABLES", "INTERMEDIOS"),
                mandatory_core_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
                allow_alpha_satellite_bots=False,
                monthly_profit_withdrawal_pct=0.20,
                fixed_income_migration_pct=0.15,
                require_correlation_filter=True,
                max_effective_inventory_leverage=1.5,
                horizon_days=30,
                phase_target_multiplier=1.35,
                must_double_in_30d=False,
                min_30d_target_equity_usdt=round(bal * 1.35, 2),
                min_30d_target_pnl_usdt=round(bal * 0.35, 2),
                required_daily_compound_roi_pct=round(((1.35 ** (1.0 / 30.0)) - 1.0) * 100.0, 4),
                min_confidence_pct=75.0,
            )

        # =====================================================================
        # FASE 5: 100,000 a 1,000,000 USDT (Preservación Patrimonial Soberana)
        # =====================================================================
        bot_alloc = round(bal * 0.15, 2)
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
            capital_allocation_per_bot_usdt=bot_alloc,
            target_roi_per_bot_pct=15.0,
            max_account_risk_per_bot_pct=0.5,
            allowed_market_tiers=("ESTABLES",),
            mandatory_core_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            allow_alpha_satellite_bots=False,
            monthly_profit_withdrawal_pct=0.20,
            fixed_income_migration_pct=0.30,
            require_correlation_filter=True,
            max_effective_inventory_leverage=1.0,
            horizon_days=30,
            phase_target_multiplier=1.15,
            must_double_in_30d=False,
            min_30d_target_equity_usdt=round(bal * 1.15, 2),
            min_30d_target_pnl_usdt=round(bal * 0.15, 2),
            required_daily_compound_roi_pct=round(((1.15 ** (1.0 / 30.0)) - 1.0) * 100.0, 4),
            min_confidence_pct=80.0,
        )

    def execute_monthly_closing(self) -> dict:
        profile = self.resolve_active_behavior_profile()
        if (
            profile.monthly_profit_withdrawal_pct <= 0.0
            or self.monthly_realized_profit_usdt <= 0.0
        ):
            return {
                "phase": profile.phase.value,
                "withdrawn_dividend_usdt": 0.0,
                "reinvested_profit_usdt": round(max(0.0, self.monthly_realized_profit_usdt), 2),
                "new_wallet_balance_usdt": round(self.wallet_balance_usdt, 2),
            }

        dividend_usdt = round(
            self.monthly_realized_profit_usdt * profile.monthly_profit_withdrawal_pct, 2
        )
        self.wallet_balance_usdt = round(self.wallet_balance_usdt - dividend_usdt, 2)
        self.total_withdrawn_dividends_usdt = round(
            self.total_withdrawn_dividends_usdt + dividend_usdt, 2
        )
        reinvested = round(self.monthly_realized_profit_usdt - dividend_usdt, 2)
        self.monthly_realized_profit_usdt = 0.0

        return {
            "phase": profile.phase.value,
            "withdrawn_dividend_usdt": dividend_usdt,
            "reinvested_profit_usdt": reinvested,
            "new_wallet_balance_usdt": self.wallet_balance_usdt,
        }
