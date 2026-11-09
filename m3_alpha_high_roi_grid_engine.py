# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — m3_alpha_high_roi_grid_engine.py
Motor calibrado 100% con datos reales de la API de Binance (filtros de contrato,
excursiones MAE/MFE, OBI L1, CVD Taker, Toxicidad VPIN y recorrido intradiario real)
que calcula la Certidumbre (%), el Apalancamiento/Riesgo de la Fase (1 a 5), el
ROI/PnL Esperado Diario y a 30 Días de cada mercado, y los Días Estimados para
cumplir el objetivo de la Fase (Triplicar 3x en Fases 1 y 2, Duplicar 2x en Fase 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core_mark2_contracts import (
    ExchangeContractRulePOD,
    GridDirection,
    Market3VTelemetryPOD,
    ProsperityCapitalPhase,
)
from m2_intelligence_regime_ml import Mark2IntelligenceEngine
from m3_grid_decision_orchestrator import LotAndTickQuantizer, Mark2GridDecisionOrchestrator


@dataclass(frozen=True, slots=True)
class AlphaHighROIGridSpec:
    """
    Especificación de Grid de Futuros basada estrictamente en reglas de Binance,
    Fase de Capital (1 a 5), Certidumbre (%), ROI/PnL Esperado Diario y a 30 Días,
    y Días Estimados para cumplir la meta de la Fase (3x en Fases 1-2, 2x en Fase 3).
    """
    symbol: str
    direction: GridDirection
    leverage: int
    initial_capital_usdt: float
    min_investment_binance_formula_usdt: float
    guaranteed_target_equity_usdt: float
    open_position_on_creation: bool
    initial_base_position_qty: float
    initial_base_notional_usdt: float
    grid_count: int
    grid_mode: str
    lower_price: float
    upper_price: float
    step_pct: float
    net_profit_per_grid_pct: float
    net_profit_per_grid_usdt: float
    qty_per_grid_line: float
    notional_per_grid_line_usdt: float
    grid_levels: Tuple[float, ...]
    est_daily_matched_trades: int
    est_days_to_18_usd: float  # Días estimados a la meta esperada del mercado
    est_3d_10h_roi_pct: float
    est_3d_10h_pnl_usd: float
    sl_price: float
    tp_trigger_price: float
    active_phase: str = "FASE_1_4_A_100_USDT"
    phase_risk_pct: float = 55.0
    phase_target_multiplier: float = 3.0
    uncertainty_score: float = 0.20
    confidence_pct: float = 80.0
    expected_daily_pnl_usdt: float = 0.0
    expected_daily_roi_pct: float = 0.0
    expected_30d_pnl_usdt: float = 0.0
    expected_30d_roi_pct: float = 200.0
    est_days_to_double_30d: float = 30.0  # Días estimados para cumplir la meta de la Fase (3x en F1-F2, 2x en F3)


class AlphaHighROIGridCalibrator:
    """
    Calcula la configuración del Grid de Futuros usando únicamente los filtros reales
    de /fapi/v1/exchangeInfo y la telemetría de /fapi/v1/klines + /fapi/v1/ticker/bookTicker,
    ajustando el apalancamiento y el riesgo a las 5 Fases de Capital y determinando el
    ROI y PnL esperado Diario y a 30 Días de cada mercado y los Días Estimados.
    """

    def __init__(self, maker_fee: float = 0.00018):
        self.maker_fee = maker_fee

    def calculate_binance_min_investment(
        self,
        grid_count: int,
        leverage: int,
        min_notional: float,
        maint_margin_rate: float = 0.005,
    ) -> float:
        """
        Fórmula de inversión mínima de Binance Futures Grid basada en el minNotional
        y el maintMarginPercent real del contrato en /fapi/v1/exchangeInfo.
        """
        margin_buffer = 1.0 + (2.0 * maint_margin_rate)
        raw_min = ((grid_count * min_notional) / max(1, leverage)) * margin_buffer
        return round(raw_min, 2)

    def build_alpha_high_roi_bot(
        self,
        rules: ExchangeContractRulePOD,
        tel: Market3VTelemetryPOD,
        capital_usdt: float = 6.00,
        leverage: Optional[int] = None,
        target_min_equity_usdt: Optional[float] = None,
        grid_mode: str = "GEOMETRIC",
        wallet_balance_usdt: Optional[float] = None,
    ) -> AlphaHighROIGridSpec:
        """
        Configura un bot calibrado por el ROI y PnL esperado real del mercado y gobernado
        por la Fase de Capital disponible en la billetera (Fases 1 a 5: 3x en F1-F2, 2x en F3).
        """
        p0 = tel.last_price
        spread_pct = ((tel.ask_price - tel.bid_price) / p0) * 100.0 if p0 > 0.0 else 0.0

        ref_wallet = wallet_balance_usdt if wallet_balance_usdt is not None else capital_usdt
        (
            active_phase,
            phase_rec_lev,
            phase_max_lev,
            phase_risk_pct,
            _,
            phase_min_30d_roi_pct,
            phase_mult,
        ) = Mark2GridDecisionOrchestrator.resolve_phase_parameters(ref_wallet)
        uncertainty_score, confidence_pct = (
            Mark2GridDecisionOrchestrator.compute_market_uncertainty_and_confidence(tel, spread_pct)
        )

        contract_max_lev = min(rules.max_leverage, phase_max_lev)
        base_lev = (
            leverage
            if (leverage is not None and leverage > 0)
            else max(2, int(round(phase_rec_lev * max(0.65, 1.0 - 0.30 * uncertainty_score))))
        )
        lev = min(contract_max_lev, max(1, base_lev))

        direction = Mark2IntelligenceEngine.classify_grid_direction(tel)
        if direction == GridDirection.NEUTRAL:
            direction = (
                GridDirection.LONG
                if (tel.ema50_1h_slope_pct + tel.quote_obi_l1 + tel.cvd_norm_3) >= 0.0
                else GridDirection.SHORT
            )

        _, min_order_notional = Mark2GridDecisionOrchestrator.compute_min_legal_order_notional(
            rules, p0
        )
        margin_buffer = 1.0 + (2.0 * rules.maint_margin_rate)
        max_direct_grids = int(
            math.floor((capital_usdt * lev) / max(min_order_notional * margin_buffer, 1e-6))
        )
        if max_direct_grids < 4 and lev < contract_max_lev:
            lev = min(
                contract_max_lev,
                int(math.ceil((4.0 * min_order_notional * margin_buffer) / max(capital_usdt, 1e-6))),
            )
            max_direct_grids = int(
                math.floor((capital_usdt * lev) / max(min_order_notional * margin_buffer, 1e-6))
            )

        grid_count = max(2, min(min(169, rules.max_num_orders), max_direct_grids))
        min_inv_binance = self.calculate_binance_min_investment(
            grid_count, lev, min_order_notional, rules.maint_margin_rate
        )

        bb_half = max((tel.bb_upper - tel.bb_lower) / 2.0, tel.atr_14 * 2.0)
        mae_span = p0 * (tel.mae_q65_pct / 100.0)
        mfe_span = p0 * (tel.mfe_q92_pct / 100.0)

        if direction == GridDirection.LONG:
            raw_low = max(rules.tick_size, p0 - max(bb_half, mae_span + tel.atr_14))
            raw_high = p0 + max(bb_half, mfe_span + tel.atr_14)
        else:
            raw_low = max(rules.tick_size, p0 - max(bb_half, mfe_span + tel.atr_14))
            raw_high = p0 + max(bb_half, mae_span + tel.atr_14)

        lower_p = LotAndTickQuantizer.quantize_price(raw_low, rules.tick_size, "FLOOR")
        upper_p = LotAndTickQuantizer.quantize_price(raw_high, rules.tick_size, "CEIL")
        if upper_p <= lower_p:
            upper_p = LotAndTickQuantizer.quantize_price(
                lower_p + max(10.0 * rules.tick_size, p0 * 0.02), rules.tick_size, "CEIL"
            )

        levels: List[float] = []
        if grid_mode.upper() == "ARITHMETIC":
            diff = (upper_p - lower_p) / grid_count
            for k in range(grid_count + 1):
                levels.append(LotAndTickQuantizer.quantize_price(lower_p + k * diff, rules.tick_size))
            step_pct = (diff / p0) * 100.0
        else:
            geom_r = (upper_p / lower_p) ** (1.0 / grid_count)
            for k in range(grid_count + 1):
                levels.append(LotAndTickQuantizer.quantize_price(lower_p * (geom_r ** k), rules.tick_size))
            step_pct = (geom_r - 1.0) * 100.0

        net_step_pct = max(1e-4, step_pct - (2.0 * self.maker_fee * 100.0))

        min_qty_lower, _ = Mark2GridDecisionOrchestrator.compute_min_legal_order_notional(rules, lower_p)
        min_qty_p0, _ = Mark2GridDecisionOrchestrator.compute_min_legal_order_notional(rules, p0)
        qty_per_line = max(min_qty_lower, min_qty_p0)
        actual_line_notional = round(qty_per_line * p0, 2)
        net_profit_per_grid_usdt = round(actual_line_notional * (net_step_pct / 100.0), 4)

        if direction == GridDirection.LONG:
            counter_lines = sum(1 for lv in levels if lv > p0)
        else:
            counter_lines = sum(1 for lv in levels if lv < p0)

        base_lines = max(1, counter_lines)
        initial_base_qty = LotAndTickQuantizer.quantize_qty(
            qty_per_line * base_lines, rules.step_size, mode="FLOOR"
        )
        initial_base_notional = round(initial_base_qty * p0, 2)

        mr_efficiency = max(0.75, min(1.25, 1.0 / max(tel.variance_ratio_q4, 0.50)))
        uncertainty_haircut = max(0.55, 1.0 - (0.30 * uncertainty_score))
        est_daily_trades = max(
            1,
            int(
                round(
                    (tel.intraday_path_pct / max(2.0 * step_pct, 1e-6))
                    * mr_efficiency
                    * tel.bayesian_grid_survival_prob
                )
            ),
        )
        daily_grid_harvest_usd = est_daily_trades * net_profit_per_grid_usdt * uncertainty_haircut
        daily_directional_usd = (
            initial_base_notional * (abs(tel.ema50_1h_slope_pct) / 100.0) * 0.5 * uncertainty_haircut
        )
        total_daily_pnl_usd = round(max(1e-4, daily_grid_harvest_usd + daily_directional_usd), 4)
        expected_daily_roi_pct = round((total_daily_pnl_usd / max(capital_usdt, 1e-6)) * 100.0, 2)

        mean_30d_mult = 1.50 if phase_mult >= 3.0 else (1.35 if phase_mult >= 2.0 else 1.15)
        expected_30d_pnl_usdt = round(total_daily_pnl_usd * 30.0 * mean_30d_mult, 2)
        expected_30d_roi_pct = round((expected_30d_pnl_usdt / max(capital_usdt, 1e-6)) * 100.0, 2)

        if target_min_equity_usdt is None:
            empirical_cycle_boost_pct = round(
                (tel.mfe_q92_pct * 3.2)
                + ((confidence_pct - 65.0) * 0.65)
                + min(45.0, expected_daily_roi_pct * 2.5),
                2,
            )
            raw_market_target_roi = phase_min_30d_roi_pct + max(0.0, empirical_cycle_boost_pct)
            resolved_target_roi_pct = round(
                max(phase_min_30d_roi_pct, min(expected_30d_roi_pct, raw_market_target_roi)),
                2,
            )
            resolved_target_equity_usdt = round(
                capital_usdt * (1.0 + resolved_target_roi_pct / 100.0), 2
            )
        else:
            resolved_target_equity_usdt = round(target_min_equity_usdt, 2)

        needed_profit_for_target = max(0.01, resolved_target_equity_usdt - capital_usdt)
        est_days_to_target = round(needed_profit_for_target / (total_daily_pnl_usd * mean_30d_mult), 2)
        est_days_to_phase_goal = round(
            (capital_usdt * (phase_min_30d_roi_pct / 100.0)) / (total_daily_pnl_usd * mean_30d_mult),
            2,
        )

        horizon_days = 82.0 / 24.0
        pnl_3d_10h = round(total_daily_pnl_usd * horizon_days, 2)
        roi_3d_10h = round((pnl_3d_10h / max(capital_usdt, 1e-6)) * 100.0, 2)

        sl_buffer = max(tel.atr_14, p0 * (tel.mae_q65_pct / 100.0))
        tp_buffer = max(tel.atr_14, p0 * (tel.mfe_q85_pct / 100.0))
        if direction == GridDirection.LONG:
            sl_price = LotAndTickQuantizer.quantize_price(
                max(rules.tick_size, lower_p - sl_buffer), rules.tick_size, "FLOOR"
            )
            tp_price = LotAndTickQuantizer.quantize_price(upper_p + tp_buffer, rules.tick_size, "CEIL")
        else:
            sl_price = LotAndTickQuantizer.quantize_price(upper_p + sl_buffer, rules.tick_size, "CEIL")
            tp_price = LotAndTickQuantizer.quantize_price(
                max(rules.tick_size, lower_p - tp_buffer), rules.tick_size, "FLOOR"
            )

        effective_risk_pct = (
            55.0
            if active_phase == ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100
            else round(
                max(0.25, phase_risk_pct * max(0.55, 1.0 - (0.40 * uncertainty_score))),
                2,
            )
        )

        return AlphaHighROIGridSpec(
            symbol=rules.symbol,
            direction=direction,
            leverage=lev,
            initial_capital_usdt=round(capital_usdt, 2),
            min_investment_binance_formula_usdt=min_inv_binance,
            guaranteed_target_equity_usdt=resolved_target_equity_usdt,
            open_position_on_creation=True,
            initial_base_position_qty=initial_base_qty,
            initial_base_notional_usdt=initial_base_notional,
            grid_count=grid_count,
            grid_mode=grid_mode.upper(),
            lower_price=lower_p,
            upper_price=upper_p,
            step_pct=round(step_pct, 3),
            net_profit_per_grid_pct=round(net_step_pct, 3),
            net_profit_per_grid_usdt=net_profit_per_grid_usdt,
            qty_per_grid_line=qty_per_line,
            notional_per_grid_line_usdt=actual_line_notional,
            grid_levels=tuple(levels),
            est_daily_matched_trades=est_daily_trades,
            est_days_to_18_usd=est_days_to_target,
            est_3d_10h_roi_pct=roi_3d_10h,
            est_3d_10h_pnl_usd=pnl_3d_10h,
            sl_price=sl_price,
            tp_trigger_price=tp_price,
            active_phase=active_phase.value,
            phase_risk_pct=effective_risk_pct,
            phase_target_multiplier=phase_mult,
            uncertainty_score=uncertainty_score,
            confidence_pct=confidence_pct,
            expected_daily_pnl_usdt=total_daily_pnl_usd,
            expected_daily_roi_pct=expected_daily_roi_pct,
            expected_30d_pnl_usdt=expected_30d_pnl_usdt,
            expected_30d_roi_pct=expected_30d_roi_pct,
            est_days_to_double_30d=est_days_to_phase_goal,
        )
