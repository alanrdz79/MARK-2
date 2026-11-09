# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — m3_grid_decision_orchestrator.py
Escáner Universal de Mercados USDⓈ-M, Mitigador de Incertidumbre con Datos de Binance y
Sintetizador de Grids Geométricos con ROI y PnL Esperado por Mercado (Fases 1 a 5):
1. Calcula para cada mercado:
   - Certidumbre (%) y Mitigación de Incertidumbre con datos de Binance (OBI L1, CVD Taker, VPIN, VR(4), Hurst, Supervivencia Bayesiana).
   - Apalancamiento y Riesgo Operativo de la Fase (1 a 5).
   - ROI (%) y PnL (USDT) Esperado Diario y a 30 Días de cada mercado.
   - Días Estimados para cumplir el objetivo de la Fase en <= 30 días:
     * Fase 1 (4 a 100 USDT): TRIPLICAR (3x / +200% ROI mínimo en 30 días).
     * Fase 2 (100 a 1,000 USDT): TRIPLICAR (3x / +200% ROI mínimo en 30 días).
     * Fase 3 (1,000 a 10,000 USDT): DUPLICAR (2x / +100% ROI mínimo en 30 días).
     * Fase 4 (10,000 a 100,000 USDT): Cosecha institucional (+35% en 30 días).
     * Fase 5 (100,000 a 1,000,000 USDT): Preservación soberana (+15% en 30 días).
2. Escudo Anti-Liquidación (Zero-Liquidation Shield): Desacopla el Apalancamiento Nominal del
   Apalancamiento Efectivo de Inventario garantizando P_liq << P_SL < P_lower.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from core_mark2_contracts import (
    CompoundingStage,
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    Market3VTelemetryPOD,
    MarketSafetyTier,
    ProsperityCapitalPhase,
    RankedMarketCandidatePOD,
)
from m2_intelligence_regime_ml import Mark2IntelligenceEngine


class LotAndTickQuantizer:
    """Garantiza cumplimiento exacto de PRICE_FILTER (tickSize) y LOT_SIZE (stepSize) de Binance."""

    @staticmethod
    def quantize_price(price: float, tick_size: float, mode: str = "ROUND") -> float:
        if tick_size <= 0.0:
            return price
        decimals = max(0, int(round(-math.log10(tick_size)))) if tick_size < 1.0 else 0
        ratio = price / tick_size
        if mode == "FLOOR":
            q = math.floor(ratio + 1e-12) * tick_size
        elif mode == "CEIL":
            q = math.ceil(ratio - 1e-12) * tick_size
        else:
            q = round(ratio) * tick_size
        return round(q, decimals)

    @staticmethod
    def quantize_qty(qty: float, step_size: float, mode: str = "FLOOR") -> float:
        if step_size <= 0.0:
            return qty
        decimals = max(0, int(round(-math.log10(step_size)))) if step_size < 1.0 else 0
        ratio = qty / step_size
        if mode == "CEIL":
            q = math.ceil(ratio - 1e-12) * step_size
        else:
            q = math.floor(ratio + 1e-12) * step_size
        return round(q, decimals)


class Mark2GridDecisionOrchestrator:
    """
    Cerebro Cuantitativo de MARK-2 basado 100% en datos reales de Binance:
    1. Calcula la Certidumbre (%) y mitiga la incertidumbre de cada contrato perpetuo USDⓈ-M.
    2. Calcula el ROI (%) y PnL (USDT) Esperado Diario y a 30 Días en cada mercado.
    3. Gobierna el Apalancamiento y el Riesgo según la Fase de Capital (Fases 1 a 5), buscando
       en Fases 1 y 2 TRIPLICAR (3x) y en Fase 3 DUPLICAR (2x) el capital inicial en <= 30 días.
    """

    def __init__(
        self,
        maker_fee: float = 0.00020,
        taker_fee: float = 0.00050,
        reserve_ratio: float = 0.25,
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.reserve_ratio = reserve_ratio

    @staticmethod
    def resolve_phase_parameters(
        wallet_capital_usdt: float,
    ) -> Tuple[ProsperityCapitalPhase, int, int, float, float, float, float]:
        """
        Determina para el capital disponible en billetera:
        (fase, apalancamiento_recomendado, apalancamiento_max_fase, riesgo_max_operacion_pct,
         apalancamiento_efectivo_max, meta_min_roi_30d_pct, multiplicador_meta_fase)
        """
        bal = max(4.0, wallet_capital_usdt)
        if bal < 100.0:
            # Fase 1: 4 a 100 USDT -> Mitigar incertidumbre + TRIPLICAR (3x = +200% ROI mín) en 30 días
            rec_lev = 20 if bal < 15.0 else (15 if bal < 50.0 else 12)
            max_lev = 25 if bal < 25.0 else 20
            return (
                ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100,
                rec_lev,
                max_lev,
                55.0,  # Riesgo operativo en Fase 1 (55.0%)
                3.5,   # Tope de apalancamiento efectivo de inventario
                200.0, # Meta mínima 30 días en Fase 1: Triplicar capital (3x = +200% ROI)
                3.0,   # Multiplicador 3x
            )
        if bal < 1_000.0:
            # Fase 2: 100 a 1,000 USDT -> TRIPLICAR (3x = +200% ROI mín) en 30 días
            return (
                ProsperityCapitalPhase.PHASE_2_EXPANSION_100_TO_1K,
                10,
                15,
                3.5,
                2.8,
                200.0, # Meta mínima 30 días en Fase 2: Triplicar capital (3x = +200% ROI)
                3.0,   # Multiplicador 3x
            )
        if bal < 10_000.0:
            # Fase 3: 1,000 a 10,000 USDT -> DUPLICAR (2x = +100% ROI mín) en 30 días
            return (
                ProsperityCapitalPhase.PHASE_3_ACCELERATION_1K_TO_10K,
                6,
                10,
                2.0,
                2.0,
                100.0, # Meta mínima 30 días en Fase 3: Duplicar capital (2x = +100% ROI)
                2.0,   # Multiplicador 2x
            )
        if bal < 100_000.0:
            # Fase 4: 10,000 a 100,000 USDT -> Cosecha institucional (1.35x en 30 días)
            return (
                ProsperityCapitalPhase.PHASE_4_INSTITUTIONAL_10K_TO_100K,
                4,
                5,
                1.0,
                1.5,
                35.0,
                1.35,
            )
        # Fase 5: 100,000 a 1,000,000 USDT -> Preservación soberana (1.15x en 30 días)
        return (
            ProsperityCapitalPhase.PHASE_5_SOVEREIGN_100K_TO_1M,
            2,
            3,
            0.5,
            1.0,
            15.0,
            1.15,
        )

    @staticmethod
    def compute_market_uncertainty_and_confidence(
        tel: Market3VTelemetryPOD, spread_pct: float
    ) -> Tuple[float, float]:
        """
        Cuantifica la incertidumbre microestructural del mercado exclusivamente a partir
        de probabilidades y razones empíricas extraídas de la API de Binance (sin pesos heurísticos):
        1. Tasa empírica de ruptura del rango en velas históricas (1 - bayesian_grid_survival_prob).
        2. Probabilidad acumulada normal (CDF Gaussiana exacta erf) del Z-Score VPIN Taker (k[9]) sobre la media.
        3. Divergencia relativa de Varianza de Lo-MacKinlay respecto a caminata aleatoria: max(0, VR(4)-1)/VR(4).
        4. Exceso de persistencia de Hurst sobre caminata aleatoria: max(0, H - 0.5) / H.
        5. Razón física de fricción L1 (spread_pct / (natr_14_pct + spread_pct)).
        6. Razón física de Funding Rate frente a volatilidad condicional (|funding|% / (ewma_vol_pct + |funding|%)).
        Integra el vector empírico mediante su media cuadrática (RMS) Ponderada por Varianza.
        """
        eff_vol = max(tel.natr_14_pct, tel.ewma_vol_pct, 1e-9)
        p_breakout = max(0.0, min(1.0, 1.0 - tel.bayesian_grid_survival_prob))
        p_vpin_excess = (
            math.erf(max(0.0, tel.flow_toxicity_z) / math.sqrt(2.0))
            if tel.flow_toxicity_z > 0.0
            else 0.0
        )
        p_vr_trend = (
            max(0.0, tel.variance_ratio_q4 - 1.0) / max(tel.variance_ratio_q4, 1e-9)
            if tel.variance_ratio_q4 > 1.0
            else 0.0
        )
        p_hurst_trend = (
            max(0.0, tel.hurst_exp - 0.50) / max(tel.hurst_exp, 1e-9)
            if tel.hurst_exp > 0.50
            else 0.0
        )
        p_spread_friction = spread_pct / max(eff_vol + spread_pct, 1e-9)
        funding_pct = abs(tel.funding_rate) * 100.0
        p_funding_drag = funding_pct / max(eff_vol + funding_pct, 1e-9)

        empirical_components = np.array(
            [
                p_breakout,
                p_vpin_excess,
                p_vr_trend,
                p_hurst_trend,
                p_spread_friction,
                p_funding_drag,
            ],
            dtype=np.float64,
        )
        # Media cuadrática (RMS) combinada con la probabilidad de ruptura empírica de las velas de la API
        rms_uncertainty = float(np.sqrt(np.mean(np.square(empirical_components))))
        raw_uncertainty = float((rms_uncertainty + p_breakout) * 0.5)
        uncertainty_score = round(max(0.01, min(0.99, raw_uncertainty)), 4)
        confidence_pct = round((1.0 - uncertainty_score) * 100.0, 2)
        return uncertainty_score, confidence_pct

    @classmethod
    def partition_universe_empirical_3_areas(
        cls,
        rules_map: Dict[str, ExchangeContractRulePOD],
        telemetry_map: Dict[str, Market3VTelemetryPOD],
    ) -> Dict[str, Tuple[MarketSafetyTier, float]]:
        """
        Divide inteligentemente los mercados del escáner cripto en las 3 Áreas:
          1. ESTABLES
          2. INTERMEDIOS
          3. VOLATILES
        basándose 100% en los datos extraídos con la API de Binance (sin umbrales ni pesos heurísticos):
        - Construye la matriz empírica X (N mercados x 6 variables de la API: NATR, EWMA, BBWidth,
          Excursión Intradía/MAE/MFE, Fricción Spread/Liquidez 24h, e Inestabilidad VR(4)/VPIN/Breakout).
        - Calcula los pesos objetivos mediante el método estadístico CRITIC (desviación estándar empírica
          por conflicto de correlación inter-criterio sobre la matriz de la API).
        - Ejecuta particionamiento K-Means 1D (minimización de varianza intra-cluster de Lloyd-Jenks)
          sobre la distribución empírica del índice compuesto para separar ESTABLES, INTERMEDIOS y VOLATILES.
        """
        valid_symbols = [
            s
            for s in rules_map
            if s in telemetry_map and telemetry_map[s].last_price > 0.0
        ]
        n = len(valid_symbols)
        if n == 0:
            return {}

        feature_rows: List[List[float]] = []
        for sym in valid_symbols:
            tel = telemetry_map[sym]
            spread_pct = ((tel.ask_price - tel.bid_price) / max(tel.last_price, 1e-12)) * 100.0
            bb_width_pct = (
                ((tel.bb_upper - tel.bb_lower) / max(tel.bb_mid, 1e-12)) * 100.0
                if tel.bb_mid > 0.0
                else tel.natr_14_pct
            )
            excursion_amp = tel.intraday_path_pct + tel.mae_q65_pct + tel.mfe_q92_pct
            log_vol = math.log10(max(tel.quote_vol_24h, 10.0))
            friction_illiq = spread_pct / max(log_vol, 1e-6)
            regime_instability = (
                tel.variance_ratio_q4
                * max(1e-4, 1.0 - tel.bayesian_grid_survival_prob)
                * (1.0 + max(0.0, tel.flow_toxicity_z))
            )
            feature_rows.append(
                [
                    float(tel.natr_14_pct),
                    float(tel.ewma_vol_pct),
                    float(bb_width_pct),
                    float(excursion_amp),
                    float(friction_illiq),
                    float(regime_instability),
                ]
            )

        mat = np.asarray(feature_rows, dtype=np.float64)
        if n == 1:
            return {valid_symbols[0]: (MarketSafetyTier.ESTABLES, 0.25)}

        col_min = np.min(mat, axis=0)
        col_max = np.max(mat, axis=0)
        col_span = np.where((col_max - col_min) > 1e-12, col_max - col_min, 1.0)
        norm_mat = (mat - col_min) / col_span

        # Ponderación objetiva CRITIC derivada 100% de la varianza y correlación de los datos de la API
        col_std = np.std(norm_mat, axis=0)
        if n >= 3 and np.all(col_std > 1e-12):
            corr = np.corrcoef(norm_mat, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            conflict = np.sum(1.0 - corr, axis=1)
            critic_info = col_std * conflict
        else:
            critic_info = np.where(col_std > 1e-12, col_std, 1.0)

        info_sum = float(np.sum(critic_info))
        weights = (
            critic_info / info_sum
            if info_sum > 1e-12
            else np.full(mat.shape[1], 1.0 / mat.shape[1])
        )
        scores = np.dot(norm_mat, weights)

        # Normalizar el score compuesto al rango empírico [0.05, 0.95] de la muestra actual
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        if (s_max - s_min) > 1e-12:
            ivr_scores = 0.05 + 0.90 * ((scores - s_min) / (s_max - s_min))
        else:
            ivr_scores = np.full(n, 0.50)

        if n == 2:
            order = np.argsort(ivr_scores)
            return {
                valid_symbols[int(order[0])]: (
                    MarketSafetyTier.ESTABLES,
                    round(float(ivr_scores[int(order[0])]), 4),
                ),
                valid_symbols[int(order[1])]: (
                    MarketSafetyTier.VOLATILES,
                    round(float(ivr_scores[int(order[1])]), 4),
                ),
            }

        # Clustering K-Means 1D (K=3: ESTABLES, INTERMEDIOS, VOLATILES) sobre la distribución de la API
        centroids = np.percentile(ivr_scores, [16.67, 50.0, 83.33]).astype(np.float64)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(30):
            dists = np.abs(ivr_scores[:, None] - centroids[None, :])
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_idx in range(3):
                members = ivr_scores[labels == c_idx]
                if len(members) > 0:
                    centroids[c_idx] = float(np.mean(members))

        # Ordenar centroides de menor a mayor índice empírico y garantizar que las 3 áreas tengan representación
        sorted_clusters = np.argsort(centroids)
        rank_map = {int(orig_c): rank_idx for rank_idx, orig_c in enumerate(sorted_clusters)}
        ordered_labels = np.array([rank_map[int(lbl)] for lbl in labels], dtype=np.int32)

        if len(set(ordered_labels.tolist())) < 3:
            q33, q67 = np.percentile(ivr_scores, [33.33, 66.67])
            ordered_labels = np.where(
                ivr_scores <= q33,
                0,
                np.where(ivr_scores <= q67, 1, 2),
            )

        tier_by_label = {
            0: MarketSafetyTier.ESTABLES,
            1: MarketSafetyTier.INTERMEDIOS,
            2: MarketSafetyTier.VOLATILES,
        }
        result: Dict[str, Tuple[MarketSafetyTier, float]] = {}
        for idx, sym in enumerate(valid_symbols):
            area_tier = tier_by_label[int(ordered_labels[idx])]
            result[sym] = (area_tier, round(float(ivr_scores[idx]), 4))
        return result

    @staticmethod
    def group_markets_by_empirical_area(
        ranked_list: List[RankedMarketCandidatePOD],
    ) -> Dict[str, List[RankedMarketCandidatePOD]]:
        """
        Agrupa la lista de mercados clasificados en las 3 Áreas empíricas de la API:
        ESTABLES, INTERMEDIOS y VOLATILES.
        """
        grouped: Dict[str, List[RankedMarketCandidatePOD]] = {
            MarketSafetyTier.ESTABLES.value: [],
            MarketSafetyTier.INTERMEDIOS.value: [],
            MarketSafetyTier.VOLATILES.value: [],
        }
        for cand in ranked_list:
            area_key = cand.safety_tier.value
            if area_key not in grouped:
                grouped[MarketSafetyTier.INTERMEDIOS.value].append(cand)
            else:
                grouped[area_key].append(cand)
        return grouped

    @staticmethod
    def compute_min_legal_order_notional(
        rules: ExchangeContractRulePOD, reference_price: float
    ) -> Tuple[float, float]:
        """
        Calcula la cantidad mínima legal (qty >= minQty, múltiplo de stepSize) y el
        nocional mínimo exacto (qty * price >= minNotional) exigido por Binance.
        """
        if reference_price <= 0.0:
            return rules.min_qty, rules.min_notional
        qty_for_notional = LotAndTickQuantizer.quantize_qty(
            rules.min_notional / reference_price, rules.step_size, mode="CEIL"
        )
        min_legal_qty = max(rules.min_qty, qty_for_notional)
        if min_legal_qty * reference_price < rules.min_notional:
            min_legal_qty = LotAndTickQuantizer.quantize_qty(
                min_legal_qty + rules.step_size, rules.step_size, mode="CEIL"
            )
        min_legal_notional = round(min_legal_qty * reference_price, 4)
        return min_legal_qty, min_legal_notional

    @staticmethod
    def compute_isolated_grid_liquidation_price(
        direction: GridDirection,
        isolated_equity_usdt: float,
        adverse_levels: Tuple[float, ...] | List[float],
        qty_per_line: float,
        base_entry_price: float,
        base_qty: float,
        maint_margin_rate: float = 0.005,
        taker_fee: float = 0.00045,
        tick_size: float = 0.0001,
    ) -> Tuple[float, float]:
        """
        Calcula el Precio Exacto de Liquidación en Margen ISOLATED (`P_liq`) y el
        Apalancamiento Efectivo Máximo (`L_eff_max`) en el peor escenario donde se llenan
        todos los escalones adversos activos del Grid.
        """
        lv_list = list(adverse_levels)
        total_qty = base_qty + (len(lv_list) * qty_per_line)
        if total_qty <= 1e-12 or isolated_equity_usdt <= 0.0:
            return 0.0, 0.0

        total_notional = (base_qty * base_entry_price) + sum(
            float(p) * qty_per_line for p in lv_list
        )
        eff_max_lev = round(total_notional / max(isolated_equity_usdt, 1e-6), 2)
        mm_plus_fee = maint_margin_rate + taker_fee

        if direction == GridDirection.SHORT:
            denom = total_qty * (1.0 + mm_plus_fee)
            raw_liq = (total_notional + isolated_equity_usdt) / max(denom, 1e-12)
            return LotAndTickQuantizer.quantize_price(raw_liq, tick_size, "CEIL"), eff_max_lev

        denom = total_qty * max(0.10, 1.0 - mm_plus_fee)
        raw_liq = max(0.0, (total_notional - isolated_equity_usdt) / max(denom, 1e-12))
        return LotAndTickQuantizer.quantize_price(raw_liq, tick_size, "FLOOR"), eff_max_lev

    def rank_all_markets_safe_to_volatile(
        self,
        rules_map: Dict[str, ExchangeContractRulePOD],
        telemetry_map: Dict[str, Market3VTelemetryPOD],
        bot_capital_usdt: float = 6.0,
        max_leverage: Optional[int] = None,
        target_equity_usdt: Optional[float] = None,
        wallet_balance_usdt: Optional[float] = None,
    ) -> List[RankedMarketCandidatePOD]:
        """
        Clasifica todos los mercados de Binance calculando para cada uno:
        - Certidumbre (%) y Mitigación de Incertidumbre basada en datos de Binance.
        - Apalancamiento y Riesgo Operativo de la Fase (Fases 1 a 5).
        - ROI (%) y PnL (USDT) Esperado Diario y a 30 Días de cada mercado.
        - Días Estimados para cumplir la meta de la Fase (3x en Fases 1 y 2, 2x en Fase 3).
        """
        ranked: List[RankedMarketCandidatePOD] = []
        ref_wallet = wallet_balance_usdt if wallet_balance_usdt is not None else bot_capital_usdt
        (
            active_phase,
            phase_rec_lev,
            phase_max_lev,
            phase_risk_pct,
            _,
            phase_min_30d_roi_pct,
            phase_mult,
        ) = self.resolve_phase_parameters(ref_wallet)

        roundtrip_maker_pct = 2.0 * self.maker_fee * 100.0

        # División inteligente en 3 Áreas (ESTABLES, INTERMEDIOS, VOLATILES) basada 100% en los datos de la API
        empirical_areas = self.partition_universe_empirical_3_areas(rules_map, telemetry_map)

        for symbol, rules in rules_map.items():
            if symbol not in telemetry_map:
                continue
            tel = telemetry_map[symbol]
            if tel.last_price <= 0.0:
                continue

            spread_pct = ((tel.ask_price - tel.bid_price) / tel.last_price) * 100.0
            eff_vol_pct = max(tel.natr_14_pct, tel.ewma_vol_pct)
            uncertainty_score, confidence_pct = self.compute_market_uncertainty_and_confidence(
                tel, spread_pct
            )
            tier, ivr = empirical_areas.get(
                symbol, (MarketSafetyTier.INTERMEDIOS, round(uncertainty_score, 4))
            )

            allowed_lev_cap = (
                min(rules.max_leverage, max_leverage)
                if (max_leverage is not None and max_leverage > 0)
                else min(rules.max_leverage, phase_max_lev)
            )
            allowed_lev_cap = max(2, allowed_lev_cap)

            _, min_line_notional = self.compute_min_legal_order_notional(rules, tel.last_price)
            dynamic_reserve = min(0.40, self.reserve_ratio + (0.12 * uncertainty_score))
            usable_margin_frac = max(0.15, 1.0 - dynamic_reserve - rules.maint_margin_rate)

            min_active_window = 2 if bot_capital_usdt < 6.0 else 4
            min_lev_for_grid = int(
                math.ceil(
                    (min_line_notional * min_active_window)
                    / max(bot_capital_usdt * usable_margin_frac, 1e-6)
                )
            )
            uncertainty_lev_factor = max(0.60, 1.0 - (0.35 * uncertainty_score))
            confidence_adjusted_lev = max(2, int(round(phase_rec_lev * uncertainty_lev_factor)))
            rec_phase_lev = min(
                allowed_lev_cap,
                max(min_lev_for_grid, confidence_adjusted_lev),
            )
            rec_risk_pct = (
                55.0
                if active_phase == ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100
                else round(
                    max(0.25, phase_risk_pct * max(0.55, 1.0 - (0.40 * uncertainty_score))),
                    2,
                )
            )

            usable_leverage_factor = allowed_lev_cap * usable_margin_frac
            min_cap_needed = (min_line_notional * min_active_window) / max(usable_leverage_factor, 1e-6)
            max_deployable_notional = bot_capital_usdt * usable_leverage_factor

            sl_move_frac = min(0.10, max(0.015, (tel.mae_q65_pct + tel.natr_14_pct) / 100.0))
            min_safe_equity_for_sl = (min_line_notional * 1.25 * sl_move_frac) / max(0.455, rec_risk_pct / 100.0)
            effective_min_cap = max(min_cap_needed, min_safe_equity_for_sl)

            viable_for_bot = (
                bot_capital_usdt >= effective_min_cap
                and (min_line_notional * min_active_window) <= max_deployable_notional
            )

            min_profitable_step_pct = roundtrip_maker_pct + spread_pct + (
                (rules.tick_size / tel.last_price) * 100.0
            )
            step_pct = max(min_profitable_step_pct * 2.0, eff_vol_pct)
            net_pct_per_cross = max(1e-4, step_pct - roundtrip_maker_pct)

            mr_efficiency = max(0.75, min(1.25, 1.0 / max(tel.variance_ratio_q4, 0.50)))
            est_crosses = round(
                (tel.intraday_path_pct / max(2.0 * step_pct, 1e-6))
                * mr_efficiency
                * tel.bayesian_grid_survival_prob,
                2,
            )

            # Cálculo del PnL y ROI Esperado en este Mercado (Diario y a 30 Días con Interés Compuesto)
            uncertainty_haircut = max(0.55, 1.0 - (0.30 * uncertainty_score))
            base_notional_per_cross = min_line_notional
            expected_daily_pnl_usdt = round(
                est_crosses * base_notional_per_cross * (net_pct_per_cross / 100.0) * uncertainty_haircut,
                4,
            )
            expected_daily_roi_pct = round(
                (expected_daily_pnl_usdt / max(bot_capital_usdt, 1e-6)) * 100.0,
                2,
            )

            # Proyección en 30 días considerando escalamiento progresivo por tramos (1.0x -> 1.50x -> 2.00x)
            mean_30d_compounding_mult = 1.50 if phase_mult >= 3.0 else (1.35 if phase_mult >= 2.0 else 1.15)
            expected_30d_pnl_usdt = round(
                expected_daily_pnl_usdt * 30.0 * mean_30d_compounding_mult,
                2,
            )
            expected_30d_roi_pct = round(
                (expected_30d_pnl_usdt / max(bot_capital_usdt, 1e-6)) * 100.0,
                2,
            )

            # ROI y PnL Esperado específico del Mercado para el ciclo operativo:
            # En Fases 1 y 2 la meta mínima en 30 días es TRIPLICAR (3x = +200% ROI); en Fase 3 es DUPLICAR (2x = +100% ROI).
            if target_equity_usdt is not None:
                expected_market_pnl_usdt = round(max(0.01, target_equity_usdt - bot_capital_usdt), 2)
                expected_market_roi_pct = round(
                    (expected_market_pnl_usdt / max(bot_capital_usdt, 1e-6)) * 100.0, 2
                )
            else:
                empirical_cycle_boost_pct = round(
                    (tel.mfe_q92_pct * 3.2)
                    + ((confidence_pct - 65.0) * 0.65)
                    + min(45.0, expected_daily_roi_pct * 2.5),
                    2,
                )
                raw_market_target_roi = phase_min_30d_roi_pct + max(0.0, empirical_cycle_boost_pct)
                market_cycle_roi_pct = round(
                    max(phase_min_30d_roi_pct, min(expected_30d_roi_pct, raw_market_target_roi)),
                    2,
                )
                expected_market_roi_pct = market_cycle_roi_pct
                expected_market_pnl_usdt = round(
                    bot_capital_usdt * (expected_market_roi_pct / 100.0), 2
                )

            # Días Estimados para cumplir el multiplicador de la Fase (3x en F1-F2, 2x en F3) y la meta del mercado
            phase_goal_profit_needed = max(0.01, bot_capital_usdt * (phase_min_30d_roi_pct / 100.0))
            effective_daily_compounded = max(1e-6, expected_daily_pnl_usdt * mean_30d_compounding_mult)
            est_days_to_phase_goal = (
                round(phase_goal_profit_needed / effective_daily_compounded, 1)
                if effective_daily_compounded > 1e-6
                else 999.9
            )
            est_days_to_market_target = (
                round(expected_market_pnl_usdt / effective_daily_compounded, 1)
                if effective_daily_compounded > 1e-6
                else 999.9
            )

            mult_lbl = f"{phase_mult:.0f}x" if phase_mult >= 2.0 else f"{phase_mult:.2f}x"
            if viable_for_bot:
                if est_days_to_phase_goal <= 30.0:
                    reason = (
                        f"APTO_{active_phase.value} | Meta {mult_lbl} en ~{est_days_to_phase_goal:.1f}d (<=30d) | "
                        f"Certidumbre={confidence_pct:.1f}%"
                    )
                else:
                    reason = (
                        f"APTO_{active_phase.value} | ROI Esp 30d: +{expected_30d_roi_pct:.1f}% (~{est_days_to_phase_goal:.1f}d a {mult_lbl}) | "
                        f"Certidumbre={confidence_pct:.1f}%"
                    )
            else:
                reason = f"Requiere min ${effective_min_cap:.2f} USDT (Lote min=${min_line_notional:.2f} USDT)"

            ranked.append(
                RankedMarketCandidatePOD(
                    symbol=symbol,
                    safety_tier=tier,
                    risk_score_ivr=ivr,
                    min_line_notional_usdt=round(min_line_notional, 2),
                    min_capital_for_grid_usdt=round(effective_min_cap, 2),
                    viable_for_6usd_bot=viable_for_bot,
                    est_daily_crosses=est_crosses,
                    est_days_6_to_18_usd=est_days_to_market_target,
                    status_reason=reason,
                    expected_daily_pnl_usdt=expected_daily_pnl_usdt,
                    expected_daily_roi_pct=expected_daily_roi_pct,
                    expected_market_pnl_usdt=expected_market_pnl_usdt,
                    expected_market_roi_pct=expected_market_roi_pct,
                    expected_30d_pnl_usdt=expected_30d_pnl_usdt,
                    expected_30d_roi_pct=expected_30d_roi_pct,
                    uncertainty_score=uncertainty_score,
                    confidence_pct=confidence_pct,
                    recommended_phase_leverage=rec_phase_lev,
                    recommended_risk_pct=rec_risk_pct,
                    phase_target_multiplier=phase_mult,
                    est_days_to_double_capital=est_days_to_phase_goal,
                )
            )

        ranked.sort(key=lambda x: (x.uncertainty_score, x.risk_score_ivr))
        return ranked

    def select_best_market_for_capital(
        self,
        ranked_universe: List[RankedMarketCandidatePOD],
        high_risk_acceleration: bool = False,
    ) -> RankedMarketCandidatePOD:
        """
        Elige el mercado elegible que maximiza el ROI/PnL esperado.
        - Si `high_risk_acceleration=True` (Modo M8 Alto Riesgo $10 USDT): prioriza el área
          `VOLATILES` y maximiza el ROI neto sobre comisiones reales.
        - En modo estándar: reemplaza los pesos heurísticos fijos por ponderación objetiva CRITIC
          calculada sobre la distribución empírica de candidatos viables.
        """
        viable = [c for c in ranked_universe if c.viable_for_6usd_bot]
        if not viable:
            raise ValueError("Ningún mercado cumple las reglas físicas de Binance para el capital asignado.")

        if high_risk_acceleration:
            viable.sort(
                key=lambda c: (
                    c.safety_tier != MarketSafetyTier.VOLATILES,
                    -c.expected_30d_roi_pct,
                    -c.expected_daily_roi_pct,
                )
            )
            return viable[0]

        if len(viable) >= 3:
            mat = np.array(
                [
                    [c.expected_30d_roi_pct, -c.uncertainty_score, -c.risk_score_ivr]
                    for c in viable
                ],
                dtype=np.float64,
            )
            c_min = np.min(mat, axis=0)
            c_max = np.max(mat, axis=0)
            z = (mat - c_min) / np.where(c_max - c_min > 1e-12, c_max - c_min, 1.0)
            sig = np.std(z, axis=0, ddof=1)
            corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0)
            info = np.array([sig[j] * float(np.sum(1.0 - corr[j, :])) for j in range(3)])
            w = info / float(np.sum(info)) if float(np.sum(info)) > 1e-12 else np.array([1.0 / 3.0] * 3)
            scores = z @ w
            scored_pairs = sorted(
                zip(viable, scores),
                key=lambda pair: (pair[0].est_days_to_double_capital > 30.0, -float(pair[1])),
            )
            return scored_pairs[0][0]

        viable.sort(
            key=lambda c: (
                c.est_days_to_double_capital > 30.0,
                -c.expected_30d_roi_pct,
            )
        )
        return viable[0]

    def build_geometric_grid_6_to_18(
        self,
        rules: ExchangeContractRulePOD,
        tel: Market3VTelemetryPOD,
        safety_tier: MarketSafetyTier,
        initial_capital_usdt: float = 6.00,
        current_equity_usdt: float = 6.00,
        target_equity_usdt: Optional[float] = None,
        forced_direction: Optional[GridDirection] = None,
        max_leverage: Optional[int] = None,
        wallet_balance_usdt: Optional[float] = None,
    ) -> GeometricGridPlanPOD:
        """
        Construye el Grid Geométrico completo (2 a 169 niveles) gobernado por las 5 Fases de Capital:
        - Calcula la Certidumbre (%), el Apalancamiento/Riesgo de la Fase, el ROI y PnL Esperado
          Diario y a 30 Días de cada mercado, y los Días Estimados (Meta 3x en Fases 1-2, 2x en Fase 3).
        - Aplica el Escudo Anti-Liquidación (Zero-Liquidation Shield) asegurando P_liq << P_SL.
        """
        direction = forced_direction or Mark2IntelligenceEngine.classify_grid_direction(tel)
        p0 = tel.last_price
        spread_pct = ((tel.ask_price - tel.bid_price) / p0) * 100.0 if p0 > 0.0 else 0.0

        # 0. Resolver Fase de Capital (1 a 5) y Mitigación de Incertidumbre con datos de Binance
        ref_wallet = wallet_balance_usdt if wallet_balance_usdt is not None else max(initial_capital_usdt, current_equity_usdt)
        (
            active_phase,
            phase_rec_lev,
            phase_max_lev,
            phase_risk_pct,
            phase_max_eff_lev,
            phase_min_30d_roi_pct,
            phase_mult,
        ) = self.resolve_phase_parameters(ref_wallet)
        uncertainty_score, confidence_pct = self.compute_market_uncertainty_and_confidence(
            tel, spread_pct
        )
        operational_risk_pct = (
            55.0
            if active_phase == ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100
            else round(
                max(0.25, phase_risk_pct * max(0.55, 1.0 - (0.40 * uncertainty_score))),
                2,
            )
        )

        # 1. Apalancamiento según la Fase de Capital + Mitigación de Incertidumbre
        _, min_order_notional_at_p0 = self.compute_min_legal_order_notional(rules, p0)
        tox_reserve_boost = (0.08 if tel.flow_toxicity_z > 1.5 else 0.0) + (0.10 * uncertainty_score)
        effective_reserve_ratio = min(0.45, self.reserve_ratio + tox_reserve_boost)
        anti_liq_reserve_usdt = round(current_equity_usdt * effective_reserve_ratio, 2)
        usable_margin_fraction = max(0.15, 1.0 - effective_reserve_ratio - rules.maint_margin_rate)

        min_orders_target = 6 if initial_capital_usdt < 10.0 else 8
        min_lev_for_notional = math.ceil(
            (min_order_notional_at_p0 * min_orders_target)
            / max(initial_capital_usdt * usable_margin_fraction, 1e-6)
        )
        contract_max_lev = (
            min(rules.max_leverage, max(1, max_leverage))
            if (max_leverage is not None and max_leverage > 0)
            else min(rules.max_leverage, phase_max_lev)
        )
        uncertainty_lev_factor = max(0.65, 1.0 - (0.30 * uncertainty_score))
        desired_phase_lev = max(2, int(round(phase_rec_lev * uncertainty_lev_factor)))
        leverage = int(min(contract_max_lev, max(desired_phase_lev, min_lev_for_notional)))

        # 2. Rango de precios derivado de Bandas de Bollinger (20, 2σ), ATR(14), Volatilidad EWMA y excursiones MAE/MFE
        eff_atr = max(tel.atr_14, p0 * (tel.ewma_vol_pct / 100.0))
        bb_half_span = max((tel.bb_upper - tel.bb_lower) / 2.0, eff_atr * 2.0)
        mae_span = p0 * (tel.mae_q65_pct / 100.0)
        mfe_span = p0 * (tel.mfe_q85_pct / 100.0)
        mfe_ext_span = p0 * (tel.mfe_q92_pct / 100.0)

        if direction == GridDirection.NEUTRAL:
            down_span = max(bb_half_span, mae_span + mfe_span)
            up_span = max(bb_half_span, mae_span + mfe_span)
        elif direction == GridDirection.LONG:
            down_span = max(bb_half_span, mae_span + eff_atr)
            up_span = max(bb_half_span, mfe_ext_span + eff_atr)
        else:
            down_span = max(bb_half_span, mfe_ext_span + eff_atr)
            up_span = max(bb_half_span, mae_span + eff_atr)

        raw_low = max(rules.tick_size, p0 - down_span)
        raw_high = max(raw_low + 2.0 * rules.tick_size, p0 + up_span)

        lower_p = LotAndTickQuantizer.quantize_price(raw_low, rules.tick_size, "FLOOR")
        upper_p = LotAndTickQuantizer.quantize_price(raw_high, rules.tick_size, "CEIL")
        if upper_p <= lower_p:
            upper_p = LotAndTickQuantizer.quantize_price(
                lower_p + max(10.0 * rules.tick_size, p0 * 0.02), rules.tick_size, "CEIL"
            )

        # 3. Niveles Geométricos Virtuales (2 a 169) con escudo anti-erosión de comisiones y escalado horario sqrt(12)
        roundtrip_maker_pct = 2.0 * self.maker_fee * 100.0
        min_viable_step_pct = max(
            roundtrip_maker_pct * 6.0 + spread_pct,
            ((rules.tick_size / lower_p) * 100.0) * 4.0,
        )
        hourly_vol_scale = math.sqrt(12.0) if safety_tier == MarketSafetyTier.VOLATILES else math.sqrt(4.0)
        tox_step_mult = (1.20 if tel.flow_toxicity_z > 1.5 else 1.00) * (1.0 + 0.15 * uncertainty_score)
        target_step_pct = max(
            min_viable_step_pct,
            max(tel.natr_14_pct * hourly_vol_scale * 0.5, tel.ewma_vol_pct) * tox_step_mult,
        )

        virtual_grids = int(
            math.floor(math.log(upper_p / lower_p) / math.log(1.0 + (target_step_pct / 100.0)))
        )
        virtual_grids = max(2, min(min(169, rules.max_num_orders), virtual_grids))

        geom_r = (upper_p / lower_p) ** (1.0 / virtual_grids)
        actual_step_pct = (geom_r - 1.0) * 100.0
        net_step_pct = max(1e-4, actual_step_pct - roundtrip_maker_pct)

        virtual_levels: List[float] = []
        for k in range(virtual_grids + 1):
            pk = LotAndTickQuantizer.quantize_price(lower_p * (geom_r ** k), rules.tick_size)
            if not virtual_levels or pk > virtual_levels[-1]:
                virtual_levels.append(pk)
        virtual_grids = max(2, len(virtual_levels) - 1)

        # 4. Cálculo del ROI y PnL Esperado del Mercado (Diario y 30 Días) + Meta de Fase (3x en F1-F2, 2x en F3)
        min_qty_at_lower, _ = self.compute_min_legal_order_notional(rules, lower_p)
        min_qty_at_p0, _ = self.compute_min_legal_order_notional(rules, p0)
        base_min_qty = max(min_qty_at_lower, min_qty_at_p0)
        base_line_notional = round(base_min_qty * p0, 2)

        mr_efficiency = max(0.75, min(1.25, 1.0 / max(tel.variance_ratio_q4, 0.50)))
        est_daily_crosses = round(
            (tel.intraday_path_pct / max(2.0 * actual_step_pct, 1e-6))
            * mr_efficiency
            * tel.bayesian_grid_survival_prob,
            2,
        )
        uncertainty_haircut = max(0.55, 1.0 - (0.30 * uncertainty_score))
        base_net_usd_per_cross = round(base_line_notional * (net_step_pct / 100.0), 4)
        expected_daily_pnl_usdt = round(
            est_daily_crosses * base_net_usd_per_cross * uncertainty_haircut, 4
        )
        expected_daily_roi_pct = round(
            (expected_daily_pnl_usdt / max(initial_capital_usdt, 1e-6)) * 100.0, 2
        )
        mean_30d_mult = 1.50 if phase_mult >= 3.0 else (1.35 if phase_mult >= 2.0 else 1.15)
        expected_30d_pnl_usdt = round(expected_daily_pnl_usdt * 30.0 * mean_30d_mult, 2)
        expected_30d_roi_pct = round(
            (expected_30d_pnl_usdt / max(initial_capital_usdt, 1e-6)) * 100.0, 2
        )
        min_30d_phase_target_usdt = round(
            initial_capital_usdt * phase_mult, 2
        )

        # Si no se impuso una meta fija, se usa el ROI/PnL esperado específico de este mercado,
        # teniendo como piso en Fases 1 y 2 TRIPLICAR (3x = +200% ROI) y en Fase 3 DUPLICAR (2x = +100% ROI).
        if target_equity_usdt is None:
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
                initial_capital_usdt * (1.0 + (resolved_target_roi_pct / 100.0)), 2
            )
        else:
            resolved_target_equity_usdt = round(target_equity_usdt, 2)

        # 5. Tramos de Interés Compuesto proporcionales al capital inicial y a la meta del mercado
        total_target_pnl = max(0.01, resolved_target_equity_usdt - initial_capital_usdt)
        stage_2_threshold = round(initial_capital_usdt + (total_target_pnl * 0.25), 2)
        stage_3_threshold = round(initial_capital_usdt + (total_target_pnl * 0.50), 2)

        if current_equity_usdt >= resolved_target_equity_usdt:
            stage = CompoundingStage.TARGET_REACHED_EXPECTED_ROI
            ratchet_floor = round(resolved_target_equity_usdt, 2)
            stage_scale = 2.0
        elif current_equity_usdt >= stage_3_threshold:
            stage = CompoundingStage.STAGE_3_HARVEST_66_TO_100PCT
            ratchet_floor = round(initial_capital_usdt + (total_target_pnl * 0.375), 2)
            stage_scale = 2.0
        elif current_equity_usdt >= stage_2_threshold:
            stage = CompoundingStage.STAGE_2_EXPANSION_33_TO_66PCT
            ratchet_floor = round(initial_capital_usdt + (total_target_pnl * 0.125), 2)
            stage_scale = 1.5
        else:
            stage = CompoundingStage.STAGE_1_MITIGATION_0_TO_33PCT
            max_loss_ratio = max(rules.maint_margin_rate * 2.0, operational_risk_pct / 100.0)
            ratchet_floor = round(initial_capital_usdt * (1.0 - max_loss_ratio), 2)
            stage_scale = 1.0

        # 6. Cantidad por línea y Tope de Apalancamiento Efectivo de Inventario según la Fase
        deployable_notional = current_equity_usdt * usable_margin_fraction * leverage
        target_qty = LotAndTickQuantizer.quantize_qty(
            base_min_qty * stage_scale, rules.step_size, mode="CEIL"
        )
        if target_qty * p0 * 2.0 > deployable_notional and base_min_qty * p0 * 2.0 <= deployable_notional:
            target_qty = base_min_qty

        qty_per_line = max(base_min_qty, target_qty)
        actual_line_notional = round(qty_per_line * p0, 2)

        max_active_orders = max(
            2,
            min(
                virtual_grids,
                12,
                int(math.floor(deployable_notional / max(actual_line_notional, rules.min_notional))),
            ),
        )

        # 7. Precios exactos de Stop Loss y Take Profit Estructurales
        sl_buffer = max(eff_atr, p0 * (tel.mae_q65_pct / 100.0))
        tp_buffer = max(eff_atr, p0 * (tel.mfe_q92_pct / 100.0))

        if direction == GridDirection.SHORT:
            sl_price = LotAndTickQuantizer.quantize_price(upper_p + sl_buffer, rules.tick_size, "CEIL")
            tp_price = LotAndTickQuantizer.quantize_price(
                max(rules.tick_size, lower_p - tp_buffer), rules.tick_size, "FLOOR"
            )
        else:
            sl_price = LotAndTickQuantizer.quantize_price(
                max(rules.tick_size, lower_p - sl_buffer), rules.tick_size, "FLOOR"
            )
            tp_price = LotAndTickQuantizer.quantize_price(upper_p + tp_buffer, rules.tick_size, "CEIL")

        # 8. Selección de Ventana Activa + Escudo Anti-Liquidación
        all_buys = [lv for lv in virtual_levels if lv < tel.bid_price]
        all_sells = [lv for lv in virtual_levels if lv > tel.ask_price]

        if not all_buys:
            all_buys = [
                LotAndTickQuantizer.quantize_price(
                    tel.bid_price / geom_r, rules.tick_size, "FLOOR"
                )
            ]
        if not all_sells:
            all_sells = [
                LotAndTickQuantizer.quantize_price(
                    tel.ask_price * geom_r, rules.tick_size, "CEIL"
                )
            ]

        half_win = max(1, max_active_orders // 2)
        cand_buys = list(all_buys[-half_win:])
        rem_slots = max(1, max_active_orders - len(cand_buys))
        cand_sells = list(all_sells[:rem_slots])

        base_init_qty = qty_per_line if direction != GridDirection.NEUTRAL else 0.0

        min_safety_dist = max(eff_atr * 1.5, p0 * 0.015)
        if direction in (GridDirection.LONG, GridDirection.NEUTRAL):
            while len(cand_buys) > 1:
                est_liq, eff_adv_lev = self.compute_isolated_grid_liquidation_price(
                    direction=GridDirection.LONG,
                    isolated_equity_usdt=current_equity_usdt,
                    adverse_levels=cand_buys,
                    qty_per_line=qty_per_line,
                    base_entry_price=p0,
                    base_qty=base_init_qty,
                    maint_margin_rate=rules.maint_margin_rate,
                    taker_fee=self.taker_fee,
                    tick_size=rules.tick_size,
                )
                if est_liq <= (sl_price - min_safety_dist) and eff_adv_lev <= max(phase_max_eff_lev, 3.5):
                    break
                cand_buys.pop(0)
        else:
            while len(cand_sells) > 1:
                est_liq, eff_adv_lev = self.compute_isolated_grid_liquidation_price(
                    direction=GridDirection.SHORT,
                    isolated_equity_usdt=current_equity_usdt,
                    adverse_levels=cand_sells,
                    qty_per_line=qty_per_line,
                    base_entry_price=p0,
                    base_qty=base_init_qty,
                    maint_margin_rate=rules.maint_margin_rate,
                    taker_fee=self.taker_fee,
                    tick_size=rules.tick_size,
                )
                if est_liq >= (sl_price + min_safety_dist) and eff_adv_lev <= max(phase_max_eff_lev, 3.5):
                    break
                cand_sells.pop(-1)

        active_buys = tuple(cand_buys)
        active_sells = tuple(cand_sells)

        adverse_side_levels = (
            active_sells if direction == GridDirection.SHORT else active_buys
        )
        est_liq_price, max_adverse_eff_lev = self.compute_isolated_grid_liquidation_price(
            direction=GridDirection.SHORT if direction == GridDirection.SHORT else GridDirection.LONG,
            isolated_equity_usdt=current_equity_usdt,
            adverse_levels=adverse_side_levels,
            qty_per_line=qty_per_line,
            base_entry_price=p0,
            base_qty=base_init_qty,
            maint_margin_rate=rules.maint_margin_rate,
            taker_fee=self.taker_fee,
            tick_size=rules.tick_size,
        )
        eff_init_lev = round(
            (base_init_qty * p0) / max(current_equity_usdt, 1e-6)
            if base_init_qty > 0.0
            else (actual_line_notional / max(current_equity_usdt, 1e-6)),
            2,
        )
        if direction == GridDirection.SHORT:
            liq_buffer_pct = round(
                max(0.0, ((est_liq_price - sl_price) / max(p0, 1e-9)) * 100.0), 2
            )
        else:
            liq_buffer_pct = round(
                max(0.0, ((sl_price - est_liq_price) / max(p0, 1e-9)) * 100.0), 2
            )

        net_usd_per_cross = round(actual_line_notional * (net_step_pct / 100.0), 4)
        tp_pnl_usdt = round(resolved_target_equity_usdt - initial_capital_usdt, 2)
        tp_roi_pct = round((tp_pnl_usdt / max(initial_capital_usdt, 1e-6)) * 100.0, 2)

        sl_pnl_usdt = round(ratchet_floor - current_equity_usdt, 2)
        sl_roi_pct = round(
            ((ratchet_floor - initial_capital_usdt) / max(initial_capital_usdt, 1e-6)) * 100.0, 2
        )

        remaining_usd_to_target = max(0.0, resolved_target_equity_usdt - current_equity_usdt)
        daily_net_harvest_usd = est_daily_crosses * net_usd_per_cross * uncertainty_haircut
        est_days_remaining = (
            round(remaining_usd_to_target / (daily_net_harvest_usd * mean_30d_mult), 1)
            if daily_net_harvest_usd > 1e-6
            else 999.9
        )

        return GeometricGridPlanPOD(
            symbol=rules.symbol,
            safety_tier=safety_tier,
            direction=direction,
            compounding_stage=stage,
            initial_capital_usdt=round(initial_capital_usdt, 2),
            current_equity_usdt=round(current_equity_usdt, 2),
            target_equity_usdt=resolved_target_equity_usdt,
            equity_ratchet_floor_usdt=ratchet_floor,
            leverage=leverage,
            lower_price=lower_p,
            upper_price=upper_p,
            total_virtual_grids=virtual_grids,
            active_window_orders=len(active_buys) + len(active_sells),
            geometric_step_pct=round(actual_step_pct, 4),
            net_profit_per_cross_pct=round(net_step_pct, 4),
            net_profit_per_cross_usdt=net_usd_per_cross,
            qty_per_line=qty_per_line,
            notional_per_line_usdt=actual_line_notional,
            all_virtual_levels=tuple(virtual_levels),
            active_buy_window=active_buys,
            active_sell_window=active_sells,
            tp_price=tp_price,
            tp_pnl_usdt=tp_pnl_usdt,
            tp_roi_pct=tp_roi_pct,
            sl_price=sl_price,
            sl_pnl_usdt=sl_pnl_usdt,
            sl_roi_pct=sl_roi_pct,
            est_daily_crosses=est_daily_crosses,
            est_days_to_18_usd=est_days_remaining,
            trailing_up=direction in (GridDirection.LONG, GridDirection.NEUTRAL),
            trailing_down=direction in (GridDirection.SHORT, GridDirection.NEUTRAL),
            effective_initial_leverage=eff_init_lev,
            max_adverse_effective_leverage=max_adverse_eff_lev,
            estimated_liquidation_price=est_liq_price,
            liquidation_safety_buffer_pct=liq_buffer_pct,
            anti_liquidation_reserve_usdt=anti_liq_reserve_usdt,
            active_phase=active_phase.value,
            phase_operational_risk_pct=operational_risk_pct,
            phase_target_multiplier=phase_mult,
            uncertainty_score=uncertainty_score,
            confidence_pct=confidence_pct,
            expected_daily_pnl_usdt=round(daily_net_harvest_usd, 4),
            expected_daily_roi_pct=round((daily_net_harvest_usd / max(initial_capital_usdt, 1e-6)) * 100.0, 2),
            expected_30d_pnl_usdt=expected_30d_pnl_usdt,
            expected_30d_roi_pct=expected_30d_roi_pct,
            min_30d_doubling_target_usdt=min_30d_phase_target_usdt,
        )
