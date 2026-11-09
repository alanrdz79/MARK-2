# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python m2_intelligence_regime_ml.py
Transforma matrices de velas reales de Binance (/fapi/v1/klines o SQLite Warehouse),
cotizaciones L1 (/fapi/v1/ticker/bookTicker) y derivados (/fapi/v1/premiumIndex) en el
perfil de telemetría 100% empírico enriquecido con Ciencia de Datos Cuantitativa:
1. Volatilidad Condicional EWMA (RiskMetrics / GARCH(1,1) proxy con lambda=0.94).
2. Test de Razón de Varianza de Lo-MacKinlay VR(q=4) para confirmar reversión a la media.
3. Toxicidad de Flujo de Órdenes (VPIN / Order Imbalance Z-Score) contra cascadas de liquidación.
4. Estimador Bayesiano Beta-Binomial de Supervivencia de Rango del Grid.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from core_mark2_contracts import GridDirection, Market3VTelemetryPOD


class Mark2IntelligenceEngine:
    """
    Motor de telemetría cuantitativa y Ciencia de Datos en memoria RAM O(1)
    basado exclusivamente en datos reales de Binance Futures USDⓈ-M.
    """

    @staticmethod
    def _compute_ema(series: np.ndarray, span: int) -> float:
        """Calcula la Media Móvil Exponencial (EMA) exacta sobre la serie de cierres reales."""
        n = len(series)
        if n == 0:
            return 0.0
        alpha = 2.0 / (min(span, n) + 1.0)
        ema_val = float(series[0])
        for val in series[1:]:
            ema_val = alpha * float(val) + (1.0 - alpha) * ema_val
        return ema_val

    @staticmethod
    def _compute_wilder_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """Calcula el ATR de Wilder real a partir de las series High, Low y Close de Binance."""
        n = len(closes)
        if n == 0:
            return 0.0
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
        )
        if n <= period:
            return float(np.mean(tr))
        atr = float(np.mean(tr[:period]))
        alpha = 1.0 / float(period)
        for val in tr[period:]:
            atr = alpha * float(val) + (1.0 - alpha) * atr
        return atr

    @staticmethod
    def _compute_empirical_excursions_pct(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        horizon_bars: int = 12,
    ) -> tuple[float, float, float]:
        """
        Mide las excursiones reales (MAE q65, MFE q85 y MFE q92) directamente sobre
        ventanas rodantes de las velas históricas de Binance.
        """
        n = len(closes)
        w = max(2, min(horizon_bars, n // 2))
        if n <= w:
            rel_rng = ((highs - lows) / np.maximum(closes, 1e-12)) * 100.0
            return (
                float(np.percentile(rel_rng, 65)),
                float(np.percentile(rel_rng, 85)),
                float(np.percentile(rel_rng, 92)),
            )

        up_excursions = np.empty(n - w, dtype=np.float64)
        down_excursions = np.empty(n - w, dtype=np.float64)
        for i in range(n - w):
            c0 = max(float(closes[i]), 1e-12)
            max_h = float(np.max(highs[i + 1 : i + 1 + w]))
            min_l = float(np.min(lows[i + 1 : i + 1 + w]))
            up_excursions[i] = max(0.0, ((max_h - c0) / c0) * 100.0)
            down_excursions[i] = max(0.0, ((c0 - min_l) / c0) * 100.0)

        joint_adverse = np.minimum(up_excursions, down_excursions)
        joint_favorable = np.maximum(up_excursions, down_excursions)

        mae_q65 = float(np.percentile(joint_adverse, 65))
        mfe_q85 = float(np.percentile(joint_favorable, 85))
        mfe_q92 = float(np.percentile(joint_favorable, 92))
        return mae_q65, mfe_q85, mfe_q92

    @staticmethod
    def _compute_ewma_volatility_pct(closes: np.ndarray, decay_lambda: float = 0.94) -> float:
        """
        Pronóstico de Volatilidad Condicional EWMA (RiskMetrics / GARCH proxy):
        Captura explosiones súbitas de varianza asignando peso exponencial a los retornos recientes.
        """
        if len(closes) < 5:
            return 0.0
        log_rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        var_t = float(np.var(log_rets[: min(10, len(log_rets))]))
        one_minus_lam = 1.0 - decay_lambda
        for r in log_rets:
            var_t = (decay_lambda * var_t) + (one_minus_lam * float(r * r))
        return math.sqrt(max(var_t, 0.0)) * 100.0

    @staticmethod
    def _compute_variance_ratio_q(closes: np.ndarray, q: int = 4) -> float:
        """
        Estadístico de Razón de Varianza de Lo-MacKinlay VR(q):
        - VR(q) < 1.0: Autocorrelación negativa (Reversión a la Media pura -> Óptimo para Grids).
        - VR(q) ~ 1.0: Paseo aleatorio geométrico.
        - VR(q) > 1.0: Persistencia / Tendencia direccional.
        """
        if len(closes) < (q * 4):
            return 1.0
        log_p = np.log(np.maximum(closes, 1e-12))
        r1 = np.diff(log_p)
        rq = log_p[q:] - log_p[:-q]
        var_1 = float(np.var(r1, ddof=1)) if len(r1) > 1 else 0.0
        var_q = float(np.var(rq, ddof=1)) if len(rq) > 1 else 0.0
        if var_1 <= 1e-15:
            return 1.0
        vr = var_q / (float(q) * var_1)
        return float(max(0.10, min(3.00, vr)))

    @staticmethod
    def _compute_order_flow_toxicity_z(
        volumes: np.ndarray,
        taker_buy_volumes: Optional[np.ndarray],
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> float:
        """
        Mide el Z-Score de Toxicidad de Flujo de Órdenes (inspirado en VPIN de Easley, López de Prado y O'Hara):
        Detecta cuándo el desbalance direccional agresivo supera la distribución histórica reciente,
        alertando de cascadas de liquidación institucional.
        """
        n = len(volumes)
        if n < 10:
            return 0.0
        safe_vols = np.maximum(volumes, 1e-12)
        if taker_buy_volumes is not None and len(taker_buy_volumes) == n:
            taker_sell = np.maximum(0.0, volumes - taker_buy_volumes)
            abs_imbalance = np.abs(taker_buy_volumes - taker_sell) / safe_vols
        else:
            hl = np.maximum(highs - lows, 1e-12)
            clv = np.abs(((closes - lows) - (highs - closes)) / hl)
            abs_imbalance = clv

        recent_tox = float(np.mean(abs_imbalance[-3:]))
        mean_tox = float(np.mean(abs_imbalance))
        std_tox = float(np.std(abs_imbalance))
        if std_tox <= 1e-9:
            return 0.0
        z_val = (recent_tox - mean_tox) / std_tox
        return float(max(-3.0, min(5.0, z_val)))

    @staticmethod
    def _compute_bayesian_grid_survival_prob(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        atr_14: float,
        bb_upper: float,
        bb_lower: float,
        horizon_bars: int = 16,
    ) -> float:
        """
        Estimador Bayesiano Beta-Binomial de Supervivencia de Rango del Grid:
        Evalúa en cuántas ventanas rodantes de `horizon_bars` la excursión adversa no rompe
        el semiancho estructural del Grid (Bollinger + ATR) usando un prior conjugado Beta(3, 1).
        """
        n = len(closes)
        p_last = float(closes[-1]) if n > 0 else 1.0
        half_grid_span = max((bb_upper - bb_lower) * 0.5 + atr_14, p_last * 0.015)
        allowed_move_pct = (half_grid_span / max(p_last, 1e-12)) * 100.0

        w = max(2, min(horizon_bars, n // 2))
        if n <= w:
            return 0.85

        successes = 0
        trials = n - w
        for i in range(trials):
            c0 = max(float(closes[i]), 1e-12)
            max_h = float(np.max(highs[i + 1 : i + 1 + w]))
            min_l = float(np.min(lows[i + 1 : i + 1 + w]))
            adverse_pct = min(((max_h - c0) / c0) * 100.0, ((c0 - min_l) / c0) * 100.0)
            if adverse_pct <= allowed_move_pct:
                successes += 1

        alpha_prior = 3.0
        beta_prior = 1.0
        posterior_mean = (alpha_prior + successes) / (alpha_prior + beta_prior + trials)
        return float(round(max(0.25, min(0.99, posterior_mean)), 4))

    @staticmethod
    def compute_telemetry_from_ohlcv(
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        bid: float,
        ask: float,
        bid_qty: float,
        ask_qty: float,
        mark_price: float,
        quote_vol_24h: float,
        funding_rate: float = 0.0,
        timestamp_ms: int = 0,
        taker_buy_volumes: Optional[np.ndarray] = None,
        bars_per_24h: int = 288,
    ) -> Market3VTelemetryPOD:
        p_last = float(closes[-1])
        n = len(closes)

        # 1. ATR(14) real de Wilder y NATR(14)% sobre el precio de cierre actual
        atr_14 = Mark2IntelligenceEngine._compute_wilder_atr(highs, lows, closes, period=14)
        natr_14_pct = (atr_14 / p_last) * 100.0 if p_last > 0.0 else 0.0

        # 2. Bandas de Bollinger reales (20 periodos, 2 desviaciones estándar)
        window_20 = closes[-20:] if n >= 20 else closes
        bb_mid = float(np.mean(window_20))
        bb_std = float(np.std(window_20))
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = max(1e-12, bb_mid - 2.0 * bb_std)

        # 3. Exponente de Hurst por Rango Reescalado (R/S) sobre retornos logarítmicos reales
        diffs = np.diff(np.log(np.maximum(closes[-min(n, 100):], 1e-12)))
        if len(diffs) >= 10 and float(np.std(diffs)) > 1e-12:
            cum_dev = np.cumsum(diffs - np.mean(diffs))
            r_s = (float(np.max(cum_dev)) - float(np.min(cum_dev))) / float(np.std(diffs))
            hurst = float(math.log(max(r_s, 1.0001)) / math.log(len(diffs)))
        else:
            hurst = 0.50

        # 4. EMA(50) real y desviación porcentual respecto a la EMA
        ema_50 = Mark2IntelligenceEngine._compute_ema(closes[-min(n, 100):], span=50)
        ema_slope_pct = ((p_last - ema_50) / ema_50) * 100.0 if ema_50 > 0.0 else 0.0

        # 5. Desbalance real de Libro L1 (Quote OBI) desde /fapi/v1/ticker/bookTicker
        bid_notional = bid * bid_qty
        ask_notional = ask * ask_qty
        denom_obi = bid_notional + ask_notional
        quote_obi = (bid_notional - ask_notional) / denom_obi if denom_obi > 0.0 else 0.0

        # 6. Cumulative Volume Delta (CVD) real de las últimas 3 velas desde takerBuyBaseAssetVolume (k[9])
        if taker_buy_volumes is not None and len(taker_buy_volumes) == n:
            tb_3 = taker_buy_volumes[-3:]
            vol_3 = volumes[-3:]
            ts_3 = np.maximum(0.0, vol_3 - tb_3)
            denom_vol = float(np.sum(vol_3))
            cvd_norm_3 = float(np.sum(tb_3 - ts_3)) / denom_vol if denom_vol > 0.0 else 0.0
        else:
            hl_span = np.maximum(highs[-3:] - lows[-3:], 1e-12)
            clv = ((closes[-3:] - lows[-3:]) - (highs[-3:] - closes[-3:])) / hl_span
            denom_vol = float(np.sum(volumes[-3:]))
            cvd_norm_3 = (
                float(np.sum(clv * volumes[-3:])) / denom_vol if denom_vol > 0.0 else 0.0
            )

        # 7. Excursiones reales MAE/MFE y recorrido acumulado real
        mae_q65, mfe_q85, mfe_q92 = Mark2IntelligenceEngine._compute_empirical_excursions_pct(
            closes, highs, lows, horizon_bars=12
        )
        rel_ranges_pct = ((highs - lows) / np.maximum(closes, 1e-12)) * 100.0
        intraday_bars = min(n, bars_per_24h)
        intraday_path_pct = (
            float(np.sum(rel_ranges_pct[-intraday_bars:]))
            * (float(bars_per_24h) / float(max(1, intraday_bars)))
        )

        # 8. Funciones Avanzadas de Ciencia de Datos (EWMA Vol, Variance Ratio VR(4), Toxicidad VPIN y Supervivencia Bayesiana)
        ewma_vol_pct = Mark2IntelligenceEngine._compute_ewma_volatility_pct(closes, decay_lambda=0.94)
        vr_q4 = Mark2IntelligenceEngine._compute_variance_ratio_q(closes, q=4)
        tox_z = Mark2IntelligenceEngine._compute_order_flow_toxicity_z(
            volumes, taker_buy_volumes, closes, highs, lows
        )
        bayes_surv = Mark2IntelligenceEngine._compute_bayesian_grid_survival_prob(
            closes, highs, lows, atr_14, bb_upper, bb_lower
        )

        return Market3VTelemetryPOD(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            last_price=p_last,
            mark_price=mark_price if mark_price > 0.0 else p_last,
            bid_price=bid,
            ask_price=ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            quote_vol_24h=quote_vol_24h,
            atr_14=atr_14,
            natr_14_pct=round(natr_14_pct, 4),
            bb_upper=bb_upper,
            bb_mid=bb_mid,
            bb_lower=bb_lower,
            hurst_exp=round(hurst, 4),
            ema50_1h_slope_pct=round(ema_slope_pct, 4),
            quote_obi_l1=round(quote_obi, 4),
            cvd_norm_3=round(cvd_norm_3, 4),
            funding_rate=funding_rate,
            mae_q65_pct=round(mae_q65, 4),
            mfe_q85_pct=round(mfe_q85, 4),
            mfe_q92_pct=round(mfe_q92, 4),
            intraday_path_pct=round(intraday_path_pct, 4),
            ewma_vol_pct=round(ewma_vol_pct, 4),
            variance_ratio_q4=round(vr_q4, 4),
            flow_toxicity_z=round(tox_z, 4),
            bayesian_grid_survival_prob=bayes_surv,
        )

    @staticmethod
    def classify_grid_direction(tel: Market3VTelemetryPOD) -> GridDirection:
        """
        Determina la dirección del Grid (NEUTRAL, LONG o SHORT) fusionando
        indicadores microestructurales (EMA50, OBI L1, CVD Taker, Funding Rate)
        con el Test de Razón de Varianza VR(4):
        - Si VR(4) < 0.88 y la pendiente EMA50 es moderada (|slope| < 0.35%), prioriza
          NEUTRAL para cosechar reversión a la media pura en ambos sentidos sin sesgo direccional.
        """
        if tel.variance_ratio_q4 < 0.88 and abs(tel.ema50_1h_slope_pct) < 0.35:
            return GridDirection.NEUTRAL

        if (
            tel.ema50_1h_slope_pct > 0.0
            and (tel.quote_obi_l1 + tel.cvd_norm_3) > 0.0
            and tel.funding_rate <= 0.0010
        ):
            return GridDirection.LONG

        if (
            tel.ema50_1h_slope_pct < 0.0
            and (tel.quote_obi_l1 + tel.cvd_norm_3) < 0.0
            and tel.funding_rate >= -0.0010
        ):
            return GridDirection.SHORT

        return GridDirection.NEUTRAL
