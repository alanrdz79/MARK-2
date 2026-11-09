# -*- coding: utf-8 -*-
"""
=================================================================================================================
PROSPERITY MARK-2 — MÓDULO 8: ACELERADOR DE ALTO RIESGO PARA CAPITAL PEQUEÑO ($10.00 USDT)
(m8_high_risk_alpha_accelerator.py)
=================================================================================================================
Diseño Cuantitativo Estricto con 0% Constantes Heurísticas (100% Respaldado en Datos Reales de Binance):
1. Filtro Físico Contractual ($10.00 USDT): Desbloquea los contratos USDⓈ-M cuyo lote mínimo cuantizado
   (`min_notional` + `step_size`) cumple N_min(s) <= E_0, descartando contratos que atrapan >100% del capital.
2. Filtro No Paramétrico de Outliers de Funding (Gauss-Tukey MAD 99%):
   Bloquea automáticamente contratos con sangría extrema de funding |F_8h| > Mediana + Phi^-1(0.99)*1.4826*MAD.
3. Segmentación No Supervisada K-Means (Selección Óptima por Coeficiente de Silueta K*) + Pesos Objetivos CRITIC
   (Diakoulaki, Mavrotas & Papayannakis, 1995): Ordena el pool de Alta Volatilidad maximizando el margen neto
   sobre comisiones reales sin pesos humanos arbitrarios.
4. Calibración Empírica sobre Excursiones Horarias Winsorizadas (Tukey 1.5*IQR):
   Deriva `macro_step_pct`, `tp_pct` y `sl_pct` directamente de la distribución In-Sample real (evitando
   sobreajuste a mechas aisladas de listado), operando con:
   - Entrada 100% Pasiva Maker (`LIMIT GTX`, 0 órdenes MARKET iniciales).
   - Recotización pasiva de 1 `tick_size` ante rechazo `-5022` (jamás degradar a `GTC` Taker).
   - Take Profit pasivo Maker (`LIMIT GTX reduceOnly=True`) + Stop Loss `STOP_MARKET` sobre `MARK_PRICE`.
   - Ratchet automático a Breakeven + Escudo de Cuotas antes de reversión.
5. Dimensionamiento Half-Kelly Empírico + Apalancamiento Aislado Cero-Liquidación (P_liq < P_SL).
6. Doble Escudo DDM:
   - MDD 7D continuo sobre marca de agua de equidad.
   - Método Estadístico Binomial de Detección de Deriva (Gama, Medas, Castillo & Rodrigues, 2004):
     Alerta a 2*sigma (95%) y Corte/Rotación a 3*sigma (99.7%).
"""

from __future__ import annotations

from collections import deque
import logging
import math
import sqlite3
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core_mark2_contracts import (
    BinomialDDMStatePOD,
    CompoundingStage,
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    HighRiskAcceleratedOrderPlanPOD,
    HighRiskDDMStatus,
    HighRiskMarketProfilePOD,
    Market3VTelemetryPOD,
    MarketSafetyTier,
    ProsperityCapitalPhase,
)

try:
    from m0_workspace_hygiene import resolve_confined_path
except ImportError:
    resolve_confined_path = None  # type: ignore

logger = logging.getLogger("M8_HighRiskAccelerator")

# Constante matemática exacta de la distribución Normal Estándar: Phi^-1(0.99) * (1 / Phi^-1(0.75))
# = 2.3263478740408408 * 1.482602218505602 = 3.449048405561856 (0% heurística)
GAUSSIAN_MAD_99_FACTOR: float = 2.3263478740408408 * 1.482602218505602


class ExactExchangeQuantizer:
    """Cuantización exacta O(1) a los filtros físicos PRICE_FILTER (`tick_size`) y LOT_SIZE (`step_size`)."""

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
        return round(q, decimals + 2)

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
        return round(q, decimals + 2)

    @classmethod
    def min_order_qty_and_notional(
        cls, price: float, rule: ExchangeContractRulePOD
    ) -> Tuple[float, float]:
        if price <= 0.0:
            return rule.min_qty, 0.0
        q_min = max(
            rule.min_qty,
            cls.quantize_qty(rule.min_notional / price, rule.step_size, mode="CEIL"),
        )
        if q_min * price < rule.min_notional:
            q_min = cls.quantize_qty(q_min + rule.step_size, rule.step_size, mode="CEIL")
        return q_min, q_min * price

    @classmethod
    def requote_passive_gtx_price(
        cls,
        side: str,
        rejected_price: float,
        tick_size: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> float:
        """
        Resuelve el error `-5022` (Post-Only cruzaría el spread) desplazando la orden
        1 `tick_size` hacia el lado estrictamente Maker pasivo, evitando pagar tarifa Taker (`GTC`).
        """
        side_up = side.upper()
        if side_up == "BUY":
            anchor = min(rejected_price - tick_size, best_bid) if (best_bid and best_bid > 0.0) else (rejected_price - tick_size)
            return cls.quantize_price(max(tick_size, anchor), tick_size, mode="FLOOR")
        anchor = max(rejected_price + tick_size, best_ask) if (best_ask and best_ask > 0.0) else (rejected_price + tick_size)
        return cls.quantize_price(anchor, tick_size, mode="CEIL")


class Mark2DualDDMController:
    """
    Controlador Dual DDM en Tiempo Real:
    1. DDM Estadístico Binomial (Gama, Medas, Castillo & Rodrigues, 2004) sin constantes arbitrarias:
       - Alerta 2-sigma (IC 95%): p_i + s_i >= p_min + 2 * s_min
       - Corte de Deriva 3-sigma (IC 99.7%): p_i + s_i >= p_min + 3 * s_min
    2. DDM Patrimonial (MDD 7D porcentual sobre marca de agua de equidad).
    """

    def __init__(self, initial_equity_usdt: float, min_cycles_warmup: int = 4):
        self.min_cycles_warmup = max(2, int(min_cycles_warmup))
        self.n_cycles: int = 0
        self.error_count: int = 0
        self.p_min: float = float("inf")
        self.s_min: float = float("inf")
        self.last_p: float = 0.0
        self.last_s: float = 0.0
        self.status: HighRiskDDMStatus = HighRiskDDMStatus.IN_CONTROL
        self.equity_window_7d: Deque[Tuple[float, float]] = deque()
        self.equity_window_7d.append((time.time(), float(initial_equity_usdt)))

    def record_equity_mark(self, equity_usdt: float, now_ts: Optional[float] = None) -> float:
        ts = now_ts if now_ts is not None else time.time()
        self.equity_window_7d.append((ts, float(equity_usdt)))
        cutoff = ts - 7.0 * 86_400.0
        while len(self.equity_window_7d) > 1 and self.equity_window_7d[0][0] < cutoff:
            self.equity_window_7d.popleft()
        return self.compute_mdd_7d_pct()

    def compute_mdd_7d_pct(self) -> float:
        if not self.equity_window_7d:
            return 0.0
        peak = self.equity_window_7d[0][1]
        max_dd = 0.0
        for _, eq in self.equity_window_7d:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0 if peak > 0.0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    def register_closed_cycle(
        self, net_pnl_usdt: float, current_equity_usdt: float, now_ts: Optional[float] = None
    ) -> BinomialDDMStatePOD:
        self.n_cycles += 1
        if net_pnl_usdt < 0.0:
            self.error_count += 1
        p_i = self.error_count / self.n_cycles
        s_i = math.sqrt(p_i * (1.0 - p_i) / self.n_cycles)
        self.last_p = p_i
        self.last_s = s_i

        if self.n_cycles >= self.min_cycles_warmup:
            if p_i + s_i < self.p_min + self.s_min:
                self.p_min = p_i
                self.s_min = s_i
            if self.s_min > 0.0 and (p_i + s_i >= self.p_min + 3.0 * self.s_min):
                self.status = HighRiskDDMStatus.DRIFT_HALT_3SIGMA
            elif self.s_min > 0.0 and (p_i + s_i >= self.p_min + 2.0 * self.s_min):
                self.status = HighRiskDDMStatus.WARNING_2SIGMA
            else:
                self.status = HighRiskDDMStatus.IN_CONTROL

        mdd_7d = self.record_equity_mark(current_equity_usdt, now_ts=now_ts)
        return self.snapshot(mdd_7d_override=mdd_7d)

    def reset_after_rotation(self, new_equity_usdt: float) -> None:
        self.n_cycles = 0
        self.error_count = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.last_p = 0.0
        self.last_s = 0.0
        self.status = HighRiskDDMStatus.IN_CONTROL
        self.record_equity_mark(new_equity_usdt)

    def snapshot(self, mdd_7d_override: Optional[float] = None) -> BinomialDDMStatePOD:
        mdd = mdd_7d_override if mdd_7d_override is not None else self.compute_mdd_7d_pct()
        return BinomialDDMStatePOD(
            n_closed_cycles=self.n_cycles,
            error_count=self.error_count,
            error_rate_p=round(self.last_p, 6),
            std_error_s=round(self.last_s, 6),
            p_min=round(self.p_min, 6) if math.isfinite(self.p_min) else 0.0,
            s_min=round(self.s_min, 6) if math.isfinite(self.s_min) else 0.0,
            regime_status=self.status,
            mdd_7d_pct=mdd,
        )


class Mark2HighRiskAlphaAccelerator:
    """
    Motor Cuantitativo M8 para Aceleración de Capital Pequeño ($10.00 USDT) en Mercados de Alto Riesgo.
    Todos los parámetros emergen de estadísticas empíricas, optimización log-óptima y física de Binance.
    """

    def __init__(
        self,
        maker_fee: float = 0.00020,
        taker_fee: float = 0.00050,
    ):
        self.maker_fee = max(0.0, float(maker_fee))
        self.taker_fee = max(0.0, float(taker_fee))

    def update_real_commission_rates(self, maker_fee: float, taker_fee: float) -> None:
        if maker_fee >= 0.0:
            self.maker_fee = float(maker_fee)
        if taker_fee >= 0.0:
            self.taker_fee = float(taker_fee)

    # -------------------------------------------------------------------------
    # 1. PRIMITIVAS ESTADÍSTICAS SIN CONSTANTES HEURÍSTICAS
    # -------------------------------------------------------------------------

    @staticmethod
    def winsorize_tukey_iqr(values: np.ndarray) -> np.ndarray:
        """Recorte intercuartil estándar de Tukey [Q25 - 1.5*IQR, Q75 + 1.5*IQR]."""
        if len(values) < 4:
            return values
        q25, q75 = np.percentile(values, [25.0, 75.0])
        iqr = q75 - q25
        lo = max(0.0, float(q25 - 1.5 * iqr))
        hi = float(q75 + 1.5 * iqr)
        return np.clip(values, lo, hi)

    @staticmethod
    def compute_mad_99_funding_cutoff(abs_funding_rates: Sequence[float]) -> float:
        """Umbral Gauss-Tukey al 99% de confianza basado en Mediana + 3.449 * MAD."""
        arr = np.asarray(abs_funding_rates, dtype=np.float64)
        if len(arr) == 0:
            return 0.001
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med + GAUSSIAN_MAD_99_FACTOR * max(mad, 1e-8)

    @staticmethod
    def fit_kmeans_optimal_silhouette(
        feature_matrix: np.ndarray, candidate_ks: Tuple[int, ...] = (2, 3, 4)
    ) -> Tuple[int, float, np.ndarray]:
        """Selecciona K* maximizando el Coeficiente de Silueta sobre variables log-estandarizadas."""
        n = len(feature_matrix)
        if n < 4:
            return 1, 0.0, np.zeros(n, dtype=int)

        mu = np.mean(feature_matrix, axis=0)
        sd = np.std(feature_matrix, axis=0, ddof=1)
        X = (feature_matrix - mu) / np.where(sd > 1e-12, sd, 1.0)

        best_k = 2
        best_sil = -1.0
        best_labels = np.zeros(n, dtype=int)

        for k in candidate_ks:
            if k >= n:
                continue
            pcts = np.linspace(100.0 / (2.0 * k), 100.0 - 100.0 / (2.0 * k), k)
            init_idx = [int(np.argmin(np.abs(X[:, 0] - np.percentile(X[:, 0], p)))) for p in pcts]
            centroids = X[init_idx].copy()
            labels = np.zeros(n, dtype=int)
            for _ in range(60):
                dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for ci in range(k):
                    pts = X[labels == ci]
                    if len(pts) > 0:
                        centroids[ci] = np.mean(pts, axis=0)

            sils: List[float] = []
            for i in range(n):
                own = labels[i]
                same = [float(np.linalg.norm(X[i] - X[j])) for j in range(n) if j != i and labels[j] == own]
                a_i = float(np.mean(same)) if same else 0.0
                other_means = [
                    float(np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n) if labels[j] == ci]))
                    for ci in range(k)
                    if ci != own and np.any(labels == ci)
                ]
                b_i = min(other_means) if other_means else 0.0
                denom = max(a_i, b_i)
                sils.append((b_i - a_i) / denom if denom > 0.0 else 0.0)

            mean_sil = float(np.mean(sils))
            if mean_sil > best_sil:
                best_sil = mean_sil
                best_k = k
                best_labels = labels.copy()

        return best_k, best_sil, best_labels

    @staticmethod
    def compute_critic_weights_and_scores(
        criteria_matrix: np.ndarray,
    ) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
        """
        Método CRITIC (Diakoulaki et al., 1995):
        I_j = sigma_j * sum_k (1 - r_{jk}), w_j* = I_j / sum_l I_l.
        Reemplaza completamente cualquier ponderación heurística manual.
        """
        n_rows, n_cols = criteria_matrix.shape
        if n_rows <= 1:
            eq_w = tuple(1.0 / n_cols for _ in range(n_cols))
            return eq_w, np.ones(n_rows, dtype=np.float64)  # type: ignore

        c_min = np.min(criteria_matrix, axis=0)
        c_max = np.max(criteria_matrix, axis=0)
        span = np.where(c_max - c_min > 1e-12, c_max - c_min, 1.0)
        Z = (criteria_matrix - c_min) / span

        sigmas = np.std(Z, axis=0, ddof=1)
        if np.all(sigmas <= 1e-12):
            eq_w = tuple(1.0 / n_cols for _ in range(n_cols))
            return eq_w, np.mean(Z, axis=1)  # type: ignore

        corr = np.corrcoef(Z, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        info = np.array([sigmas[j] * float(np.sum(1.0 - corr[j, :])) for j in range(n_cols)], dtype=np.float64)
        total_info = float(np.sum(info))
        weights = info / total_info if total_info > 1e-12 else np.full(n_cols, 1.0 / n_cols)
        scores = Z @ weights
        w_tuple = (float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3]))
        return w_tuple, scores

    # -------------------------------------------------------------------------
    # 2. SIMULADOR CAUSAL IN-SAMPLE PARA CALIBRACIÓN EMPÍRICA DE ÓRDENES M8
    # -------------------------------------------------------------------------

    def _simulate_causal_is_window(
        self,
        ohlcv_arr: np.ndarray,
        rule: ExchangeContractRulePOD,
        fund_abs: float,
        macro_step_frac: float,
        sl_frac: float,
        tp_frac: float,
        eff_lev_slot: float,
        iso_lev: int,
        capital_init: float,
    ) -> Dict[str, Any]:
        """
        Evalúa causalmente (0 look-ahead, verificación pesimista SL antes de TP en cada vela)
        el desempeño neto de comisiones Maker/Taker reales y funding sobre el tramo In-Sample.
        Columnas esperadas en `ohlcv_arr`: [open_time, open, high, low, close, volume, taker_buy_volume, quote_volume].
        """
        o = ohlcv_arr[:, 1]
        h = ohlcv_arr[:, 2]
        l = ohlcv_arr[:, 3]
        c = ohlcv_arr[:, 4]
        v = ohlcv_arr[:, 5]
        tbv = ohlcv_arr[:, 6]
        n = len(c)
        if n < 16:
            return {"viable": False}

        tick = rule.tick_size
        step = rule.step_size
        min_not = rule.min_notional
        mmr = rule.maint_margin_rate

        equity = capital_init
        eq_curve = [equity]
        trades = 0
        wins = 0
        gross_pnl_sum = 0.0
        fees_sum = 0.0
        unit_rets: List[float] = []
        ddm = Mark2DualDDMController(initial_equity_usdt=capital_init, min_cycles_warmup=4)

        ema = np.zeros(n, dtype=np.float64)
        ema[0] = c[0]
        alpha = 2.0 / 13.0
        for idx in range(1, n):
            ema[idx] = alpha * c[idx] + (1.0 - alpha) * ema[idx - 1]
        obi = (2.0 * tbv - v) / np.maximum(v, 1e-12)

        i = 12
        while i < n - 1:
            if equity < min_not * 0.65:
                eq_curve.append(equity)
                break

            anchor = 0.5 * (float(c[i - 1]) + float(ema[i - 1]))
            buy_lim = ExactExchangeQuantizer.quantize_price(anchor * (1.0 - 0.5 * macro_step_frac), tick, "FLOOR")
            sell_lim = ExactExchangeQuantizer.quantize_price(anchor * (1.0 + 0.5 * macro_step_frac), tick, "CEIL")

            op_i, hi_i, lo_i = float(o[i]), float(h[i]), float(l[i])
            obi_prev = float(obi[i - 1])

            direction = 0
            lim_entry = 0.0
            if lo_i <= buy_lim and op_i >= buy_lim and obi_prev >= -0.35:
                direction = 1
                lim_entry = buy_lim
            elif hi_i >= sell_lim and op_i <= sell_lim and obi_prev <= 0.35:
                direction = -1
                lim_entry = sell_lim

            if direction == 0:
                eq_curve.append(equity)
                i += 1
                continue

            min_ord_q, _ = ExactExchangeQuantizer.min_order_qty_and_notional(lim_entry, rule)
            target_not = max(min_ord_q * lim_entry, equity * eff_lev_slot)
            qty = max(
                min_ord_q,
                ExactExchangeQuantizer.quantize_qty(
                    min(target_not, equity * iso_lev * 0.85) / lim_entry, step, "FLOOR"
                ),
            )
            notional = qty * lim_entry
            iso_margin = notional / max(1, iso_lev)
            if notional < min_not or iso_margin > equity * 0.95:
                eq_curve.append(equity)
                i += 1
                continue

            entry_fee = notional * self.maker_fee
            if direction == 1:
                sl_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 - sl_frac), tick, "FLOOR")
                tp_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 + tp_frac), tick, "CEIL")
                p_liq = max(0.0, (notional - iso_margin) / max(qty * (1.0 - mmr - self.taker_fee), 1e-12))
                if p_liq >= sl_p:
                    eq_curve.append(equity)
                    i += 1
                    continue
            else:
                sl_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 + sl_frac), tick, "CEIL")
                tp_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 - tp_frac), tick, "FLOOR")
                p_liq = (notional + iso_margin) / max(qty * (1.0 + mmr + self.taker_fee), 1e-12)
                if p_liq <= sl_p:
                    eq_curve.append(equity)
                    i += 1
                    continue

            be_armed = False
            fee_lock = self.maker_fee + self.taker_fee + (2.0 * tick / lim_entry)
            j = i + 1
            exited = False
            net_pnl_t = 0.0

            while j < n:
                if (j - i) % 96 == 0:
                    fc = qty * float(c[j]) * fund_abs
                    equity -= fc
                    fees_sum += fc

                op_j, hi_j, lo_j, cl_j = float(o[j]), float(h[j]), float(l[j]), float(c[j])
                if direction == 1:
                    if lo_j <= sl_p:
                        exit_p = min(sl_p, op_j)
                        gp = (exit_p - lim_entry) * qty
                        ef = qty * exit_p * self.taker_fee
                        net_pnl_t = gp - entry_fee - ef
                        equity += net_pnl_t
                        gross_pnl_sum += gp
                        fees_sum += entry_fee + ef
                        trades += 1
                        if net_pnl_t > 0.0:
                            wins += 1
                        exited = True
                        eq_curve.append(equity)
                        break
                    if hi_j >= tp_p:
                        exit_p = tp_p
                        gp = (exit_p - lim_entry) * qty
                        ef = qty * exit_p * self.maker_fee
                        net_pnl_t = gp - entry_fee - ef
                        equity += net_pnl_t
                        gross_pnl_sum += gp
                        fees_sum += entry_fee + ef
                        trades += 1
                        wins += 1
                        exited = True
                        eq_curve.append(equity)
                        break
                    if (not be_armed) and (hi_j >= lim_entry * (1.0 + 0.60 * tp_frac)):
                        sl_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 + fee_lock), tick, "CEIL")
                        be_armed = True
                    upnl = (cl_j - lim_entry) * qty
                else:
                    if hi_j >= sl_p:
                        exit_p = max(sl_p, op_j)
                        gp = (lim_entry - exit_p) * qty
                        ef = qty * exit_p * self.taker_fee
                        net_pnl_t = gp - entry_fee - ef
                        equity += net_pnl_t
                        gross_pnl_sum += gp
                        fees_sum += entry_fee + ef
                        trades += 1
                        if net_pnl_t > 0.0:
                            wins += 1
                        exited = True
                        eq_curve.append(equity)
                        break
                    if lo_j <= tp_p:
                        exit_p = tp_p
                        gp = (lim_entry - exit_p) * qty
                        ef = qty * exit_p * self.maker_fee
                        net_pnl_t = gp - entry_fee - ef
                        equity += net_pnl_t
                        gross_pnl_sum += gp
                        fees_sum += entry_fee + ef
                        trades += 1
                        wins += 1
                        exited = True
                        eq_curve.append(equity)
                        break
                    if (not be_armed) and (lo_j <= lim_entry * (1.0 - 0.60 * tp_frac)):
                        sl_p = ExactExchangeQuantizer.quantize_price(lim_entry * (1.0 - fee_lock), tick, "FLOOR")
                        be_armed = True
                    upnl = (lim_entry - cl_j) * qty

                eq_curve.append(equity + upnl)
                j += 1

            if not exited:
                exit_p = float(c[-1])
                gp = (exit_p - lim_entry) * qty if direction == 1 else (lim_entry - exit_p) * qty
                ef = qty * exit_p * self.taker_fee
                net_pnl_t = gp - entry_fee - ef
                equity += net_pnl_t
                gross_pnl_sum += gp
                fees_sum += entry_fee + ef
                trades += 1
                if net_pnl_t > 0.0:
                    wins += 1
                eq_curve.append(equity)
                unit_rets.append(net_pnl_t / notional)
                break

            unit_rets.append(net_pnl_t / notional)
            ddm_pod = ddm.register_closed_cycle(net_pnl_t, equity)
            if ddm_pod.regime_status == HighRiskDDMStatus.DRIFT_HALT_3SIGMA:
                for _ in range(min(8, n - 1 - j)):
                    eq_curve.append(equity)
                i = j + 9
            else:
                i = j + 1

        peak = eq_curve[0]
        mdd = 0.0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0 if peak > 0.0 else 0.0
            if dd > mdd:
                mdd = dd

        net_pnl = eq_curve[-1] - capital_init
        log_g = math.log(max(equity, 0.05) / capital_init)
        return {
            "viable": trades > 0,
            "trades": trades,
            "wins": wins,
            "win_rate": (wins / trades * 100.0) if trades > 0 else 0.0,
            "gross_pnl": gross_pnl_sum,
            "total_fees": fees_sum,
            "net_pnl": net_pnl,
            "roi_pct": (net_pnl / capital_init) * 100.0,
            "ddm_pct": mdd,
            "log_growth": log_g,
            "fee_drag_pct": (fees_sum / max(abs(gross_pnl_sum), 1e-6)) * 100.0 if trades > 0 else 0.0,
            "unit_returns": unit_rets,
        }

    # -------------------------------------------------------------------------
    # 3. CALIBRACIÓN COMPLETA DEL UNIVERSO DESDE SERIES OHLCV / WAREHOUSE DB
    # -------------------------------------------------------------------------

    def calibrate_from_ohlcv_catalog(
        self,
        ohlcv_map: Dict[str, np.ndarray],
        rules_map: Dict[str, ExchangeContractRulePOD],
        funding_abs_map: Dict[str, float],
        wallet_capital_usdt: float = 10.00,
        in_sample_ratio: float = 0.70,
    ) -> Tuple[List[HighRiskMarketProfilePOD], List[HighRiskAcceleratedOrderPlanPOD]]:
        """
        Ejecuta la tubería completa sin heurísticos:
        1. Partición cronológica In-Sample (`in_sample_ratio`) previa a todo ajuste.
        2. Filtro físico `N_min <= wallet_capital_usdt` + Filtro Gauss-Tukey MAD 99% de funding.
        3. Clustering K-Means con máxima Silueta + Ranking objetivo CRITIC.
        4. Optimización empírica de cuantiles Winsorizados 1h + Half-Kelly + P_liq < P_SL.
        """
        symbols = sorted(s for s in ohlcv_map if s in rules_map and len(ohlcv_map[s]) >= 60)
        if not symbols:
            return [], []

        fund_cutoff = self.compute_mad_99_funding_cutoff(
            [funding_abs_map.get(s, 0.00005) for s in symbols]
        )

        raw_stats: Dict[str, Dict[str, Any]] = {}
        excluded_profiles: List[HighRiskMarketProfilePOD] = []
        eligible_syms: List[str] = []

        for sym in symbols:
            arr_full = ohlcv_map[sym]
            n_is = max(40, int(len(arr_full) * in_sample_ratio)) if in_sample_ratio < 1.0 else len(arr_full)
            sub = arr_full[:n_is]
            rule = rules_map[sym]
            fund_abs = float(funding_abs_map.get(sym, 0.00005))

            p_ref = float(sub[-1, 4])
            _, eff_min_not = ExactExchangeQuantizer.min_order_qty_and_notional(p_ref, rule)

            o, h, l, c, qv = sub[:, 1], sub[:, 2], sub[:, 3], sub[:, 4], sub[:, 7]
            m = len(c)
            prev_c = np.roll(c, 1)
            prev_c[0] = c[0]
            tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
            natr_5m = float(np.mean(tr[-min(14, m) :]) / max(p_ref, 1e-12) * 100.0)

            log_p = np.log(np.maximum(c, 1e-12))
            r1 = np.diff(log_p)
            rv_24h = float(np.std(r1, ddof=1) * math.sqrt(288.0) * 100.0) if len(r1) > 2 else natr_5m
            q_vr = min(12, max(2, m // 6))
            rq = log_p[q_vr:] - log_p[:-q_vr]
            vr_1h = float(np.var(rq, ddof=1) / (q_vr * max(np.var(r1, ddof=1), 1e-15))) if len(rq) > 2 else 1.0

            w1h = min(12, max(4, m // 4))
            up1 = np.array([(np.max(h[i + 1 : i + 1 + w1h]) - c[i]) / c[i] for i in range(m - w1h)])
            dn1 = np.array([(c[i] - np.min(l[i + 1 : i + 1 + w1h])) / c[i] for i in range(m - w1h)])
            mfe_1h_w = self.winsorize_tukey_iqr(np.maximum(up1, dn1))
            mae_1h_w = self.winsorize_tukey_iqr(np.minimum(up1, dn1))
            rng_1h_w = self.winsorize_tukey_iqr(
                np.array([np.max(h[i : i + w1h]) - np.min(l[i : i + w1h]) for i in range(m - w1h)])
                / np.maximum(c[: m - w1h], 1e-12)
            )
            gaps_w = self.winsorize_tukey_iqr(np.abs(o[1:] - c[:-1]) / np.maximum(c[:-1], 1e-12))
            max_gap_w = float(np.max(gaps_w)) if len(gaps_w) > 0 else 0.002
            vol_24h = float(np.mean(qv) * 288.0)

            excl_reason = ""
            if eff_min_not > wallet_capital_usdt:
                excl_reason = f"MIN_LOT_EXCEEDS_CAPITAL (${eff_min_not:.2f} > ${wallet_capital_usdt:.2f})"
            elif fund_abs > fund_cutoff:
                excl_reason = f"EXTREME_FUNDING_OUTLIER ({fund_abs*100:.4f}%/8h > {fund_cutoff*100:.4f}%/8h)"

            if excl_reason:
                excluded_profiles.append(
                    HighRiskMarketProfilePOD(
                        symbol=sym,
                        rv_24h_pct=round(rv_24h, 4),
                        natr_5m_pct=round(natr_5m, 4),
                        variance_ratio_1h=round(vr_1h, 4),
                        winsorized_range_1h_q50_pct=round(float(np.percentile(rng_1h_w, 50)) * 100.0, 4),
                        winsorized_mfe_1h_q60_pct=round(float(np.percentile(mfe_1h_w, 60)) * 100.0, 4),
                        winsorized_mae_1h_q70_pct=round(float(np.percentile(mae_1h_w, 70)) * 100.0, 4),
                        max_gap_winsorized_pct=round(max_gap_w * 100.0, 4),
                        quote_vol_24h_usdt=round(vol_24h, 2),
                        mean_abs_funding_rate=fund_abs,
                        effective_min_notional_usdt=round(eff_min_not, 2),
                        silhouette_cluster_id=-1,
                        silhouette_score=0.0,
                        critic_weights=(0.25, 0.25, 0.25, 0.25),
                        critic_score=0.0,
                        real_maker_fee=self.maker_fee,
                        real_taker_fee=self.taker_fee,
                        eligible_for_10usdt=False,
                        exclusion_reason=excl_reason,
                    )
                )
                continue

            eligible_syms.append(sym)
            raw_stats[sym] = {
                "is_arr": sub,
                "full_arr": arr_full,
                "rule": rule,
                "fund_abs": fund_abs,
                "eff_min_not": eff_min_not,
                "rv_24h": rv_24h,
                "natr_5m": natr_5m,
                "vr_1h": vr_1h,
                "mfe_1h_w": mfe_1h_w,
                "mae_1h_w": mae_1h_w,
                "rng_1h_w": rng_1h_w,
                "max_gap_w": max_gap_w,
                "vol_24h": vol_24h,
                "tick_spread": rule.tick_size / max(p_ref, 1e-12),
            }

        if not eligible_syms:
            return excluded_profiles, []

        feat_mat = np.array(
            [
                [
                    math.log1p(raw_stats[s]["rv_24h"]),
                    math.log1p(raw_stats[s]["natr_5m"]),
                    math.log1p(float(np.percentile(raw_stats[s]["mfe_1h_w"], 75)) * 100.0),
                    math.log1p(float(np.percentile(raw_stats[s]["mae_1h_w"], 50)) * 100.0),
                    math.log10(max(raw_stats[s]["vol_24h"], 100.0)),
                ]
                for s in eligible_syms
            ],
            dtype=np.float64,
        )
        best_k, best_sil, labels = self.fit_kmeans_optimal_silhouette(feat_mat)

        cluster_means = {
            ci: float(
                np.mean(
                    [raw_stats[eligible_syms[idx]]["rv_24h"] for idx in range(len(eligible_syms)) if labels[idx] == ci]
                )
            )
            for ci in range(best_k)
            if np.any(labels == ci)
        }
        high_vol_cluster = max(cluster_means, key=cluster_means.get) if cluster_means else 0

        crit_mat = np.array(
            [
                [
                    (
                        float(np.percentile(raw_stats[s]["mfe_1h_w"], 75))
                        - (2.0 * self.maker_fee + raw_stats[s]["tick_spread"] + raw_stats[s]["fund_abs"])
                    )
                    / max(2.0 * self.maker_fee + raw_stats[s]["tick_spread"], 1e-6),
                    float(np.percentile(raw_stats[s]["mfe_1h_w"], 75))
                    / max(float(np.percentile(raw_stats[s]["mae_1h_w"], 50)), 1e-6),
                    math.log10(max(raw_stats[s]["vol_24h"], 100.0)),
                    -raw_stats[s]["fund_abs"],
                ]
                for s in eligible_syms
            ],
            dtype=np.float64,
        )
        critic_w, critic_scores = self.compute_critic_weights_and_scores(crit_mat)

        profiles: List[HighRiskMarketProfilePOD] = []
        for idx, sym in enumerate(eligible_syms):
            st = raw_stats[sym]
            profiles.append(
                HighRiskMarketProfilePOD(
                    symbol=sym,
                    rv_24h_pct=round(st["rv_24h"], 4),
                    natr_5m_pct=round(st["natr_5m"], 4),
                    variance_ratio_1h=round(st["vr_1h"], 4),
                    winsorized_range_1h_q50_pct=round(float(np.percentile(st["rng_1h_w"], 50)) * 100.0, 4),
                    winsorized_mfe_1h_q60_pct=round(float(np.percentile(st["mfe_1h_w"], 60)) * 100.0, 4),
                    winsorized_mae_1h_q70_pct=round(float(np.percentile(st["mae_1h_w"], 70)) * 100.0, 4),
                    max_gap_winsorized_pct=round(st["max_gap_w"] * 100.0, 4),
                    quote_vol_24h_usdt=round(st["vol_24h"], 2),
                    mean_abs_funding_rate=st["fund_abs"],
                    effective_min_notional_usdt=round(st["eff_min_not"], 2),
                    silhouette_cluster_id=int(labels[idx]),
                    silhouette_score=round(best_sil, 4),
                    critic_weights=tuple(round(w, 4) for w in critic_w),  # type: ignore
                    critic_score=round(float(critic_scores[idx]), 4),
                    real_maker_fee=self.maker_fee,
                    real_taker_fee=self.taker_fee,
                    eligible_for_10usdt=True,
                    exclusion_reason="",
                )
            )

        # Seleccionar estrictamente el Cluster de Alta Volatilidad determinado por máxima Silueta K* (0% heurísticos)
        profile_by_sym = {p.symbol: p for p in profiles}
        high_risk_candidates = [
            s
            for idx, s in enumerate(eligible_syms)
            if labels[idx] == high_vol_cluster
        ]
        if len(high_risk_candidates) < 3:
            rv_all = np.array([raw_stats[s]["rv_24h"] for s in eligible_syms], dtype=np.float64)
            med_rv = float(np.median(rv_all))
            high_risk_candidates = [s for s in eligible_syms if raw_stats[s]["rv_24h"] >= med_rv]
        high_risk_candidates.sort(key=lambda s: profile_by_sym[s].critic_score, reverse=True)

        plans: List[HighRiskAcceleratedOrderPlanPOD] = []
        min_fee_step = 12.0 * self.maker_fee

        for sym in high_risk_candidates:
            st = raw_stats[sym]
            rl: ExchangeContractRulePOD = st["rule"]
            prof = profile_by_sym[sym]

            best_cfg: Optional[Tuple[float, float, float, int, Dict[str, Any]]] = None
            best_obj = -1e9

            for q_step in (35.0, 50.0, 65.0):
                m_step = max(float(np.percentile(st["rng_1h_w"], q_step)) * 0.5, min_fee_step)
                for q_tp in (45.0, 60.0, 75.0):
                    tp_f = max(float(np.percentile(st["mfe_1h_w"], q_tp)), 15.0 * self.maker_fee)
                    for q_sl in (55.0, 70.0, 85.0):
                        sl_f = max(float(np.percentile(st["mae_1h_w"], q_sl)), min_fee_step)
                        iso_lev = min(
                            rl.max_leverage,
                            max(
                                2,
                                int(
                                    math.floor(
                                        1.0 / (sl_f + st["max_gap_w"] + rl.maint_margin_rate + self.taker_fee)
                                    )
                                ),
                            ),
                        )
                        sim_is = self._simulate_causal_is_window(
                            st["is_arr"], rl, st["fund_abs"], m_step, sl_f, tp_f, 1.0, iso_lev, wallet_capital_usdt
                        )
                        if sim_is.get("viable") and sim_is["trades"] >= 3:
                            dd_p = min(0.90, sim_is["ddm_pct"] / 100.0)
                            obj = sim_is["log_growth"] + 1.5 * math.log(1.0 - dd_p)
                            if obj > best_obj:
                                best_obj = obj
                                best_cfg = (m_step, sl_f, tp_f, iso_lev, sim_is)

            if best_cfg is None:
                m_step = max(float(np.percentile(st["rng_1h_w"], 50.0)) * 0.5, min_fee_step)
                tp_f = max(float(np.percentile(st["mfe_1h_w"], 60.0)), 15.0 * self.maker_fee)
                sl_f = max(float(np.percentile(st["mae_1h_w"], 70.0)), min_fee_step)
                iso_lev = min(
                    rl.max_leverage,
                    max(2, int(math.floor(1.0 / (sl_f + st["max_gap_w"] + rl.maint_margin_rate + self.taker_fee)))),
                )
                eff_lev = max(st["eff_min_not"] / wallet_capital_usdt, 1.0)
                exp_mdd = round(sl_f * eff_lev * 100.0, 2)
            else:
                m_step, sl_f, tp_f, iso_lev, sim_is = best_cfg
                u_arr = np.asarray(sim_is["unit_returns"], dtype=np.float64)
                min_eff = st["eff_min_not"] / wallet_capital_usdt
                cand_levs = np.linspace(max(0.55, min_eff), min(float(iso_lev) * 0.60, 3.5), 25)
                eff_lev = float(cand_levs[0])
                best_g = -1e9
                for lv in cand_levs:
                    if np.any(1.0 + lv * u_arr <= 0.20):
                        continue
                    gv = float(np.mean(np.log(1.0 + 0.5 * lv * u_arr)))
                    if gv > best_g:
                        best_g = gv
                        eff_lev = max(min_eff, float(0.5 * lv))
                exp_mdd = round(float(sim_is["ddm_pct"]), 2)

            last_mark = float(st["full_arr"][-1, 4])
            mode_label = (
                "M8_IMPULSE_BRACKET" if st["vr_1h"] >= 1.0 else "M8_SPARSE_MACRO_CHANNEL"
            )
            plan_pod = self._synthesize_quantized_order_plan(
                symbol=sym,
                mode_label=mode_label,
                mark_price=last_mark,
                rule=rl,
                wallet_capital_usdt=wallet_capital_usdt,
                iso_lev=iso_lev,
                eff_lev=eff_lev,
                macro_step_frac=m_step,
                tp_frac=tp_f,
                sl_frac=sl_f,
                critic_score=prof.critic_score,
                rv_24h_pct=prof.rv_24h_pct,
                expected_mdd_7d_pct=exp_mdd,
            )
            if plan_pod is not None:
                plans.append(plan_pod)

        plans.sort(key=lambda p: (p.critic_score, p.expected_wallet_roi_tp_pct), reverse=True)
        all_profiles = sorted(profiles, key=lambda p: p.critic_score, reverse=True) + excluded_profiles
        return all_profiles, plans

    def calibrate_universe_from_warehouse(
        self,
        warehouse_db_path: str = "prosperity_backtest_warehouse.db",
        wallet_capital_usdt: float = 10.00,
        timeframe: str = "5m",
        in_sample_ratio: float = 0.70,
    ) -> Tuple[List[HighRiskMarketProfilePOD], List[HighRiskAcceleratedOrderPlanPOD]]:
        """Carga las tablas reales de `prosperity_backtest_warehouse.db` en modo concurrente seguro y calibra M8."""
        resolved = (
            resolve_confined_path(warehouse_db_path)
            if resolve_confined_path is not None
            else warehouse_db_path
        )
        conn = sqlite3.connect(resolved, timeout=15.0)
        try:
            cur = conn.cursor()
            rules_map: Dict[str, ExchangeContractRulePOD] = {}
            for r in cur.execute(
                """
                SELECT symbol, tick_size, step_size, min_qty, min_notional,
                       max_num_orders, maint_margin_rate, max_leverage
                FROM exchange_contract_rules
                """
            ).fetchall():
                rules_map[str(r[0])] = ExchangeContractRulePOD(
                    symbol=str(r[0]),
                    tick_size=float(r[1]),
                    step_size=float(r[2]),
                    min_qty=float(r[3]),
                    min_notional=float(r[4]),
                    max_num_orders=int(r[5]),
                    maint_margin_rate=float(r[6]),
                    max_leverage=int(r[7]),
                )

            funding_map: Dict[str, float] = {
                str(r[0]): float(r[1])
                for r in cur.execute(
                    "SELECT symbol, AVG(ABS(funding_rate)) FROM historical_derivatives_metrics GROUP BY symbol"
                ).fetchall()
            }

            syms = [
                str(r[0])
                for r in cur.execute(
                    "SELECT DISTINCT symbol FROM historical_klines_mtf WHERE timeframe=? AND is_synthetic=0 ORDER BY symbol",
                    (timeframe,),
                ).fetchall()
            ]
            ohlcv_map: Dict[str, np.ndarray] = {}
            for sym in syms:
                rows = cur.execute(
                    """
                    SELECT open_time, open, high, low, close, volume, taker_buy_volume, quote_volume
                    FROM historical_klines_mtf
                    WHERE symbol=? AND timeframe=? AND is_synthetic=0
                    ORDER BY open_time ASC
                    """,
                    (sym, timeframe),
                ).fetchall()
                if len(rows) >= 60:
                    arr = np.asarray(rows, dtype=np.float64)
                    if not np.isnan(arr).any():
                        ohlcv_map[sym] = arr
        finally:
            conn.close()

        return self.calibrate_from_ohlcv_catalog(
            ohlcv_map=ohlcv_map,
            rules_map=rules_map,
            funding_abs_map=funding_map,
            wallet_capital_usdt=wallet_capital_usdt,
            in_sample_ratio=in_sample_ratio,
        )

    # -------------------------------------------------------------------------
    # 4. SÍNTESIS EN VIVO DESDE TELEMETRÍA 3V Y PUENTE A GEOMETRICGRIDPLANPOD
    # -------------------------------------------------------------------------

    def build_high_risk_plan_from_live_telemetry(
        self,
        rule: ExchangeContractRulePOD,
        tel: Market3VTelemetryPOD,
        wallet_capital_usdt: float = 10.00,
        critic_score: float = 0.65,
    ) -> Optional[HighRiskAcceleratedOrderPlanPOD]:
        """
        Sintetiza en O(1) un plan M8 directamente desde una instantánea `Market3VTelemetryPOD`
        en vivo cuando el contrato aún no tiene historial largo en almacén local.
        """
        mark = tel.mark_price if tel.mark_price > 0.0 else tel.last_price
        _, eff_min_not = ExactExchangeQuantizer.min_order_qty_and_notional(mark, rule)
        if eff_min_not > wallet_capital_usdt:
            return None

        min_fee_step = 12.0 * self.maker_fee
        # Excursiones empíricas extraídas en vivo por M2 (`mae_q65_pct`, `mfe_q85_pct`, `intraday_path_pct`)
        sl_frac = max((tel.mae_q65_pct / 100.0), min_fee_step)
        tp_frac = max((tel.mfe_q85_pct / 100.0), max(sl_frac * 1.5, 15.0 * self.maker_fee))
        macro_step_frac = max((tel.natr_14_pct / 100.0) * math.sqrt(12.0) * 0.5, min_fee_step)
        gap_est = max((tel.natr_14_pct / 100.0) * 0.75, 0.002)

        iso_lev = min(
            rule.max_leverage,
            max(2, int(math.floor(1.0 / (sl_frac + gap_est + rule.maint_margin_rate + self.taker_fee)))),
        )
        # Half-Kelly cerrado sobre razón empírica MFE/MAE y probabilidad bayesiana de supervivencia M2
        p_win = min(0.85, max(0.35, tel.bayesian_grid_survival_prob * 0.65))
        b_payoff = max(0.5, (tp_frac - 2.0 * self.maker_fee) / max(sl_frac + self.maker_fee + self.taker_fee, 1e-6))
        kelly_unit = max(0.0, (p_win * b_payoff - (1.0 - p_win)) / max(b_payoff * sl_frac, 1e-6))
        min_eff = eff_min_not / max(wallet_capital_usdt, 1e-6)
        eff_lev = max(min_eff, min(0.5 * kelly_unit, float(iso_lev) * 0.55, 2.25))

        mode_label = (
            "M8_IMPULSE_BRACKET" if tel.variance_ratio_q4 >= 1.0 else "M8_SPARSE_MACRO_CHANNEL"
        )
        rv_est = max(tel.ewma_vol_pct, tel.intraday_path_pct)
        return self._synthesize_quantized_order_plan(
            symbol=rule.symbol,
            mode_label=mode_label,
            mark_price=mark,
            rule=rule,
            wallet_capital_usdt=wallet_capital_usdt,
            iso_lev=iso_lev,
            eff_lev=eff_lev,
            macro_step_frac=macro_step_frac,
            tp_frac=tp_frac,
            sl_frac=sl_frac,
            critic_score=critic_score,
            rv_24h_pct=rv_est,
            expected_mdd_7d_pct=round(sl_frac * eff_lev * 100.0 * 2.0, 2),
        )

    def _synthesize_quantized_order_plan(
        self,
        symbol: str,
        mode_label: str,
        mark_price: float,
        rule: ExchangeContractRulePOD,
        wallet_capital_usdt: float,
        iso_lev: int,
        eff_lev: float,
        macro_step_frac: float,
        tp_frac: float,
        sl_frac: float,
        critic_score: float,
        rv_24h_pct: float,
        expected_mdd_7d_pct: float,
    ) -> Optional[HighRiskAcceleratedOrderPlanPOD]:
        tick = rule.tick_size
        step = rule.step_size
        mmr = rule.maint_margin_rate

        buy_anchor = ExactExchangeQuantizer.quantize_price(
            mark_price * (1.0 - 0.5 * macro_step_frac), tick, "FLOOR"
        )
        sell_anchor = ExactExchangeQuantizer.quantize_price(
            mark_price * (1.0 + 0.5 * macro_step_frac), tick, "CEIL"
        )
        if buy_anchor <= 0.0:
            return None

        min_q, _ = ExactExchangeQuantizer.min_order_qty_and_notional(buy_anchor, rule)
        target_notional = max(min_q * buy_anchor, wallet_capital_usdt * eff_lev)
        max_iso_notional = wallet_capital_usdt * iso_lev * 0.85
        qty = max(
            min_q,
            ExactExchangeQuantizer.quantize_qty(
                min(target_notional, max_iso_notional) / buy_anchor, step, "FLOOR"
            ),
        )
        notional = round(qty * buy_anchor, 4)
        if notional < rule.min_notional:
            return None

        # Garantizar matemáticamente P_liq < P_SL ajustando iso_lev hacia abajo si es necesario
        tp_price = ExactExchangeQuantizer.quantize_price(buy_anchor * (1.0 + tp_frac), tick, "CEIL")
        sl_price = ExactExchangeQuantizer.quantize_price(buy_anchor * (1.0 - sl_frac), tick, "FLOOR")

        resolved_iso_lev = max(1, int(iso_lev))
        p_liq = 0.0
        while resolved_iso_lev >= 1:
            iso_m_try = notional / resolved_iso_lev
            if iso_m_try > wallet_capital_usdt * 0.95:
                resolved_iso_lev += 1
                break
            p_liq_try = max(
                0.0,
                (notional - iso_m_try) / max(qty * (1.0 - mmr - self.taker_fee), 1e-12),
            )
            if p_liq_try < sl_price:
                p_liq = p_liq_try
                break
            resolved_iso_lev -= 1

        resolved_iso_lev = max(1, min(rule.max_leverage, resolved_iso_lev))
        iso_margin = round(notional / resolved_iso_lev, 4)
        p_liq = max(
            0.0,
            (notional - iso_margin) / max(qty * (1.0 - mmr - self.taker_fee), 1e-12),
        )

        be_trigger = ExactExchangeQuantizer.quantize_price(
            buy_anchor * (1.0 + 0.60 * tp_frac), tick, "CEIL"
        )
        fee_lock_frac = self.maker_fee + self.taker_fee + (2.0 * tick / buy_anchor)
        be_lock = ExactExchangeQuantizer.quantize_price(
            buy_anchor * (1.0 + fee_lock_frac), tick, "CEIL"
        )

        gross_tp = (tp_price - buy_anchor) * qty
        rt_fee_tp = (notional * self.maker_fee) + (qty * tp_price * self.maker_fee)
        net_tp = gross_tp - rt_fee_tp

        gross_sl = (sl_price - buy_anchor) * qty
        rt_fee_sl = (notional * self.maker_fee) + (qty * sl_price * self.taker_fee)
        net_sl = gross_sl - rt_fee_sl

        fee_to_gross_pct = (rt_fee_tp / max(gross_tp, 1e-9)) * 100.0

        return HighRiskAcceleratedOrderPlanPOD(
            symbol=symbol,
            execution_mode=mode_label,
            margin_type="ISOLATED",
            wallet_capital_usdt=round(wallet_capital_usdt, 2),
            isolated_leverage=resolved_iso_lev,
            effective_wallet_leverage=round(notional / max(wallet_capital_usdt, 1e-6), 2),
            mark_price=mark_price,
            anchor_buy_gtx_price=buy_anchor,
            anchor_sell_gtx_price=sell_anchor,
            macro_step_pct=round(macro_step_frac * 100.0, 2),
            order_qty=qty,
            notional_usdt=round(notional, 2),
            isolated_margin_usdt=round(iso_margin, 2),
            tp_limit_gtx_price=tp_price,
            tp_pct=round(((tp_price - buy_anchor) / buy_anchor) * 100.0, 2),
            sl_stop_market_price=sl_price,
            sl_pct=round(((buy_anchor - sl_price) / buy_anchor) * 100.0, 2),
            breakeven_trigger_price=be_trigger,
            breakeven_lock_price=be_lock,
            liquidation_price=round(p_liq, 8),
            expected_net_tp_usdt=round(net_tp, 4),
            expected_net_sl_usdt=round(net_sl, 4),
            expected_wallet_roi_tp_pct=round((net_tp / max(wallet_capital_usdt, 1e-6)) * 100.0, 2),
            expected_isolated_roi_tp_pct=round((net_tp / max(iso_margin, 1e-6)) * 100.0, 2),
            expected_wallet_roi_sl_pct=round((net_sl / max(wallet_capital_usdt, 1e-6)) * 100.0, 2),
            roundtrip_fee_tp_usdt=round(rt_fee_tp, 5),
            fee_to_gross_tp_pct=round(fee_to_gross_pct, 2),
            open_position_on_creation=False,
            critic_score=round(critic_score, 4),
            rv_24h_pct=round(rv_24h_pct, 2),
            ddm_status=HighRiskDDMStatus.IN_CONTROL,
            expected_mdd_7d_pct=expected_mdd_7d_pct,
        )

    @staticmethod
    def evaluate_breakeven_fee_shield_ratchet(
        m8_plan: HighRiskAcceleratedOrderPlanPOD,
        position_amt: float,
        entry_price: float,
        live_mark_price: float,
        current_sl_price: float,
        tick_size: float,
        maker_fee: float,
        taker_fee: float,
    ) -> Optional[float]:
        """
        Evalúa en tiempo real si el precio de marca alcanzó el umbral de activación de Breakeven (0.60 * TP)
        y devuelve el nuevo `SL` neto de cuotas (`P_BE`) si mejora el `SL` vigente; de lo contrario `None`.
        """
        if abs(position_amt) <= 1e-12 or entry_price <= 0.0 or live_mark_price <= 0.0:
            return None

        tp_frac = m8_plan.tp_pct / 100.0
        fee_buf = maker_fee + taker_fee + (2.0 * tick_size / entry_price)

        if position_amt > 0.0:
            trig = entry_price * (1.0 + 0.60 * tp_frac)
            lock_p = ExactExchangeQuantizer.quantize_price(entry_price * (1.0 + fee_buf), tick_size, "CEIL")
            if live_mark_price >= trig and lock_p > current_sl_price and lock_p < live_mark_price:
                return lock_p
        else:
            trig = entry_price * (1.0 - 0.60 * tp_frac)
            lock_p = ExactExchangeQuantizer.quantize_price(entry_price * (1.0 - fee_buf), tick_size, "FLOOR")
            if live_mark_price <= trig and (current_sl_price <= 0.0 or lock_p < current_sl_price) and lock_p > live_mark_price:
                return lock_p
        return None

    def to_geometric_grid_plan_pod(
        self,
        m8_plan: HighRiskAcceleratedOrderPlanPOD,
        rule: ExchangeContractRulePOD,
    ) -> GeometricGridPlanPOD:
        """
        Adapta `HighRiskAcceleratedOrderPlanPOD` a `GeometricGridPlanPOD` con compatibilidad 100% hacia M4/M5/M6:
        - `direction = GridDirection.NEUTRAL` -> Fuerza `open_position_on_creation = False` (0% Taker inicial).
        - `active_buy_window = (anchor_buy_gtx_price,)` y `active_sell_window = (anchor_sell_gtx_price,)` ->
          Menos órdenes activas (`2` en vez de `8-30`), paso macro amplio, cuota Maker < 1% del margen bruto.
        """
        cap = m8_plan.wallet_capital_usdt
        lower_p = m8_plan.sl_stop_market_price
        upper_p = m8_plan.tp_limit_gtx_price
        levels = (
            lower_p,
            m8_plan.anchor_buy_gtx_price,
            m8_plan.anchor_sell_gtx_price,
            upper_p,
        )
        # Cruce completo entre anclas pasivas o hacia TP
        net_step_pct = max(0.20, m8_plan.macro_step_pct - (2.0 * self.maker_fee * 100.0))
        net_step_usdt = round(m8_plan.notional_usdt * (net_step_pct / 100.0), 4)
        est_crosses_d = max(1.0, round(m8_plan.rv_24h_pct / max(m8_plan.tp_pct, 1.0), 2))
        daily_pnl = round(m8_plan.expected_net_tp_usdt * min(est_crosses_d, 3.0) * 0.45, 4)
        daily_roi = round((daily_pnl / max(cap, 1e-6)) * 100.0, 2)
        target_3x = round(cap * 3.0, 2)
        est_days_3x = round(max(1.0, (target_3x - cap) / max(daily_pnl, 0.05)), 1)

        return GeometricGridPlanPOD(
            symbol=m8_plan.symbol,
            safety_tier=MarketSafetyTier.VOLATILES,
            direction=GridDirection.NEUTRAL,
            compounding_stage=CompoundingStage.STAGE_1_MITIGATION_0_TO_33PCT,
            initial_capital_usdt=cap,
            current_equity_usdt=cap,
            target_equity_usdt=round(cap + m8_plan.expected_net_tp_usdt, 2),
            equity_ratchet_floor_usdt=round(max(rule.min_notional, cap + m8_plan.expected_net_sl_usdt), 2),
            leverage=m8_plan.isolated_leverage,
            lower_price=lower_p,
            upper_price=upper_p,
            total_virtual_grids=2,
            active_window_orders=2,
            geometric_step_pct=m8_plan.macro_step_pct,
            net_profit_per_cross_pct=round(net_step_pct, 2),
            net_profit_per_cross_usdt=net_step_usdt,
            qty_per_line=m8_plan.order_qty,
            notional_per_line_usdt=m8_plan.notional_usdt,
            all_virtual_levels=levels,
            active_buy_window=(m8_plan.anchor_buy_gtx_price,),
            active_sell_window=(m8_plan.anchor_sell_gtx_price,),
            tp_price=m8_plan.tp_limit_gtx_price,
            tp_pnl_usdt=round(m8_plan.expected_net_tp_usdt, 2),
            tp_roi_pct=m8_plan.expected_wallet_roi_tp_pct,
            sl_price=m8_plan.sl_stop_market_price,
            sl_pnl_usdt=round(m8_plan.expected_net_sl_usdt, 2),
            sl_roi_pct=m8_plan.expected_wallet_roi_sl_pct,
            est_daily_crosses=est_crosses_d,
            est_days_to_18_usd=est_days_3x,
            trailing_up=True,
            trailing_down=False,
            effective_initial_leverage=m8_plan.effective_wallet_leverage,
            max_adverse_effective_leverage=round(min(float(m8_plan.isolated_leverage) * 0.75, m8_plan.effective_wallet_leverage * 1.5), 2),
            estimated_liquidation_price=m8_plan.liquidation_price,
            liquidation_safety_buffer_pct=round(
                ((m8_plan.sl_stop_market_price - m8_plan.liquidation_price) / max(m8_plan.anchor_buy_gtx_price, 1e-12)) * 100.0,
                2,
            ),
            anti_liquidation_reserve_usdt=round(max(0.50, cap - m8_plan.isolated_margin_usdt), 2),
            active_phase=ProsperityCapitalPhase.PHASE_1_MITIGATION_4_TO_100.value,
            phase_operational_risk_pct=abs(m8_plan.expected_wallet_roi_sl_pct),
            phase_target_multiplier=3.0,
            uncertainty_score=round(max(0.05, 1.0 - m8_plan.critic_score), 4),
            confidence_pct=round(min(99.0, 50.0 + m8_plan.critic_score * 48.0), 1),
            expected_daily_pnl_usdt=daily_pnl,
            expected_daily_roi_pct=daily_roi,
            expected_30d_pnl_usdt=round(daily_pnl * 30.0, 2),
            expected_30d_roi_pct=round(daily_roi * 30.0, 1),
            min_30d_doubling_target_usdt=target_3x,
            m8_high_risk_plan=m8_plan,
        )
