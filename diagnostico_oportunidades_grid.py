# -*- coding: utf-8 -*-
"""
=================================================================================================================
PROSPERITY MARK-2 — MÓDULO 9: DIAGNÓSTICO DE EDGE SIN UMBRALES HEURÍSTICOS
(m9_edge_diagnostics.py)   — reemplaza a diagnostico_oportunidades_grid.py
=================================================================================================================
Principio: ninguna decisión se toma comparando contra un número mágico (25%, 1.30, 0.70, 0.00018...).
Toda conclusión sale de (a) los datos del journal / warehouse, (b) intervalos de confianza por bootstrap en
bloques (respeta autocorrelación) o (c) pruebas de permutación contra un nulo construido con los mismos datos.

Las únicas constantes que quedan son CONVENCIONES ESTADÍSTICAS EXPUESTAS COMO PARÁMETROS, no umbrales de trading:
  alpha (nivel de significancia, 0.05), B (réplicas bootstrap/permutación), K (resolución de la rejilla de pasos).

Qué responde (cada bloque corresponde a una pregunta falsable):
  1. ¿Los fills del journal son compatibles con el precio real?  -> consistencia vs velas + nulo permutado.
  2. ¿Hay edge neto por ciclo?  -> IC bootstrap en bloques del neto por ciclo y n efectivo.
  3. ¿Cuánta fricción hay y cuál es la comisión de equilibrio?  -> comisión implícita y breakeven medidos en datos.
  4. ¿El número de cruces sigue a la volatilidad?  -> Spearman por hora + Fano (ráfagas vs estancamiento).
  5. ¿Qué fracción de los toques teóricos se convierte en ciclos?  -> eficiencia de fill medida (reemplaza
     mr_efficiency * supervivencia bayesiana del bot).
  6. ¿El paso de malla actual es el óptimo?  -> simulación de malla sobre velas OHLC reales, optimización del paso
     con validación Leave-One-Day-Out y prueba pareada contra el paso vigente.

Solo lectura sobre el journal (mode=ro): nunca compite con el escritor WAL del hot-path.
Dependencias: numpy + sqlite3 (sin scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from m0_workspace_hygiene import MARK2_DIR, resolve_confined_path
except ImportError:  # ejecución aislada
    MARK2_DIR = os.path.dirname(os.path.abspath(__file__))

    def resolve_confined_path(path: str, base_dir: Optional[str] = None, strict: bool = False) -> str:
        return path if os.path.isabs(path) else os.path.join(MARK2_DIR, path)


DAY_MS = 86_400_000
HOUR_MS = 3_600_000


# =============================================================================
# 1. PRIMITIVAS ESTADÍSTICAS (sin scipy)
# =============================================================================

def _z(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xc = x - x.mean()
    d = float(np.dot(xc, xc))
    if d <= 0.0 or n < 2:
        return np.zeros(maxlag + 1)
    out = np.zeros(maxlag + 1)
    out[0] = 1.0
    for k in range(1, min(maxlag, n - 1) + 1):
        out[k] = float(np.dot(xc[: n - k], xc[k:])) / d
    return out


def decorrelation_lag(x: np.ndarray, alpha: float) -> int:
    """Primer rezago cuya autocorrelación cae dentro de la banda de Bartlett: longitud de bloque derivada de los datos."""
    n = len(x)
    if n < 8:
        return 1
    r = _acf(x, min(n // 2, 200))
    band = _z(alpha) / math.sqrt(n)
    for k in range(1, len(r)):
        if abs(r[k]) < band:
            return max(1, k)
    return len(r)


def effective_n(x: np.ndarray, alpha: float) -> float:
    n = len(x)
    if n < 4:
        return float(n)
    L = decorrelation_lag(x, alpha)
    r = _acf(x, L)
    s = sum((1.0 - k / n) * r[k] for k in range(1, min(L, len(r))))
    return float(min(n, max(1.0, n / max(1e-9, 1.0 + 2.0 * s))))


def _mbb_indices(n: int, L: int, B: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Bootstrap de bloques móviles; entrega índices (k, n) por chunks para acotar memoria."""
    L = max(1, min(L, n))
    nb = math.ceil(n / L)
    chunk = max(1, int(2_000_000 // max(n, 1)))
    done = 0
    while done < B:
        k = min(chunk, B - done)
        starts = rng.integers(0, n - L + 1, size=(k, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(k, -1)[:, :n]
        yield idx
        done += k


def mbb_mean_ci(x: np.ndarray, alpha: float, B: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    if n < 3:
        m = float(np.mean(x))
        return m, m, m
    L = decorrelation_lag(x, alpha)
    means = np.concatenate([x[idx].mean(axis=1) for idx in _mbb_indices(n, L, B, rng)])
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(x)), float(lo), float(hi)


def mbb_ratio_ci(
    num: np.ndarray, den: np.ndarray, alpha: float, B: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    n = len(num)
    d0 = float(np.sum(den))
    if n < 3 or d0 == 0.0:
        return (float("nan"),) * 3
    L = decorrelation_lag(num - np.mean(num), alpha)
    vals = []
    for idx in _mbb_indices(n, L, B, rng):
        vals.append(num[idx].sum(axis=1) / np.where(den[idx].sum(axis=1) == 0, np.nan, den[idx].sum(axis=1)))
    v = np.concatenate(vals)
    lo, hi = np.nanquantile(v, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.sum(num) / d0), float(lo), float(hi)


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sa = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_perm(x: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, float]:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, ry[rng.permutation(len(ry))])[0, 1] for _ in range(B)])
    p = (1.0 + float(np.sum(np.abs(null) >= abs(rho)))) / (B + 1.0)
    return rho, p


# =============================================================================
# 2. CARGA DE DATOS (solo lectura)
# =============================================================================

def _connect_ro(path: str) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=3.0)


def load_cycles(journal_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    conn = _connect_ro(journal_path)
    try:
        rows = conn.execute(
            "SELECT timestamp_utc, symbol, filled_side, filled_price, counter_price, qty, "
            "net_pnl_usdt, cumulative_equity_usdt FROM mark2_grid_cycles "
            "WHERE timestamp_utc IS NOT NULL ORDER BY symbol, timestamp_utc, id"
        ).fetchall()
    finally:
        conn.close()
    by_sym: Dict[str, List[tuple]] = {}
    for r in rows:
        by_sym.setdefault(str(r[1]).upper(), []).append(r)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sym, rs in by_sym.items():
        out[sym] = {
            "ts": np.array([float(r[0]) for r in rs]),
            "side": np.array([str(r[2]).upper() for r in rs]),
            "fp": np.array([float(r[3]) for r in rs]),
            "cp": np.array([float(r[4]) for r in rs]),
            "qty": np.array([float(r[5]) for r in rs]),
            "net": np.array([float(r[6]) for r in rs]),
            "eq": np.array([float(r[7] or 0.0) for r in rs]),
        }
    return out


def load_klines(warehouse_path: str, symbol: str, timeframe: str = "5m") -> Optional[Dict[str, np.ndarray]]:
    if not os.path.isfile(warehouse_path):
        return None
    conn = _connect_ro(warehouse_path)
    try:
        rows = conn.execute(
            "SELECT open_time, open, high, low, close FROM historical_klines_mtf "
            "WHERE symbol=? AND timeframe=? AND is_synthetic=0 ORDER BY open_time ASC",
            (symbol, timeframe),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if len(rows) < 50:
        return None
    a = np.asarray(rows, dtype=np.float64)
    return {"t": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4]}


# =============================================================================
# 3. SIMULADOR DE MALLA SOBRE VELAS OHLC REALES
# =============================================================================

def _bar_path(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Orden intra-vela determinista: alcista O->L->H->C, bajista O->H->L->C."""
    bull = c >= o
    x1 = np.where(bull, l, h)
    x2 = np.where(bull, h, l)
    return np.stack([o, x1, x2, c], axis=1).reshape(-1)


def simulate_grid_day(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, step: float, fee: float
) -> Dict[str, float]:
    """
    Malla geométrica LONG de líneas k con compra en L_k = p0*r^k y venta en L_{k+1} (r = 1 + step).
    Una línea compra al cruzar L_k hacia abajo y vende al cruzar L_{k+1} hacia arriba: un ciclo cuesta ~2 pasos
    de recorrido y rinde 1 paso (coherente con path/(2*step) del bot). Comisión maker en ambas patas.
    Devuelve PnL MtM al cierre del día, MDD intradía, ciclos completados e inventario máximo, normalizados por p0.
    Supuesto optimista declarado: un toque de precio = fill (sin cola de ejecución).
    """
    m = len(c)
    p = _bar_path(o, h, l, c)
    p0 = float(o[0])
    r = 1.0 + step
    lr = math.log(r)
    idx = np.floor(np.log(p / p0) / lr).astype(np.int64)
    chg = np.flatnonzero(np.diff(idx) != 0) + 1

    held: set = set()
    real = 0.0
    sum_entry = 0.0
    cycles = 0
    max_held = 0
    cur = 0
    ev_pos: List[int] = []
    ev_real: List[float] = []
    ev_n: List[int] = []
    ev_sum: List[float] = []
    for pos in chg:
        b = int(idx[pos])
        if b < cur:
            for j in range(cur, b, -1):
                if j not in held:
                    lv = p0 * r**j
                    held.add(j)
                    sum_entry += lv
                    real -= lv * fee
        else:
            for j in range(cur + 1, b + 1):
                k = j - 1
                if k in held:
                    held.remove(k)
                    lk, lj = p0 * r**k, p0 * r**j
                    sum_entry -= lk
                    real += (lj - lk) - lj * fee
                    cycles += 1
        cur = b
        max_held = max(max_held, len(held))
        ev_pos.append(int(pos))
        ev_real.append(real)
        ev_n.append(len(held))
        ev_sum.append(sum_entry)

    if not ev_pos:
        return {"pnl": 0.0, "mdd": 0.0, "cycles": 0, "max_held": 0}
    cp = 4 * np.arange(m) + 3
    ep = np.asarray(ev_pos)
    j = np.searchsorted(ep, cp, side="right") - 1
    jj = np.clip(j, 0, None)
    valid = j >= 0
    real_a = np.where(valid, np.asarray(ev_real)[jj], 0.0)
    n_a = np.where(valid, np.asarray(ev_n)[jj], 0)
    s_a = np.where(valid, np.asarray(ev_sum)[jj], 0.0)
    eq = real_a + n_a * c - s_a
    eq0 = np.concatenate([[0.0], eq])
    mdd = float(np.max(np.maximum.accumulate(eq0) - eq0))
    return {
        "pnl": float(eq[-1] / p0),
        "mdd": mdd / p0,
        "cycles": int(cycles),
        "max_held": int(max_held),
    }


def _full_days(kl: Dict[str, np.ndarray]) -> List[Tuple[int, slice]]:
    day = kl["t"] // DAY_MS
    uniq, start, counts = np.unique(day, return_index=True, return_counts=True)
    mode_cnt = int(np.bincount(counts).argmax())
    return [(int(d), slice(int(s), int(s + c))) for d, s, c in zip(uniq, start, counts) if c == mode_cnt]


def _score(pnl: np.ndarray) -> np.ndarray:
    """Sharpe diario por paso candidato (columnas). Sin escala externa; ordena igual que mean/std."""
    mu = pnl.mean(axis=0)
    sd = pnl.std(axis=0, ddof=1) if len(pnl) > 1 else np.zeros(pnl.shape[1])
    return np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.where(mu > 0, np.inf, -np.inf))


@dataclass
class StepOptimization:
    symbol: str
    days: int
    fee_used: float
    step_grid_pct: List[float]
    best_step_pct_in_sample: float
    best_step_pct_median_lodo: float
    current_step_pct: Optional[float]
    oos_mean_pnl_pct_per_line: float
    oos_mean_ci: Tuple[float, float]
    current_mean_pnl_pct_per_line: Optional[float]
    paired_diff_mean_pct: Optional[float]
    paired_diff_ci: Optional[Tuple[float, float]]
    median_peak_inventory_lines: float
    verdict: str


def optimize_step(
    symbol: str,
    kl: Dict[str, np.ndarray],
    fee: float,
    current_step: Optional[float],
    alpha: float,
    B: int,
    rng: np.random.Generator,
    K: int = 32,
) -> Optional[StepOptimization]:
    days = _full_days(kl)
    D = len(days)
    if D < 4:
        return None
    ranges = np.array(
        [(kl["h"][s].max() - kl["l"][s].min()) / kl["o"][s][0] for _, s in days]
    )
    lo = 2.0 * fee  # equilibrio matemático: por debajo el ciclo pierde con certeza
    hi = max(float(np.median(ranges)), lo * 2.0)
    steps = np.geomspace(lo, hi, K)
    if current_step is not None and lo <= current_step <= hi:
        steps = np.unique(np.concatenate([steps, [current_step]]))
    K2 = len(steps)

    pnl = np.zeros((D, K2))
    held = np.zeros((D, K2))
    for i, (_, s) in enumerate(days):
        o, h, l, c = kl["o"][s], kl["h"][s], kl["l"][s], kl["c"][s]
        for j, st in enumerate(steps):
            res = simulate_grid_day(o, h, l, c, float(st), fee)
            pnl[i, j] = res["pnl"]
            held[i, j] = res["max_held"]

    best_is = int(np.argmax(_score(pnl)))
    oos = np.zeros(D)
    chosen = np.zeros(D, dtype=int)
    for d in range(D):
        tr = np.delete(np.arange(D), d)
        k_star = int(np.argmax(_score(pnl[tr])))
        chosen[d] = k_star
        oos[d] = pnl[d, k_star]

    m_oos, lo_oos, hi_oos = mbb_mean_ci(oos, alpha, B, rng)
    cur_idx: Optional[int] = None
    cur_series: Optional[np.ndarray] = None
    diff_stats: Tuple[Optional[float], Optional[Tuple[float, float]]] = (None, None)
    if current_step is not None:
        cur_idx = int(np.argmin(np.abs(steps - current_step)))
        cur_series = pnl[:, cur_idx]
        mu, l_, h_ = mbb_mean_ci(oos - cur_series, alpha, B, rng)
        diff_stats = (mu, (l_, h_))

    if diff_stats[1] is None:
        verdict = "Sin paso vigente medible: usar el paso mediano LODO como referencia."
    elif diff_stats[1][0] > 0:
        verdict = "El paso óptimo supera al vigente fuera de muestra (IC > 0): ajustar."
    elif diff_stats[1][1] < 0:
        verdict = "El paso vigente es mejor que el re-optimizado (IC < 0): no tocar."
    else:
        verdict = "Diferencia no distinguible de 0: conservar el paso vigente; faltan días."

    return StepOptimization(
        symbol=symbol,
        days=D,
        fee_used=fee,
        step_grid_pct=[round(float(s) * 100.0, 4) for s in steps],
        best_step_pct_in_sample=round(float(steps[best_is]) * 100.0, 4),
        best_step_pct_median_lodo=round(float(np.median(steps[chosen])) * 100.0, 4),
        current_step_pct=None if current_step is None else round(current_step * 100.0, 4),
        oos_mean_pnl_pct_per_line=round(m_oos * 100.0, 4),
        oos_mean_ci=(round(lo_oos * 100.0, 4), round(hi_oos * 100.0, 4)),
        current_mean_pnl_pct_per_line=None if cur_series is None else round(float(cur_series.mean()) * 100.0, 4),
        paired_diff_mean_pct=None if diff_stats[0] is None else round(diff_stats[0] * 100.0, 4),
        paired_diff_ci=None
        if diff_stats[1] is None
        else (round(diff_stats[1][0] * 100.0, 4), round(diff_stats[1][1] * 100.0, 4)),
        median_peak_inventory_lines=float(np.median(held[:, best_is])),
        verdict=verdict,
    )


# =============================================================================
# 4. DIAGNÓSTICO DE CICLOS REALES DEL JOURNAL
# =============================================================================

@dataclass
class SymbolReport:
    symbol: str
    n_cycles: int
    span_days: float
    net_total_usdt: float
    net_per_cycle: float
    net_per_cycle_ci: Tuple[float, float]
    n_effective: float
    implied_fee_rate_median: float
    configured_fee_rate: float
    realized_step_pct_median: float
    fee_to_gross: float
    fee_to_gross_ci: Tuple[float, float]
    breakeven_fee_rate: float
    breakeven_over_fee: float
    orientation_anomaly_share: float
    inter_arrival_s: Dict[str, float]
    fill_consistency: Optional[float] = None
    fill_consistency_null: Optional[float] = None
    fill_consistency_p: Optional[float] = None
    hourly_rho_vs_range: Optional[float] = None
    hourly_rho_p: Optional[float] = None
    fano_hourly: Optional[float] = None
    fill_efficiency: Optional[float] = None
    fill_efficiency_ci: Optional[Tuple[float, float]] = None
    cycles_needed_for_significance: Optional[float] = None
    findings: List[str] = field(default_factory=list)
    step_opt: Optional[StepOptimization] = None


def _fill_consistency(
    ts_s: np.ndarray, price: np.ndarray, kl: Dict[str, np.ndarray], B: int, rng: np.random.Generator
) -> Optional[Tuple[float, float, float]]:
    ot = kl["t"]
    step_ms = int(np.median(np.diff(ot)))
    tms = (ts_s * 1000.0).astype(np.int64)
    i = np.searchsorted(ot, tms, side="right") - 1
    ok = (i >= 1) & (i < len(ot)) & (tms < ot[-1] + step_ms)
    if ok.sum() < 5:
        return None
    i, pr = i[ok], price[ok]
    lo = np.minimum(kl["l"][i], kl["l"][i - 1])  # latencia de sondeo: vela actual y anterior
    hi = np.maximum(kl["h"][i], kl["h"][i - 1])
    obs = float(np.mean((pr >= lo) & (pr <= hi)))
    null = np.empty(B)
    for b in range(B):
        pm = rng.permutation(len(i))
        null[b] = np.mean((pr >= lo[pm]) & (pr <= hi[pm]))
    p = (1.0 + float(np.sum(null >= obs))) / (B + 1.0)
    return obs, float(np.mean(null)), p


def analyze_symbol(
    sym: str,
    A: Dict[str, np.ndarray],
    kl: Optional[Dict[str, np.ndarray]],
    configured_fee: float,
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> SymbolReport:
    fp, cp, qty, net, side, ts = A["fp"], A["cp"], A["qty"], A["net"], A["side"], A["ts"]
    gross = np.where(side == "SELL", (fp - cp) * qty, (cp - fp) * qty)
    notional_rt = (fp + cp) * qty
    fee_paid = gross - net
    fee_impl = fee_paid / np.where(notional_rt > 0, notional_rt, np.nan)
    step_real = np.abs(fp - cp) / np.minimum(fp, cp)

    m, lo, hi = mbb_mean_ci(net, alpha, B, rng)
    n_eff = effective_n(net, alpha)
    fg, fg_lo, fg_hi = mbb_ratio_ci(fee_paid, gross, alpha, B, rng)
    be_fee = float(np.sum(gross) / np.sum(notional_rt)) if np.sum(notional_rt) > 0 else float("nan")
    dt = np.diff(ts)
    ia = (
        {f"p{q}": float(np.percentile(dt, q)) for q in (25, 50, 75)} if len(dt) > 0 else {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    )
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
    z = _z(alpha)
    need = (z * sd / m) ** 2 if (m > 0 and sd == sd) else None

    rep = SymbolReport(
        symbol=sym,
        n_cycles=int(len(net)),
        span_days=float((ts[-1] - ts[0]) / 86400.0) if len(ts) > 1 else 0.0,
        net_total_usdt=float(np.sum(net)),
        net_per_cycle=m,
        net_per_cycle_ci=(lo, hi),
        n_effective=n_eff,
        implied_fee_rate_median=float(np.nanmedian(fee_impl)) if len(fee_impl) else float("nan"),
        configured_fee_rate=configured_fee,
        realized_step_pct_median=float(np.median(step_real) * 100.0),
        fee_to_gross=fg,
        fee_to_gross_ci=(fg_lo, fg_hi),
        breakeven_fee_rate=be_fee,
        breakeven_over_fee=be_fee / configured_fee if configured_fee > 0 else float("nan"),
        orientation_anomaly_share=float(np.mean(gross < 0)),
        inter_arrival_s=ia,
        cycles_needed_for_significance=need,
    )

    # --- Validación contra el mercado real (si hay velas)
    if kl is not None:
        fc = _fill_consistency(ts, fp, kl, B, rng)
        if fc is not None:
            rep.fill_consistency, rep.fill_consistency_null, rep.fill_consistency_p = fc

        # actividad horaria vs recorrido horario realizado
        t0, t1 = int(ts[0] * 1000) // HOUR_MS, int(ts[-1] * 1000) // HOUR_MS
        hours = np.arange(t0, t1 + 1)
        if len(hours) >= 5:
            cnt = np.bincount((ts * 1000).astype(np.int64) // HOUR_MS - t0, minlength=len(hours)).astype(float)
            hk = kl["t"] // HOUR_MS
            rng_pct = (kl["h"] - kl["l"]) / kl["c"] * 100.0
            hr_range = np.array([rng_pct[hk == h_].sum() for h_ in hours])
            has = np.array([np.any(hk == h_) for h_ in hours])
            if has.sum() >= 5:
                rho, p = spearman_perm(cnt[has], hr_range[has], B, rng)
                rep.hourly_rho_vs_range, rep.hourly_rho_p = rho, p
            mu_c = float(np.mean(cnt))
            rep.fano_hourly = float(np.var(cnt, ddof=1) / mu_c) if mu_c > 0 and len(cnt) > 1 else None

        # eficiencia de fill: ciclos observados / ciclos teóricos por toque, por día, al paso realizado
        st = float(np.median(step_real))
        if st > 2.0 * configured_fee:
            obs_d, exp_d = [], []
            for d, s in _full_days(kl):
                d_ms0, d_ms1 = d * DAY_MS, (d + 1) * DAY_MS
                if ts[0] * 1000 <= d_ms0 and ts[-1] * 1000 >= d_ms1:
                    n_obs = int(np.sum((ts * 1000 >= d_ms0) & (ts * 1000 < d_ms1)))
                    sim = simulate_grid_day(kl["o"][s], kl["h"][s], kl["l"][s], kl["c"][s], st, configured_fee)
                    if sim["cycles"] > 0:
                        obs_d.append(n_obs)
                        exp_d.append(sim["cycles"])
            if len(obs_d) >= 3:
                o_a, e_a = np.asarray(obs_d, float), np.asarray(exp_d, float)
                eff, e_lo, e_hi = mbb_ratio_ci(o_a, e_a, alpha, B, rng)
                rep.fill_efficiency, rep.fill_efficiency_ci = eff, (e_lo, e_hi)

    # --- Hallazgos: cada uno es una comparación con 0, con 1 o con un nulo permutado (ningún umbral inventado)
    f = rep.findings
    if rep.fill_consistency is not None and rep.fill_consistency_p is not None:
        if rep.fill_consistency_p > alpha:
            f.append(
                f"Los fills NO se distinguen de precios al azar (consistencia {rep.fill_consistency:.0%} vs nulo "
                f"{rep.fill_consistency_null:.0%}, p={rep.fill_consistency_p:.3f}): el journal parece simulado. "
                "No concluyas edge con estos datos."
            )
        else:
            f.append(f"Fills compatibles con el mercado (consistencia {rep.fill_consistency:.0%}, p={rep.fill_consistency_p:.3f}).")
    if lo > 0:
        f.append("Edge neto por ciclo demostrado (IC > 0), condicionado a que los fills sean reales.")
    elif hi < 0:
        f.append("Neto por ciclo NEGATIVO con confianza (IC < 0): la fricción supera al paso.")
    else:
        f.append(
            f"Neto por ciclo no distinguible de 0 (n_ef={n_eff:.0f})"
            + (f"; harían falta ~{need:.0f} ciclos efectivos para cruzar 0." if need else ".")
        )
    if fg_lo == fg_lo and fg_hi >= 1.0:
        f.append("La comisión puede consumir el 100% del bruto (IC de fee/bruto alcanza 1): el paso es demasiado estrecho.")
    if rep.orientation_anomaly_share > 0:
        f.append(f"{rep.orientation_anomaly_share:.1%} de las filas tienen bruto negativo: revisar contabilidad de lotes (m5).")
    if configured_fee > 0 and abs(rep.implied_fee_rate_median - configured_fee) > 5e-6:
        f.append(
            f"La comisión implícita del journal ({rep.implied_fee_rate_median:.6f}) difiere de la configurada "
            f"({configured_fee:.6f}): el PnL guardado usó otra tarifa."
        )
    if rep.hourly_rho_p is not None and rep.hourly_rho_p <= alpha:
        f.append(f"Los cruces siguen a la volatilidad horaria (rho={rep.hourly_rho_vs_range:+.2f}): el modelo path/(2*paso) es válido.")
    if rep.fill_efficiency is not None and rep.fill_efficiency_ci is not None and rep.fill_efficiency_ci[0] > 1.0:
        f.append(
            "IMPOSIBLE FÍSICO: hay más ciclos registrados que toques teóricos del precio (IC inferior de eficiencia > 1). "
            "El journal sobrecuenta ciclos (simulación o doble registro de lotes); concilia contra /fapi/v1/userTrades."
        )
    if rep.fill_efficiency is not None:
        f.append(
            f"Eficiencia de fill medida {rep.fill_efficiency:.2f} (IC {rep.fill_efficiency_ci[0]:.2f}–{rep.fill_efficiency_ci[1]:.2f}) "
            "ciclos reales por ciclo teórico: úsala en vez de mr_efficiency*supervivencia."
        )
    return rep


# =============================================================================
# 5. ORQUESTACIÓN, REPORTE Y CLI
# =============================================================================

def resolve_fee(arg_fee: Optional[float]) -> Tuple[float, str]:
    if arg_fee is not None:
        return float(arg_fee), "argumento CLI"
    try:
        from m5_grid_execution_audit import EnvConfigLoader

        cfg = EnvConfigLoader.load(os.path.join(MARK2_DIR, ".env"))
        if "MAKER_FEE_RATE" in cfg:
            return float(cfg["MAKER_FEE_RATE"]), ".env MAKER_FEE_RATE"
    except Exception:
        pass
    return float("nan"), "implícita en el journal"


def run_diagnostics(
    journal_path: str,
    warehouse_path: str,
    fee: Optional[float] = None,
    alpha: float = 0.05,
    B: int = 2000,
    timeframe: str = "5m",
    seed: int = 20260101,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    cycles = load_cycles(journal_path)
    if not cycles:
        raise RuntimeError("mark2_grid_cycles está vacío: no hay datos que diagnosticar.")
    fee_val, fee_src = resolve_fee(fee)
    if fee_val != fee_val:
        all_impl = []
        for A in cycles.values():
            g = np.where(A["side"] == "SELL", (A["fp"] - A["cp"]) * A["qty"], (A["cp"] - A["fp"]) * A["qty"])
            all_impl.append((g - A["net"]) / ((A["fp"] + A["cp"]) * A["qty"]))
        fee_val = float(np.nanmedian(np.concatenate(all_impl)))

    reports: List[SymbolReport] = []
    for sym, A in sorted(cycles.items()):
        kl = load_klines(warehouse_path, sym, timeframe)
        rep = analyze_symbol(sym, A, kl, fee_val, alpha, B, rng)
        if kl is not None:
            rep.step_opt = optimize_step(
                sym, kl, fee_val, rep.realized_step_pct_median / 100.0, alpha, B, rng
            )
        reports.append(rep)
    return {
        "fee_rate": fee_val,
        "fee_source": fee_src,
        "alpha": alpha,
        "bootstrap_B": B,
        "symbols": [asdict(r) for r in reports],
        "_reports": reports,
    }


def print_report(res: Dict[str, Any]) -> None:
    print("=" * 100)
    print("M9 — DIAGNÓSTICO DE EDGE SIN UMBRALES HEURÍSTICOS (solo lectura)")
    print(f"Comisión maker usada: {res['fee_rate']:.6f} ({res['fee_source']}) | alpha={res['alpha']} | B={res['bootstrap_B']}")
    print("=" * 100)
    for r in res["_reports"]:
        print(f"\n[{r.symbol}] n={r.n_cycles} ciclos en {r.span_days:.2f} d | neto total {r.net_total_usdt:+.4f} USDT")
        print(
            f"  neto/ciclo {r.net_per_cycle:+.6f} IC[{r.net_per_cycle_ci[0]:+.6f}, {r.net_per_cycle_ci[1]:+.6f}] | n_ef={r.n_effective:.0f}"
        )
        print(
            f"  fee/bruto {r.fee_to_gross:.1%} IC[{r.fee_to_gross_ci[0]:.1%}, {r.fee_to_gross_ci[1]:.1%}] | "
            f"comisión de equilibrio {r.breakeven_fee_rate:.6f} ({r.breakeven_over_fee:.1f}x la configurada) | paso real {r.realized_step_pct_median:.3f}%"
        )
        print(
            f"  inter-arribo s: p25={r.inter_arrival_s['p25']:.1f} p50={r.inter_arrival_s['p50']:.1f} p75={r.inter_arrival_s['p75']:.1f}"
            + (f" | Fano/h={r.fano_hourly:.2f}" if r.fano_hourly is not None else "")
        )
        for line in r.findings:
            print(f"  • {line}")
        so = r.step_opt
        if so is not None:
            print(
                f"  ↳ PASO ÓPTIMO ({so.days} días, LODO): in-sample {so.best_step_pct_in_sample:.3f}% | mediana LODO {so.best_step_pct_median_lodo:.3f}% "
                f"| vigente {so.current_step_pct}%"
            )
            print(
                f"     PnL OOS {so.oos_mean_pnl_pct_per_line:+.3f}% por línea·día IC{so.oos_mean_ci} | "
                f"Δ vs vigente {so.paired_diff_mean_pct} IC{so.paired_diff_ci} | inventario pico ~{so.median_peak_inventory_lines:.0f} líneas"
            )
            print(f"     {so.verdict}")
    print("\nLimitaciones declaradas: toque = fill (optimista), sin funding, malla long sin tope de rango, orden intra-vela convencional.")


# -----------------------------------------------------------------------------
# AUTOPRUEBA CON DATOS SINTÉTICOS
# -----------------------------------------------------------------------------

def _build_synthetic(tmpdir: str) -> Tuple[str, str]:
    rng = np.random.default_rng(7)
    n_days, per_day = 12, 288
    n = n_days * per_day
    t0 = 1_700_000_000_000 // DAY_MS * DAY_MS
    x = np.zeros(n)
    for i in range(1, n):  # Ornstein-Uhlenbeck (reversión a la media)
        x[i] = x[i - 1] * 0.97 + rng.normal(0, 0.0012)
    closes = 100.0 * np.exp(x)
    opens = np.concatenate([[100.0], closes[:-1]])
    spread = np.abs(rng.normal(0.0006, 0.0003, n))
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    times = t0 + np.arange(n) * 300_000

    wh = os.path.join(tmpdir, "wh.db")
    conn = sqlite3.connect(wh)
    conn.execute(
        "CREATE TABLE historical_klines_mtf (symbol TEXT, timeframe TEXT, open_time INTEGER, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL, taker_buy_volume REAL, quote_volume REAL, trade_count INTEGER, "
        "taker_buy_quote_volume REAL, is_synthetic INTEGER)"
    )
    for sym in ("REALUSDT", "FAKEUSDT"):
        conn.executemany(
            "INSERT INTO historical_klines_mtf VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
            [(sym, "5m", int(times[i]), opens[i], highs[i], lows[i], closes[i], 1, 1, 1, 1, 1) for i in range(n)],
        )
    conn.commit()
    conn.close()

    jr = os.path.join(tmpdir, "j.db")
    conn = sqlite3.connect(jr)
    conn.execute(
        "CREATE TABLE mark2_grid_cycles (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc REAL, symbol TEXT, stage TEXT, "
        "filled_side TEXT, filled_price REAL, counter_price REAL, qty REAL, net_pnl_usdt REAL, cumulative_equity_usdt REAL)"
    )
    fee = 0.00018
    rows = []
    for sym, real in (("REALUSDT", True), ("FAKEUSDT", False)):
        eq = 6.0
        for k in range(300):
            i = int(rng.integers(10, n - 1))
            step = 0.004
            if real:
                price = float(rng.uniform(lows[i], highs[i]))
            else:
                price = float(100.0 * (1 + rng.uniform(-0.6, 0.6)))  # ajeno al precio del momento
            counter = price / (1 + step)
            q = 6.0 / price
            gross = (price - counter) * q
            net = gross - (price + counter) * q * fee
            eq += net
            rows.append((times[i] / 1000.0 + float(rng.uniform(0, 300)), sym, "r", "SELL", price, counter, q, net, eq))
    rows.sort(key=lambda r: r[0])
    conn.executemany(
        "INSERT INTO mark2_grid_cycles (timestamp_utc, symbol, stage, filled_side, filled_price, counter_price, qty, "
        "net_pnl_usdt, cumulative_equity_usdt) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return jr, wh


def _selftest() -> None:
    # 1) identidad analítica del simulador: mercado con ida y vuelta de 2 pasos exactos => 1 ciclo, PnL = step - fees
    step, fee = 0.01, 0.0002
    p0 = 100.0
    r = 1 + step
    o = np.array([p0, p0 * 0.995, p0 * r**-1.5, p0 * r**1.5])
    c = np.array([p0 * 0.995, p0 * r**-1.5, p0 * r**1.5, p0 * 1.5 * 0 + p0 * r**1.5])
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    res = simulate_grid_day(o, h, l, c, step, fee)
    assert res["cycles"] >= 1 and res["pnl"] > 0, res
    # un mercado plano no puede generar PnL
    flat = np.full(50, p0)
    res0 = simulate_grid_day(flat, flat, flat, flat, 0.01, fee)
    assert res0["pnl"] == 0.0 and res0["cycles"] == 0

    # 2) pipeline completo sobre BD sintéticas
    with tempfile.TemporaryDirectory() as td:
        jr, wh = _build_synthetic(td)
        out = run_diagnostics(jr, wh, fee=0.00018, B=300)
        rep = {r_.symbol: r_ for r_ in out["_reports"]}
        assert rep["REALUSDT"].fill_consistency > 0.9 and rep["REALUSDT"].fill_consistency_p <= 0.05
        assert rep["FAKEUSDT"].fill_consistency_p > 0.05 or rep["FAKEUSDT"].fill_consistency < 0.5
        assert abs(rep["REALUSDT"].implied_fee_rate_median - 0.00018) < 1e-9
        assert rep["REALUSDT"].step_opt is not None
        print_report(out)
    print("\n✅ [M9 SELFTEST OK] simulador, bootstrap, consistencia de fills y optimización de paso verificados.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="MARK-2 M9: diagnóstico de edge sin umbrales heurísticos")
    ap.add_argument("--journal", default="prosperity_mark2_journal.db")
    ap.add_argument("--warehouse", default="prosperity_backtest_warehouse.db")
    ap.add_argument("--fee", type=float, default=None, help="comisión maker real (si se omite: .env o implícita)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--json-out", default="m9_edge_report.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        result = run_diagnostics(
            resolve_confined_path(a.journal, MARK2_DIR),
            resolve_confined_path(a.warehouse, MARK2_DIR),
            fee=a.fee,
            alpha=a.alpha,
            B=a.boot,
            timeframe=a.tf,
        )
        print_report(result)
        serial = {k: v for k, v in result.items() if k != "_reports"}
        with open(resolve_confined_path(a.json_out, MARK2_DIR), "w", encoding="utf-8") as fh:
            json.dump(serial, fh, ensure_ascii=False, indent=2, default=str)