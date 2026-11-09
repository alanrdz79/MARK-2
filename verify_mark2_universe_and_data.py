# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python verify_mark2_universe_and_data.py
Inspector y Verificador de Caja Blanca (100% Datos Reales de la API de Binance):
Muestra en consola exactamente:
1. Qué mercados evalúa el sistema y qué reglas físicas de Binance valida (tickSize, stepSize, minNotional).
2. Cómo mitiga la incertidumbre mediante datos reales de Binance (L1 Bid/Ask, Mark Price, Funding Rate,
   Hurst, Bollinger, ATR, OBI, Taker CVD, Toxicidad VPIN, Razón de Varianza VR(4), Supervivencia Bayesiana).
3. Certidumbre (%), Apalancamiento/Riesgo de la Fase, ROI/PnL Esperado Diario y a 30 Días de cada
   mercado y los Días Estimados (3x en Fases 1 y 2; 2x en Fase 3).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

try:
    import aiohttp
except ImportError:
    aiohttp = None

from m1_universe_3v_ingestion import Mark2UniverseIngestionEngine
from m2_intelligence_regime_ml import Mark2IntelligenceEngine
from m5_grid_execution_audit import MARK2_DIR
from m6_binance_bot_sheet_inspector import Mark2BotSheetMonitor
from main_mark2 import ProsperityMark2Orchestrator


async def run_full_transparency_audit(capital_per_bot_usdt: float = 6.00) -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 136)
    print(
        f"🔍 AUDITORÍA DE INCERTIDUMBRE, 5 FASES Y ROI/PNL ESPERADO POR MERCADO (BINANCE API) — PROSPERITY MARK-2 (Bot: ${capital_per_bot_usdt:.2f} USDT)"
    )
    print("=" * 136)

    orch = ProsperityMark2Orchestrator(env_path=os.path.join(MARK2_DIR, ".env"))

    if aiohttp is None:
        raise RuntimeError("Se requiere 'aiohttp' para conectar a la API de Binance.")

    connector = aiohttp.TCPConnector(
        limit=30,
        limit_per_host=20,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60.0,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=12, connect=5)
    headers = {"Connection": "keep-alive", "Accept-Encoding": "gzip, deflate"}

    print("\n[PASO 1/5] Extrayendo reglas físicas (/fapi/v1/exchangeInfo) y telemetría 3V en vivo de Binance USDⓈ-M...")
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        await orch.sync_universe_and_telemetry(session)

    rules_map = orch.rules_catalog
    telemetry_map = orch.telemetry_catalog
    m3_orchestrator = orch.m3_brain

    # 1B. Imprimir Tabla de las 5 Fases de Capital en Billetera
    print("\n" + "=" * 136)
    print("🏛️ TABLA 0: MATRIZ DE LAS 5 FASES DE CAPITAL EN BILLETERA (RIESGO, APALANCAMIENTO Y META 30 DÍAS)")
    print("=" * 136)
    print(
        f"{'FASE DE CAPITAL':<32} | {'RANGO BILLETERA (USDT)':<24} | {'APALANCAMIENTO':<16} | "
        f"{'RIESGO OP/BOT':<14} | {'LEV EFECTIVO MAX':<16} | {'META MÍNIMA EN 30 DÍAS'}"
    )
    print("-" * 136)
    for sample_bal, range_lbl in (
        (18.0, "$4.00 - $100.00 USDT"),
        (250.0, "$100.00 - $1,000.00 USDT"),
        (2500.0, "$1,000 - $10,000 USDT"),
        (25000.0, "$10,000 - $100,000 USDT"),
        (250000.0, "$100,000 - $1,000,000 USDT"),
    ):
        prof = orch.m4_treasury.resolve_phase_behavior_profile(sample_bal)
        if prof.phase_target_multiplier >= 3.0:
            goal_lbl = f"TRIPLICAR CAPITAL (3x / +{prof.target_roi_per_bot_pct:.0f}% ROI en 30d)"
        elif prof.must_double_in_30d:
            goal_lbl = f"DUPLICAR CAPITAL (2x / +{prof.target_roi_per_bot_pct:.0f}% ROI en 30d)"
        else:
            goal_lbl = f"+{prof.target_roi_per_bot_pct:.0f}% ROI 30d + Retiro {prof.monthly_profit_dividend_pct*100:.0f}%"
        lev_lbl = f"{prof.min_leverage}x-{prof.max_leverage}x (Rec {prof.recommended_leverage}x)"
        print(
            f"{prof.phase.value:<32} | {range_lbl:<24} | {lev_lbl:<16} | "
            f"{prof.max_account_risk_per_bot_pct:>12.1f}% | {prof.max_effective_inventory_leverage:>14.1f}x | {goal_lbl}"
        )

    # 2. Imprimir Tabla 1: Validación de Reglas Físicas de Binance y Datos Crudos 3V
    print("\n" + "=" * 136)
    print("📋 TABLA 1: REGLAS FÍSICAS DE BINANCE Y DATOS CRUDOS EN VIVO VALIDADOS EN LA ADUANA O(1)")
    print("=" * 136)
    print(
        f"{'MERCADO':<12} | {'MARK PRICE':>11} | {'TICK_SIZE':>10} | {'STEP_SIZE':>10} | "
        f"{'MIN_NOTIONAL':>12} | {'SPREAD(bps)':>11} | {'VOL_24H($M)':>11} | {'FUNDING_8H':>10} | {'ADUANA L1'}"
    )
    print("-" * 136)

    for sym in sorted(telemetry_map.keys()):
        rule = rules_map[sym]
        tel = telemetry_map[sym]
        is_ok = Mark2UniverseIngestionEngine.validate_l1_quote_o1(
            tel.bid_price, tel.ask_price, tel.bid_qty, tel.ask_qty
        )
        spread_bps = ((tel.ask_price - tel.bid_price) / max(tel.last_price, 1e-12)) * 10_000.0
        vol_m = tel.quote_vol_24h / 1_000_000.0
        status_3v = "✅ VÁLIDO" if is_ok else "❌ SPREAD INVÁLIDO"
        print(
            f"{sym:<12} | {tel.mark_price:>11.5f} | {rule.tick_size:>10.5f} | {rule.step_size:>10.4f} | "
            f"${rule.min_notional:>11.2f} | {spread_bps:>9.2f}bp | ${vol_m:>10.1f}M | "
            f"{tel.funding_rate * 100:>+9.4f}% | {status_3v}"
        )

    # 3. Imprimir Tabla 2: División Inteligente del Escáner Cripto en 3 Áreas (ESTABLES, INTERMEDIOS, VOLÁTILES)
    active_prof = orch.m4_treasury.resolve_phase_behavior_profile()
    mult_str = f"{active_prof.phase_target_multiplier:.0f}X"
    print("\n" + "=" * 144)
    print(
        f"🧠 TABLA 2: DIVISIÓN INTELIGENTE DEL ESCÁNER CRIPTO EN 3 ÁREAS (ESTABLES | INTERMEDIOS | VOLÁTILES) BASADA EN DATOS DE LA API "
        f"({active_prof.phase.value} | META {mult_str} EN 30 DÍAS)"
    )
    print("=" * 144)

    ranked_candidates = m3_orchestrator.rank_all_markets_safe_to_volatile(
        rules_map=rules_map,
        telemetry_map=telemetry_map,
        bot_capital_usdt=capital_per_bot_usdt,
        max_leverage=None,
        target_equity_usdt=None,
        wallet_balance_usdt=orch.m4_treasury.wallet_balance_usdt,
    )
    grouped_areas = m3_orchestrator.group_markets_by_empirical_area(ranked_candidates)
    area_definitions = (
        ("ESTABLES", "🟢 ÁREA 1: MERCADOS CRIPTO ESTABLES (Datos API: Menor Varianza NATR/EWMA, Menor Spread L1 y Mayor Liquidez 24h)"),
        ("INTERMEDIOS", "🟡 ÁREA 2: MERCADOS CRIPTO INTERMEDIOS (Datos API: Régimen Central de Volatilidad y Rotación Intradía)"),
        ("VOLATILES", "🔴 ÁREA 3: MERCADOS CRIPTO VOLÁTILES (Datos API: Mayor Excursión MAE/MFE, Recorrido 24h y Amplitud de Rango)"),
    )

    global_rank = 1
    for area_key, area_title in area_definitions:
        area_list = grouped_areas.get(area_key, [])
        area_list.sort(
            key=lambda c: (
                not c.viable_for_6usd_bot,
                c.est_days_to_double_capital > 30.0,
                c.uncertainty_score,
                -c.expected_30d_roi_pct,
            )
        )
        print(f"\n{area_title} — [{len(area_list)} mercados]")
        print("-" * 144)
        print(
            f"{'RANK':<4} | {'ÁREA API':<11} | {'MERCADO':<10} | {'CERTIDUMBRE':>11} | {'VR(4)':>6} | {'SURV%':>6} | "
            f"{'DIR / LEV':<12} | {'RIESGO':>6} | {'PNL/ROI ESP. DÍA':<20} | {'PNL/ROI ESP. 30D':<22} | {f'DÍAS EST. ({mult_str})'}"
        )
        print("-" * 144)
        for c in area_list:
            tel = telemetry_map[c.symbol]
            direction = Mark2IntelligenceEngine.classify_grid_direction(tel)
            dir_lev = f"{direction.value:<7} {c.recommended_phase_leverage:>2}x"
            day_esp = f"+${c.expected_daily_pnl_usdt:.3f} (+{c.expected_daily_roi_pct:.2f}%)"
            m30_esp = f"+${c.expected_30d_pnl_usdt:.2f} (+{c.expected_30d_roi_pct:.1f}%)"
            c_mult = f"{c.phase_target_multiplier:.0f}x"
            verdict = (
                f"✅ {c_mult} ~{c.est_days_to_double_capital:.1f}d (<=30d OK)"
                if (c.viable_for_6usd_bot and c.est_days_to_double_capital <= 30.0)
                else (
                    f"🟡 {c_mult} ~{c.est_days_to_double_capital:.1f}d"
                    if c.viable_for_6usd_bot
                    else f"🔒 Min ${c.min_capital_for_grid_usdt:.2f}"
                )
            )
            print(
                f"#{global_rank:<3} | {c.safety_tier.value:<11} | {c.symbol:<10} | {c.confidence_pct:>10.1f}% | {tel.variance_ratio_q4:>6.2f} | "
                f"{tel.bayesian_grid_survival_prob*100:>5.1f}% | {dir_lev:<12} | {c.recommended_risk_pct:>5.2f}% | "
                f"{day_esp:<20} | {m30_esp:<22} | {verdict}"
            )
            global_rank += 1

    ranked_candidates.sort(
        key=lambda c: (
            not c.viable_for_6usd_bot,
            c.est_days_to_double_capital > 30.0,
            c.uncertainty_score,
            -c.expected_30d_roi_pct,
        )
    )

    # 4. Mostrar la Ficha Completa "Detalles del Bot" del Mercado #1 Aprobado
    approved = [c for c in ranked_candidates if c.viable_for_6usd_bot]
    if approved:
        winner = approved[0]
        w_tel = telemetry_map[winner.symbol]
        w_rule = rules_map[winner.symbol]
        plan = m3_orchestrator.build_geometric_grid_6_to_18(
            rules=w_rule,
            tel=w_tel,
            safety_tier=winner.safety_tier,
            initial_capital_usdt=capital_per_bot_usdt,
            current_equity_usdt=capital_per_bot_usdt,
            target_equity_usdt=None,
            max_leverage=winner.recommended_phase_leverage,
            wallet_balance_usdt=orch.m4_treasury.wallet_balance_usdt,
        )
        monitor = Mark2BotSheetMonitor(plan=plan)
        sheet = monitor.build_binance_style_sheet(current_equity_usdt=capital_per_bot_usdt)

        print("\n" + "=" * 136)
        print(
            f"🏆 PASO 3/5: FICHA MATEMÁTICA DEL MERCADO ÓPTIMO ({winner.symbol}) CON ${capital_per_bot_usdt:.2f} USDT EN {plan.active_phase}"
        )
        print("=" * 136)
        print(Mark2BotSheetMonitor.format_sheet_as_binance_card(sheet))

    # 5. Auditar lo que está guardado en la base de datos SQLite WAL local
    db_path = os.path.join(MARK2_DIR, "prosperity_mark2_journal.db")
    print("\n" + "=" * 136)
    print(f"🗄️ PASO 4/5: AUDITORÍA DE LA BASE DE DATOS SQLITE WAL ({db_path})")
    print("=" * 136)
    if not os.path.exists(db_path):
        print(
            "⚠️ Aún no existe 'prosperity_mark2_journal.db'. Ejecuta 'python backtest_mark2_grid_sql.py' para generarla."
        )
        return

    conn = sqlite3.connect(db_path)
    try:
        sessions = conn.execute(
            "SELECT session_id, symbol, direction, leverage, initial_capital_usdt, final_equity_usdt, "
            "net_pnl_usdt, roi_pct, total_crosses, trailing_shifts, mdd_7d_pct, close_reason "
            "FROM mark2_bot_sessions ORDER BY session_id DESC LIMIT 5"
        ).fetchall()

        if sessions:
            print("📌 Últimas Sesiones de Bots Grid Ejecutadas:")
            for s in sessions:
                print(
                    f"   • Sesión #{s[0]} | Mercado: {s[1]} ({s[2]} {s[3]}x) | "
                    f"Capital: ${s[4]:.2f} -> ${s[5]:.2f} USDT (PnL: +${s[6]:.2f} / ROI: +{s[7]:.1f}%) | "
                    f"Trades Emparejados: {s[8]} | Trailing Shifts: {s[9]} | MDD 7D: {s[10]:.2f}% | Cierre: {s[11]}"
                )
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(run_full_transparency_audit(6.00))
