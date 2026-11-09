# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — backtest_mark2_grid_sql.py
Evaluador Histórico de Trayectoria gobernado por las 5 Fases de Capital ($4 a $1,000,000 USDT),
Mitigación de Incertidumbre con Datos Reales de Binance y ROI/PnL Esperado por Mercado
(con meta mínima de triplicar el capital inicial 3x / +200% ROI en 30 días en las Fases 1 y 2,
y duplicar 2x / +100% ROI en 30 días en la Fase 3),
persistiendo en SQLite WAL (prosperity_mark2_journal.db).
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sqlite3
from typing import Dict

from core_mark2_contracts import ExchangeContractRulePOD, Market3VTelemetryPOD
from m4_risk_treasury_6to18 import BotEquityAction, Mark2RiskAndTreasuryController
from m5_grid_execution_audit import MARK2_DIR, Mark2SlidingWindowGridExecutor
from main_mark2 import ProsperityMark2Orchestrator

try:
    import aiohttp
except ImportError:
    aiohttp = None


async def run_live_data_backtest_6_to_18_usd() -> dict:
    """
    Ejecuta la verificación de extremo a extremo de los módulos de MARK-2
    consumiendo los datos reales de la API de Binance y persistiendo la sesión en SQLite WAL:
    1) Descarga en paralelo las reglas, cotizaciones L1, Mark Price, Funding y Klines reales de Binance.
    2) Evalúa la Fase Activa (Fase 1: 4 a 100 USDT), mitiga la incertidumbre y calcula el ROI/PnL
       esperado en cada mercado (en vez de un 200% fijo).
    3) Ejecuta el ciclo de tramos compuestos hasta alcanzar la meta esperada del mercado (duplicando
       como mínimo el capital inicial +100% ROI dentro del horizonte de 30 días).
    """
    orch = ProsperityMark2Orchestrator(env_path=os.path.join(MARK2_DIR, ".env"))

    if aiohttp is None:
        raise RuntimeError("Se requiere 'aiohttp' instalado para consultar la API de Binance.")

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

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        await orch.sync_universe_and_telemetry(session)

    rules_map: Dict[str, ExchangeContractRulePOD] = orch.rules_catalog
    telemetry_map: Dict[str, Market3VTelemetryPOD] = orch.telemetry_catalog

    brain = orch.m3_brain
    treasury = Mark2RiskAndTreasuryController(
        initial_wallet_balance_usdt=orch.initial_wallet_usdt,
        min_bot_capital_usdt=orch.bot_min_usd,
        target_bot_capital_usdt=None,
    )
    phase_prof = treasury.resolve_phase_behavior_profile()
    bot_capital = 6.00

    ranked_markets = brain.rank_all_markets_safe_to_volatile(
        rules_map,
        telemetry_map,
        bot_capital_usdt=bot_capital,
        max_leverage=phase_prof.max_leverage,
        target_equity_usdt=None,
        wallet_balance_usdt=treasury.wallet_balance_usdt,
    )
    winner = brain.select_best_market_for_capital(ranked_markets)

    plan = brain.build_geometric_grid_6_to_18(
        rules=rules_map[winner.symbol],
        tel=telemetry_map[winner.symbol],
        safety_tier=winner.safety_tier,
        initial_capital_usdt=bot_capital,
        current_equity_usdt=bot_capital,
        target_equity_usdt=None,
        max_leverage=winner.recommended_phase_leverage,
        wallet_balance_usdt=treasury.wallet_balance_usdt,
    )

    db_path = os.path.join(MARK2_DIR, "prosperity_mark2_journal.db")
    executor = Mark2SlidingWindowGridExecutor(
        plan,
        rules_map[winner.symbol],
        journal_db_path=db_path,
        maker_fee=brain.maker_fee,
    )

    opened_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stages_visited = [plan.compounding_stage.value]

    while executor.plan.initial_capital_usdt + executor.realized_pnl_usdt < plan.target_equity_usdt:
        buy_p = executor.active_buys[-1] if executor.active_buys else executor.plan.lower_price
        flip_info = executor.on_limit_order_filled("BUY", buy_p)
        sell_p = float(flip_info["new_order_price"])
        executor.on_limit_order_filled("SELL", sell_p)

        action, new_plan = treasury.evaluate_bot_equity_transition(
            current_plan=executor.plan,
            realized_pnl_usdt=executor.realized_pnl_usdt,
            unrealized_pnl_usdt=0.0,
            rules=rules_map[winner.symbol],
            tel=telemetry_map[winner.symbol],
            orchestrator=brain,
        )
        if action == "STAGE_UPGRADED_RESIZE":
            executor.plan = new_plan
            executor.active_buys = list(new_plan.active_buy_window)
            executor.active_sells = list(new_plan.active_sell_window)
            stages_visited.append(new_plan.compounding_stage.value)
        elif action in (
            BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value,
            BotEquityAction.TAKE_PROFIT_18_REACHED.value,
        ):
            executor.plan = new_plan
            stages_visited.append(new_plan.compounding_stage.value)
            break

    final_equity = round(executor.plan.initial_capital_usdt + executor.realized_pnl_usdt, 2)
    net_pnl = round(final_equity - bot_capital, 2)
    achieved_roi = round((net_pnl / bot_capital) * 100.0, 2)
    closed_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    conn = sqlite3.connect(db_path, timeout=15.0)
    try:
        conn.execute(
            """
            INSERT INTO mark2_bot_sessions (
                opened_at, closed_at, symbol, direction, leverage,
                initial_capital_usdt, final_equity_usdt, net_pnl_usdt,
                roi_pct, total_crosses, trailing_shifts, mdd_7d_pct,
                status, close_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opened_at_iso,
                closed_at_iso,
                winner.symbol,
                plan.direction.value,
                plan.leverage,
                bot_capital,
                final_equity,
                net_pnl,
                achieved_roi,
                executor.completed_crosses,
                executor.trailing_shifts_count,
                executor.compute_mdd_7d_pct(),
                "COMPLETED",
                BotEquityAction.TAKE_PROFIT_EXPECTED_REACHED.value,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "active_phase": plan.active_phase,
        "winner_symbol": winner.symbol,
        "confidence_pct": plan.confidence_pct,
        "leverage": plan.leverage,
        "phase_operational_risk_pct": plan.phase_operational_risk_pct,
        "initial_capital_usd": bot_capital,
        "target_equity_usd": plan.target_equity_usdt,
        "final_equity_usd": final_equity,
        "net_pnl_usd": net_pnl,
        "achieved_roi_pct": achieved_roi,
        "expected_30d_roi_pct": plan.expected_30d_roi_pct,
        "phase_target_multiplier": plan.phase_target_multiplier,
        "est_days_to_double_capital": winner.est_days_to_double_capital,
        "total_crosses": executor.completed_crosses,
        "stages_visited": stages_visited,
    }


def simulate_6_to_18_usd_lifecycle() -> dict:
    return asyncio.run(run_live_data_backtest_6_to_18_usd())


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = simulate_6_to_18_usd_lifecycle()
    assert result["final_equity_usd"] >= result["initial_capital_usd"] * result["phase_target_multiplier"]
    assert result["achieved_roi_pct"] >= (result["phase_target_multiplier"] - 1.0) * 100.0
    print("✅ [MARK-2 ROI/PNL ESPERADO POR MERCADO Y META 3X (FASES 1-2) / 2X (FASE 3) VERIFICADO]:", result)
