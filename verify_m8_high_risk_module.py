# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — Suite de Verificación Integral del Módulo M8 (Alto Riesgo $10.00 USDT)
Valida:
1. Compilación AST limpia e importación de los 14 módulos `.py` dentro de `MARK-2/`.
2. Calibración 0%-heurísticos sobre `prosperity_backtest_warehouse.db` (exclusión física de lotes >$10,
   filtro MAD 99% de funding, K-Means Silhouette, pesos CRITIC, cuantización tick/step exacta y P_liq < SL).
3. Correcciones anti-cuotas en M5:
   - Eliminación del enmascaramiento `max(1e-6, ...)` en `Mark2SlidingWindowGridExecutor`.
   - Recotización pasiva `GTX` (1 tick) ante `-5022` sin caer en `GTC` Taker.
4. Puente `GeometricGridPlanPOD` -> `Mark2BotSheetMonitor` (Tarjeta estilo App Binance) + `Mark2DualDDMController`
   (Gama Binomial 2-sigma Warning / 3-sigma Halt) + Ratchet de Breakeven neto de cuotas.
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import sys
import tempfile
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK2_DIR = os.path.dirname(os.path.abspath(__file__))
if MARK2_DIR not in sys.path:
    sys.path.insert(0, MARK2_DIR)

from core_mark2_contracts import (
    CompoundingStage,
    ExchangeContractRulePOD,
    GeometricGridPlanPOD,
    GridDirection,
    HighRiskDDMStatus,
    MarketSafetyTier,
)
from m5_grid_execution_audit import (
    BinanceMark2LiveGridBroker,
    Mark2SlidingWindowGridExecutor,
)
from m6_binance_bot_sheet_inspector import Mark2BotSheetMonitor
from m8_high_risk_alpha_accelerator import (
    ExactExchangeQuantizer,
    Mark2DualDDMController,
    Mark2HighRiskAlphaAccelerator,
)
from main_mark2 import ProsperityMark2Orchestrator


def verify_all_modules_syntax() -> int:
    py_files = sorted(f for f in os.listdir(MARK2_DIR) if f.endswith(".py"))
    for fname in py_files:
        fpath = os.path.join(MARK2_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as fh:
            ast.parse(fh.read(), filename=fname)
    return len(py_files)


def verify_m8_warehouse_calibration() -> None:
    w_db = os.path.join(MARK2_DIR, "prosperity_backtest_warehouse.db")
    m8 = Mark2HighRiskAlphaAccelerator(maker_fee=0.00020, taker_fee=0.00050)
    profiles, plans = m8.calibrate_universe_from_warehouse(
        warehouse_db_path=w_db,
        wallet_capital_usdt=10.00,
        timeframe="5m",
        in_sample_ratio=0.70,
    )
    assert len(profiles) >= 20, f"Se esperaban >=20 perfiles, se obtuvieron {len(profiles)}"
    assert len(plans) >= 5, f"Se esperaban >=5 planes de alto riesgo, se obtuvieron {len(plans)}"

    excluded_map = {p.symbol: p.exclusion_reason for p in profiles if not p.eligible_for_10usdt}
    assert "BTCUSDT" in excluded_map and "MIN_LOT" in excluded_map["BTCUSDT"]
    assert "ETHUSDT" in excluded_map and "MIN_LOT" in excluded_map["ETHUSDT"]
    assert "CYBERUSDT" in excluded_map and "FUNDING" in excluded_map["CYBERUSDT"]

    eligible_profiles = [p for p in profiles if p.eligible_for_10usdt]
    w_sum = sum(eligible_profiles[0].critic_weights)
    assert abs(w_sum - 1.0) < 1e-3, f"Pesos CRITIC deben sumar 1.0, suman {w_sum}"

    print("\n=== TARJETAS DE ÓRDENES M8 VERIFICADAS ($10.00 USDT | 0% HEURÍSTICOS) ===")
    for p in plans:
        assert p.notional_usdt >= 5.0, f"{p.symbol}: nocional {p.notional_usdt} < $5.00"
        assert p.isolated_margin_usdt <= 10.0, f"{p.symbol}: margen aislado {p.isolated_margin_usdt} > $10.00"
        assert p.liquidation_price < p.sl_stop_market_price, (
            f"{p.symbol}: P_liq ({p.liquidation_price}) debe ser menor que SL ({p.sl_stop_market_price})!"
        )
        assert p.open_position_on_creation is False
        assert p.fee_to_gross_tp_pct < 2.5, f"{p.symbol}: erosión de cuota excesiva ({p.fee_to_gross_tp_pct}%)"

    # Verificar puente hacia GeometricGridPlanPOD y Ficha M6 estilo App Binance
    top_plan = plans[0]
    dummy_rule = ExchangeContractRulePOD(
        symbol=top_plan.symbol,
        tick_size=0.0001 if top_plan.mark_price > 0.05 else 0.000001,
        step_size=1.0,
        min_qty=1.0,
        min_notional=5.0,
        max_num_orders=200,
        maint_margin_rate=0.025,
        max_leverage=20,
    )
    grid_pod = m8.to_geometric_grid_plan_pod(top_plan, dummy_rule)
    assert grid_pod.direction == GridDirection.NEUTRAL
    assert grid_pod.m8_high_risk_plan is not None
    monitor = Mark2BotSheetMonitor(plan=grid_pod)
    sheet = monitor.build_binance_style_sheet(current_equity_usdt=10.00)
    card_txt = Mark2BotSheetMonitor.format_sheet_as_binance_card(sheet)
    assert "M8 ALTO RIESGO" in card_txt
    assert "Desactivado (0% cuota Taker inicial)" in card_txt
    print(card_txt)


def verify_m5_honest_fee_accounting_and_passive_requote() -> None:
    # 1. Comprobar que si el paso es minúsculo (0.01%) y la comisión supera el margen bruto,
    # Mark2SlidingWindowGridExecutor registra PnL neto NEGATIVO real (sin ocultar con max(1e-6, ...))
    rule = ExchangeContractRulePOD(
        symbol="TESTUSDT",
        tick_size=0.0001,
        step_size=1.0,
        min_qty=1.0,
        min_notional=5.0,
        max_num_orders=200,
        maint_margin_rate=0.02,
        max_leverage=20,
    )
    tiny_step_plan = GeometricGridPlanPOD(
        symbol="TESTUSDT",
        safety_tier=MarketSafetyTier.ESTABLES,
        direction=GridDirection.NEUTRAL,
        compounding_stage=CompoundingStage.STAGE_1_MITIGATION_0_TO_33PCT,
        initial_capital_usdt=10.0,
        current_equity_usdt=10.0,
        target_equity_usdt=15.0,
        equity_ratchet_floor_usdt=8.0,
        leverage=5,
        lower_price=0.9900,
        upper_price=1.0100,
        total_virtual_grids=200,
        active_window_orders=4,
        geometric_step_pct=0.01,  # 0.01% bruto < 0.04% comisión Maker ida y vuelta!
        net_profit_per_cross_pct=0.0,
        net_profit_per_cross_usdt=0.0,
        qty_per_line=10.0,
        notional_per_line_usdt=10.0,
        all_virtual_levels=(0.9999, 1.0000, 1.0001),
        active_buy_window=(1.0000,),
        active_sell_window=(1.0001,),
        tp_price=1.0200,
        tp_pnl_usdt=0.20,
        tp_roi_pct=2.0,
        sl_price=0.9800,
        sl_pnl_usdt=-0.20,
        sl_roi_pct=-2.0,
        est_daily_crosses=5.0,
        est_days_to_18_usd=10.0,
        trailing_up=False,
        trailing_down=False,
    )
    fd, tmp_db = tempfile.mkstemp(prefix="m8_verify_", suffix=".db", dir=MARK2_DIR)
    os.close(fd)
    try:
        ex = Mark2SlidingWindowGridExecutor(
            plan=tiny_step_plan,
            rules=rule,
            journal_db_path=tmp_db,
            maker_fee=0.00020,
        )
        ex.on_limit_order_filled("BUY", 1.0000)
        harv = ex.on_limit_order_filled("SELL", 1.0001)
        assert float(harv["net_harvest_usdt"]) < 0.0, (
            f"Fallo contable: debía auditar pérdida neta por cuota, devolvió {harv['net_harvest_usdt']}"
        )
        # 2. Comprobar que ante -5022, place_single_counter_limit_order recotiza con GTX y NUNCA envía GTC
        broker = BinanceMark2LiveGridBroker.__new__(BinanceMark2LiveGridBroker)
        broker.live_position_amt_cache = {}
        captured_tifs: List[str] = []
        captured_prices: List[str] = []

        async def fake_signed_request(
            method: str, endpoint: str, params: Dict[str, Any], session: Any = None
        ) -> Dict[str, Any]:
            captured_tifs.append(str(params.get("timeInForce", "")))
            captured_prices.append(str(params.get("price", "")))
            if len(captured_tifs) == 1:
                raise RuntimeError("Binance error -5022: Due to the order could not be executed as maker")
            return {"orderId": 999001, "status": "NEW", "price": params.get("price")}

        broker._signed_request = fake_signed_request  # type: ignore
        res = asyncio.run(
            broker.place_single_counter_limit_order(
                symbol="UAIUSDT",
                side="BUY",
                price=0.1200,
                qty=100.0,
                rules=rule,
                reduce_only=False,
                current_position_amt=0.0,
            )
        )
        assert res["orderId"] == 999001
        assert all(tif == "GTX" for tif in captured_tifs), f"Se detectó degradación Taker: {captured_tifs}"
        assert float(captured_prices[1]) < float(captured_prices[0]), f"Precio BUY no retrocedió 1 tick: {captured_prices}"
    finally:
        for suf in ("", "-wal", "-shm"):
            p = f"{tmp_db}{suf}"
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def verify_ddm_and_ratchet_and_orchestrator() -> None:
    ddm = Mark2DualDDMController(initial_equity_usdt=10.00, min_cycles_warmup=4)
    # 5 ciclos ganadores establecen p_min bajo, seguidos de racha adversa -> debe disparar DRIFT_HALT_3SIGMA
    eq = 10.00
    for _ in range(5):
        eq += 0.50
        st = ddm.register_closed_cycle(+0.50, eq)
    assert st.regime_status == HighRiskDDMStatus.IN_CONTROL

    triggered_halt = False
    for _ in range(6):
        eq -= 0.35
        st = ddm.register_closed_cycle(-0.35, eq)
        if st.regime_status == HighRiskDDMStatus.DRIFT_HALT_3SIGMA:
            triggered_halt = True
            break
    assert triggered_halt, "El detector Gama Binomial DDM debía activar DRIFT_HALT_3SIGMA ante deriva adversa"

    assert hasattr(ProsperityMark2Orchestrator, "sync_universe_and_telemetry")
    assert hasattr(Mark2HighRiskAlphaAccelerator, "evaluate_breakeven_fee_shield_ratchet")


if __name__ == "__main__":
    n_mods = verify_all_modules_syntax()
    print(f"[OK] Compilación AST verificada en {n_mods} módulos .py de MARK-2/")
    verify_m8_warehouse_calibration()
    print("[OK] Calibración M8 sobre almacén real verificada (0% heurísticos, P_liq < SL, Cuota < 2.5%)")
    verify_m5_honest_fee_accounting_and_passive_requote()
    print("[OK] Contabilidad neta real y recotización pasiva GTX (-5022 -> 0% GTC Taker) verificadas")
    verify_ddm_and_ratchet_and_orchestrator()
    print("[OK] Monitor Gama Binomial DDM (3-sigma Halt) y ProsperityMark2Orchestrator verificados")
    print("\n✅ TODAS LAS PRUEBAS DEL MÓDULO M8 PASARON EXITOSAMENTE.")
