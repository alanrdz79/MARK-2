# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — python m6_binance_bot_sheet_inspector.py
Generador de Ficha "Detalles del Bot" y Tarjeta "Grid de Futuros" idéntica a la App
de Binance, integrando:
- Certidumbre (%) y Mitigación de Incertidumbre con datos de Binance.
- Apalancamiento y Riesgo Operativo de la Fase (Fases 1 a 5: $4 a $1,000,000 USDT).
- ROI y PnL Esperado Diario y a 30 Días de cada mercado, y Días Estimados (Meta 3x en Fases 1-2, 2x en Fase 3).
- Densidad (grids/%), Trades Emparejados (24h/Total), MDD 7D y Escudo Cero-Liquidación (P_liq).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Deque, Optional, Tuple

from core_mark2_contracts import (
    BinomialDDMStatePOD,
    GeometricGridPlanPOD,
    GridDirection,
    HighRiskAcceleratedOrderPlanPOD,
)


@dataclass(frozen=True, slots=True)
class BinanceAppBotDetailsSheet:
    """
    Estructura de la ficha 'Detalles del bot / Grid de futuros' de Binance,
    calibrada por MARK-2 con Certidumbre (%), Apalancamiento/Riesgo de la Fase,
    ROI/PnL Esperado Diario y a 30 Días de cada mercado, y Días Estimados (Meta 3x en F1-F2, 2x en F3).
    """
    symbol: str
    direction_label: str              # Ej: "Long 15x (Efectivo: 0.93x / Max: 3.10x)"
    active_phase_label: str           # Ej: "FASE_1_4_A_100_USDT | Apalancamiento/Riesgo: 15x / 5.20% | Certidumbre: 93.6%"
    pnl_usd: float                    # PnL actual en USDT
    roi_pct: float                    # ROI actual en %
    expected_market_roi_pnl_str: str  # Ej: "Diario: +$0.57 (+9.5%/d) | 30 Días: +$25.87 (+431.2%) | Días Est.: ~5.2d (Meta 3x: $18.00 USDT)"
    runtime_str: str                  # Ej: "3d 15h 8m"
    min_investment_usdt: float        # Capital asignado al bot
    matched_trades_24h: int           # Trades emparejados en las últimas 24h
    matched_trades_total: int         # Total trades emparejados desde creación
    mdd_7d_pct: float                 # Máximo Drawdown en ventana de 7 días (%)
    price_range_usdt: str             # Ej: "2.2850 - 2.5420"
    grid_count_virtual: int           # Cantidad total de grids del canal (2 - 169)
    grid_count_active_book: int       # Órdenes activas simultáneas en el libro
    grid_mode: str                    # "Geométrico"
    profit_per_grid_net_str: str      # Ganancia/grid (se descuentan comisiones Maker)
    grid_density_str: str             # Ej: "2.45 grids/% | Óptima"
    trailing_up_str: str              # "Activado" / "Desactivado"
    trailing_down_str: str            # "Activado" / "Desactivado"
    activation_price_str: str         # Precio inmediato de arranque validado por Binance 3V
    stop_loss_full_str: str           # Precio USDT + PnL USD + ROI % + Piso Ratchet + P_liq
    take_profit_full_str: str         # Precio USDT + Meta Esperada del Mercado (+PnL / +ROI %)
    open_position_on_creation: str    # "Activado" (en Long/Short)
    close_all_on_stop: str            # "Activado" (Purga atómica MARK-2)


class Mark2BotSheetMonitor:
    """
    Monitorea en tiempo real el desempeño del Grid de Futuros en Binance y
    calcula el MDD en 7D, los trades emparejados (24h/Total) y la Densidad de Grid.
    """

    def __init__(self, plan: GeometricGridPlanPOD, start_ts: Optional[float] = None):
        self.plan = plan
        self.start_ts = start_ts or time.time()
        self.fill_timestamps: Deque[float] = deque()
        self.total_matched_trades: int = 0
        self.cumulative_net_pnl_usdt: float = 0.0
        self.equity_history_7d: Deque[Tuple[float, float]] = deque()
        self.equity_history_7d.append((self.start_ts, plan.initial_capital_usdt))

    def register_matched_trade(
        self,
        current_equity_usdt: Optional[float] = None,
        now_ts: Optional[float] = None,
        *,
        net_pnl_usdt: float = 0.0,
        new_equity_usdt: Optional[float] = None,
    ) -> None:
        """Registra cada trade emparejado (compra + venta completada) y actualiza el MDD 7D."""
        ts = now_ts or time.time()
        resolved_equity = (
            float(current_equity_usdt)
            if current_equity_usdt is not None
            else (
                float(new_equity_usdt)
                if new_equity_usdt is not None
                else float(self.plan.initial_capital_usdt + self.cumulative_net_pnl_usdt + net_pnl_usdt)
            )
        )
        self.total_matched_trades += 1
        self.cumulative_net_pnl_usdt += float(net_pnl_usdt)
        self.fill_timestamps.append(ts)
        self.equity_history_7d.append((ts, resolved_equity))

        cutoff_24h = ts - 86_400.0
        while self.fill_timestamps and self.fill_timestamps[0] < cutoff_24h:
            self.fill_timestamps.popleft()

        cutoff_7d = ts - (7.0 * 86_400.0)
        while self.equity_history_7d and self.equity_history_7d[0][0] < cutoff_7d:
            self.equity_history_7d.popleft()

    def compute_mdd_7d_pct(self) -> float:
        """Calcula el Maximum Drawdown porcentual de los últimos 7 días (MDD en 7D)."""
        if not self.equity_history_7d:
            return 0.0
        peak = self.equity_history_7d[0][1]
        max_dd_pct = 0.0
        for _, eq in self.equity_history_7d:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
        return round(max_dd_pct, 2)

    def build_binance_style_sheet(
        self, current_equity_usdt: float, now_ts: Optional[float] = None
    ) -> BinanceAppBotDetailsSheet:
        """
        Genera la ficha exacta con Certidumbre (%), Apalancamiento/Riesgo de la Fase,
        ROI/PnL Esperado Diario y a 30 Días de cada mercado y Días Estimados (3x en F1-F2, 2x en F3).
        """
        ts = now_ts or time.time()
        elapsed_sec = max(1, int(ts - self.start_ts))
        days = elapsed_sec // 86400
        hours = (elapsed_sec % 86400) // 3600
        minutes = (elapsed_sec % 3600) // 60
        runtime_str = f"{days}d {hours}h {minutes}m"

        pnl_usd = round(current_equity_usdt - self.plan.initial_capital_usdt, 2)
        roi_pct = round((pnl_usd / max(self.plan.initial_capital_usdt, 1e-6)) * 100.0, 2)

        total_range_pct = ((self.plan.upper_price - self.plan.lower_price) / self.plan.lower_price) * 100.0
        grids_per_pct = round(self.plan.total_virtual_grids / max(total_range_pct, 0.01), 2)
        density_label = "Alta, objetivo >1" if grids_per_pct >= 1.0 else "Media, paso amplio"

        open_on_create = (
            "Activado"
            if self.plan.direction in (GridDirection.LONG, GridDirection.SHORT)
            else "Desactivado (Neutral espera límite)"
        )
        mult_lbl = (
            f"{self.plan.phase_target_multiplier:.0f}x"
            if self.plan.phase_target_multiplier >= 2.0
            else f"{self.plan.phase_target_multiplier:.2f}x"
        )

        m8 = self.plan.m8_high_risk_plan
        if m8 is not None:
            return BinanceAppBotDetailsSheet(
                symbol=f"{self.plan.symbol} Perp.",
                direction_label=(
                    f"Aislado {m8.isolated_leverage}x "
                    f"(Efectivo: {m8.effective_wallet_leverage:.2f}x | Margen Aislado: ${m8.isolated_margin_usdt:.2f} USDT)"
                ),
                active_phase_label=(
                    f"M8 ALTO RIESGO (${m8.wallet_capital_usdt:.2f} USDT) | CRITIC: {m8.critic_score:.4f} | "
                    f"RV24h: {m8.rv_24h_pct:.1f}% | Cuota/Bruto TP: {m8.fee_to_gross_tp_pct:.2f}% | DDM: {m8.ddm_status.value}"
                ),
                pnl_usd=pnl_usd,
                roi_pct=roi_pct,
                expected_market_roi_pnl_str=(
                    f"TP Neto: +${m8.expected_net_tp_usdt:.3f} (+{m8.expected_wallet_roi_tp_pct:.2f}% Billetera / "
                    f"+{m8.expected_isolated_roi_tp_pct:.1f}% Margen Aislado) | Cuota RT: ${m8.roundtrip_fee_tp_usdt:.4f}"
                ),
                runtime_str=runtime_str,
                min_investment_usdt=self.plan.initial_capital_usdt,
                matched_trades_24h=len(self.fill_timestamps),
                matched_trades_total=self.total_matched_trades,
                mdd_7d_pct=self.compute_mdd_7d_pct(),
                price_range_usdt=f"Compra GTX {m8.anchor_buy_gtx_price} | Venta GTX {m8.anchor_sell_gtx_price}",
                grid_count_virtual=2,
                grid_count_active_book=2,
                grid_mode=(
                    f"{m8.execution_mode} (100% Post-Only GTX Maker | "
                    f"Ratchet BE @ {m8.breakeven_trigger_price} -> {m8.breakeven_lock_price})"
                ),
                profit_per_grid_net_str=(
                    f"+{m8.tp_pct:.2f}% TP (+${m8.expected_net_tp_usdt:.3f} USDT netos | "
                    f"Cuota: {m8.fee_to_gross_tp_pct:.2f}% del margen)"
                ),
                grid_density_str=f"Macro-Paso {m8.macro_step_pct:.2f}% | Menos órdenes, mayor PnL neto",
                trailing_up_str="Activado (Ratchet Breakeven + Escudo de Cuotas)",
                trailing_down_str="Desactivado (Corte Estadístico Binomial DDM)",
                activation_price_str=f"Mark Ref: {m8.mark_price} | Lote: {m8.order_qty} (${m8.notional_usdt:.2f} USDT)",
                stop_loss_full_str=(
                    f"{m8.sl_stop_market_price} USDT ({m8.expected_net_sl_usdt:+.3f} USDT / "
                    f"{m8.expected_wallet_roi_sl_pct:+.2f}% Wallet) | P_liq: {m8.liquidation_price} USDT (< SL)"
                ),
                take_profit_full_str=(
                    f"{m8.tp_limit_gtx_price} USDT (+${m8.expected_net_tp_usdt:.3f} USDT / "
                    f"+{m8.expected_wallet_roi_tp_pct:.2f}% Wallet | +{m8.expected_isolated_roi_tp_pct:.1f}% Margen)"
                ),
                open_position_on_creation="Desactivado (0% cuota Taker inicial)",
                close_all_on_stop="Activado (Purga Atómica + Monitor Gama DDM)",
            )

        return BinanceAppBotDetailsSheet(
            symbol=f"{self.plan.symbol} Perp.",
            direction_label=(
                f"{self.plan.direction.value.capitalize()} {self.plan.leverage}x "
                f"(Efectivo: {self.plan.effective_initial_leverage:.2f}x / Max: {self.plan.max_adverse_effective_leverage:.2f}x)"
            ),
            active_phase_label=(
                f"Área API: {self.plan.safety_tier.value} | {self.plan.active_phase} | "
                f"Apalancamiento/Riesgo Fase: {self.plan.leverage}x / {self.plan.phase_operational_risk_pct:.2f}% | "
                f"Certidumbre: {self.plan.confidence_pct:.1f}%"
            ),
            pnl_usd=pnl_usd,
            roi_pct=roi_pct,
            expected_market_roi_pnl_str=(
                f"Diario: +${self.plan.expected_daily_pnl_usdt:.3f} (+{self.plan.expected_daily_roi_pct:.2f}%/d) | "
                f"30 Días: +${self.plan.expected_30d_pnl_usdt:.2f} (+{self.plan.expected_30d_roi_pct:.1f}%) | "
                f"Días Estimados: ~{self.plan.est_days_to_18_usd:.1f}d (Meta {mult_lbl} 30d: ${self.plan.min_30d_doubling_target_usdt:.2f} USDT)"
            ),
            runtime_str=runtime_str,
            min_investment_usdt=self.plan.initial_capital_usdt,
            matched_trades_24h=len(self.fill_timestamps),
            matched_trades_total=self.total_matched_trades,
            mdd_7d_pct=self.compute_mdd_7d_pct(),
            price_range_usdt=f"{self.plan.lower_price} - {self.plan.upper_price}",
            grid_count_virtual=self.plan.total_virtual_grids,
            grid_count_active_book=self.plan.active_window_orders,
            grid_mode="Geométrico (Mitigación de Incertidumbre + Escudo Cero-Liquidación)",
            profit_per_grid_net_str=(
                f"{self.plan.net_profit_per_cross_pct:.2f}% "
                f"(+${self.plan.net_profit_per_cross_usdt:.4f} USDT netos/cruce)"
            ),
            grid_density_str=f"{grids_per_pct} grids/% | {density_label}",
            trailing_up_str="Activado" if self.plan.trailing_up else "Desactivado",
            trailing_down_str="Activado" if self.plan.trailing_down else "Desactivado",
            activation_price_str="Inmediato (Validado por OBI L1 + Hurst + VR(4) + VPIN)",
            stop_loss_full_str=(
                f"{self.plan.sl_price} USDT | Piso Protegido: ${self.plan.equity_ratchet_floor_usdt:.2f} USDT "
                f"({self.plan.sl_roi_pct:+.1f}% ROI) | P_liq: {self.plan.estimated_liquidation_price} USDT"
            ),
            take_profit_full_str=(
                f"{self.plan.tp_price} USDT | Meta Mercado: ${self.plan.target_equity_usdt:.2f} USDT "
                f"(+${self.plan.tp_pnl_usdt:.2f} USDT / +{self.plan.tp_roi_pct:.1f}% ROI Esperado en ~{self.plan.est_days_to_18_usd:.1f}d)"
            ),
            open_position_on_creation=open_on_create,
            close_all_on_stop="Activado (Purga Atómica MARK-2)",
        )

    @staticmethod
    def format_sheet_as_binance_card(sheet: BinanceAppBotDetailsSheet) -> str:
        """Renderiza en consola la ficha completa con Certidumbre (%), Apalancamiento/Riesgo, ROI/PnL Esperado y Días Estimados."""
        return (
            f"====================================================================================\n"
            f"  GRID DE FUTUROS (MARK-2 API) — {sheet.symbol} [{sheet.direction_label}]\n"
            f"====================================================================================\n"
            f"Fase y Certidumbre:         {sheet.active_phase_label}\n"
            f"PnL (USDT) Actual:          {sheet.pnl_usd:+.2f} USDT\n"
            f"ROI Actual:                 {sheet.roi_pct:+.2f}%\n"
            f"ROI / PnL Esperado Mercado: {sheet.expected_market_roi_pnl_str}\n"
            f"Tiempo de ejecución:        {sheet.runtime_str}\n"
            f"Capital Asignado al Bot:    {sheet.min_investment_usdt:.2f} USDT\n"
            f"Total trades emparejados:   {sheet.matched_trades_24h}/{sheet.matched_trades_total} (24h/Total)\n"
            f"MDD en 7D:                  {sheet.mdd_7d_pct:.2f}%\n"
            f"------------------------------------------------------------------------------------\n"
            f"DETALLES DEL BOT (Parámetros Autocalculados con Datos Reales de Binance):\n"
            f"• Dirección / Apalancam.:   {sheet.direction_label}\n"
            f"• Rango de precios (USDT):  {sheet.price_range_usdt}\n"
            f"• Cant. de grids:           {sheet.grid_count_virtual} ({sheet.grid_count_active_book} activas en libro)\n"
            f"• Modo:                     {sheet.grid_mode}\n"
            f"• Ganancia/grid (neta):     {sheet.profit_per_grid_net_str}\n"
            f"• Densidad de grid:         {sheet.grid_density_str}\n"
            f"• Trailing ascendente:      {sheet.trailing_up_str}\n"
            f"• Trailing descendente:     {sheet.trailing_down_str}\n"
            f"• Precio de activación:     {sheet.activation_price_str}\n"
            f"• Stop Loss (USDT/Riesgo):  {sheet.stop_loss_full_str}\n"
            f"• Take Profit (ROI Esp.):   {sheet.take_profit_full_str}\n"
            f"• Abrir pos. con creación:  {sheet.open_position_on_creation}\n"
            f"• Cerrar pos. ante parada:  {sheet.close_all_on_stop}\n"
            f"===================================================================================="
        )
