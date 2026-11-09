# -*- coding: utf-8 -*-
"""
PROSPERITY MARK-2 — m1_universe_3v_ingestion.py
Extiende p1_data_3v_engine.py de MARK-1 para escanear todos los mercados
perpetuos USDⓈ-M de Binance, validar invariantes en tiempo constante O(1)
y acoplar el almacén histórico SQLite WAL (`m1_historical_sqlite_warehouse.py`).
"""

from __future__ import annotations

import math
from typing import Dict

from core_mark2_contracts import ExchangeContractRulePOD
from m1_historical_sqlite_warehouse import (
    HistoricalCandleRecordM2,
    HistoricalDerivativeRecordM2,
    Mark2BinanceHistoricalETLService,
    Mark2HistoricalDataValidator,
    Mark2SQLiteWarehouseRepository,
)


class Mark2UniverseIngestionEngine:
    """
    Aduana O(1) de datos y parser oficial de /fapi/v1/exchangeInfo.
    Extrae exclusivamente los filtros reales reportados por la API de Binance,
    incluyendo el apalancamiento máximo real del contrato sin topes artificiales.
    """

    @staticmethod
    def parse_binance_exchange_info(raw_exchange_info: dict) -> Dict[str, ExchangeContractRulePOD]:
        """
        Extrae todos los contratos perpetuos USDⓈ-M en estado TRADING y sus
        filtros PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL, maintMarginPercent y
        apalancamiento máximo derivado de requiredMarginPercent reales.
        """
        rules_catalog: Dict[str, ExchangeContractRulePOD] = {}
        symbols_list = raw_exchange_info.get("symbols", [])

        for item in symbols_list:
            if (
                item.get("contractType") != "PERPETUAL"
                or item.get("quoteAsset") != "USDT"
                or item.get("status") != "TRADING"
            ):
                continue

            symbol = item["symbol"]
            tick_size = 0.0
            step_size = 0.0
            min_qty = 0.0
            min_notional = 0.0
            max_num_orders = 200

            raw_maint_pct = float(item.get("maintMarginPercent", 0.5))
            maint_margin_rate = max(0.001, raw_maint_pct / 100.0)

            raw_req_margin_pct = float(item.get("requiredMarginPercent", 0.8))
            if raw_req_margin_pct > 0.0:
                derived_max_lev = int(round(100.0 / raw_req_margin_pct))
                max_leverage = max(20, min(125, derived_max_lev))
            else:
                max_leverage = 125

            for flt in item.get("filters", []):
                f_type = flt.get("filterType")
                if f_type == "PRICE_FILTER":
                    tick_size = float(flt.get("tickSize", 0.0))
                elif f_type == "LOT_SIZE":
                    step_size = float(flt.get("stepSize", 0.0))
                    min_qty = float(flt.get("minQty", 0.0))
                elif f_type in ("MIN_NOTIONAL", "NOTIONAL"):
                    notional_val = flt.get("notional", flt.get("minNotional", 0.0))
                    min_notional = float(notional_val)
                elif f_type == "MAX_NUM_ORDERS":
                    max_num_orders = int(flt.get("limit", 200))

            if tick_size <= 0.0 or step_size <= 0.0 or min_qty <= 0.0 or min_notional <= 0.0:
                continue

            rules_catalog[symbol] = ExchangeContractRulePOD(
                symbol=symbol,
                tick_size=tick_size,
                step_size=step_size,
                min_qty=min_qty,
                min_notional=min_notional,
                max_num_orders=max_num_orders,
                maint_margin_rate=maint_margin_rate,
                max_leverage=max_leverage,
            )
        return rules_catalog

    @staticmethod
    def validate_l1_quote_o1(bid: float, ask: float, bid_qty: float, ask_qty: float) -> bool:
        """Aduana determinista O(1): descarta spreads invertidos, nulos o NaNs."""
        if math.isnan(bid) or math.isnan(ask) or bid <= 0.0 or ask <= 0.0:
            return False
        if bid >= ask or bid_qty <= 0.0 or ask_qty <= 0.0:
            return False
        return True
