"""Los orderId deben sobrevivir EXACTOS al ciclo guardar -> leer en los CSV de ejecución.

Mismo borde que tp_stage_state.csv (01/10/2026): sin dtype, una fila con el id vacío
vuelve la columna float64 y un id de BingX (~2,1e18) sale como '2.1e+18'.
En sl_watch.csv eso ya se veía en prod (31 `stop_loss_hit` con id en notación
científica), y `_append_sl_watch` además pisaba la fila de otro par con
`df.loc[len(df)]`.
"""
import json

import pandas as pd

import pkg.monkey_bx as mb

SL_A = "2103045458464489472"   # STOP de ONDO, 24/09
SL_B = "2103047639611621376"


def _register(rows):
    pd.DataFrame(rows, columns=["symbol", "orderId", "type", "side", "positionSide",
                                "price", "stopPrice", "time"]).to_csv(
        "./archivos/order_id_register.csv", index=False)


class TestSlWatch:
    def test_append_no_pisa_el_watch_de_otro_par(self, isolated_workspace):
        mb._append_sl_watch("ONDO-USDT", 0.4309, "SHORT", SL_A)
        mb._append_sl_watch("BCH-USDT", 246.0, "LONG", None)
        # ONDO ya no es la última fila: antes esto sobrescribía a BCH.
        mb._append_sl_watch("ONDO-USDT", 0.4350, "SHORT", SL_B)

        df = mb._read_orders_csv(mb.SL_WATCH_CSV)
        assert sorted(df["symbol"]) == ["BCH-USDT", "ONDO-USDT"]
        assert df.set_index("symbol").loc["ONDO-USDT", "orderId"] == SL_B

    def test_sync_reconoce_los_stops_vivos_tras_el_csv(self, isolated_workspace, monkeypatch):
        """Antes: el id volvía '2.1e+18' (no pendiente) y la fila sin id daba 'nan'
        (tampoco), así que con la posición viva las DOS filas se descartaban."""
        mb._append_sl_watch("ONDO-USDT", 0.4309, "SHORT", SL_A)
        mb._append_sl_watch("BCH-USDT", 246.0, "LONG", None)
        _register([
            {"symbol": "ONDO-USDT", "orderId": SL_A, "type": "STOP_MARKET", "stopPrice": 0.4309},
            {"symbol": "BCH-USDT", "orderId": "2103047639611625472", "type": "STOP_MARKET",
             "stopPrice": 246.0},
        ])
        monkeypatch.setattr(mb, "total_positions", lambda s: (s, "SHORT", 1.0, 1.0, 0.0))

        mb.sync_cooldowns_from_sl_fills()

        df = mb._read_orders_csv(mb.SL_WATCH_CSV)
        assert sorted(df["symbol"]) == ["BCH-USDT", "ONDO-USDT"]


class TestPendingSnapshot:
    def test_dos_snapshots_iguales_no_inventan_pending_gone(self, isolated_workspace, monkeypatch):
        orders = [
            {"symbol": "ONDO-USDT", "orderId": int(SL_A), "type": "STOP_MARKET",
             "side": "BUY", "positionSide": "SHORT", "stopPrice": "0.4309"},
            {"symbol": "ONDO-USDT", "orderId": 2103045465791528960, "type": "LIMIT",
             "side": "BUY", "positionSide": "SHORT", "price": "0.4251"},
        ]
        monkeypatch.setattr(mb.pkg.bingx, "query_pending_orders",
                            lambda: json.dumps({"code": 0, "data": {"orders": orders}}))
        monkeypatch.setattr(mb, "_log_pending_order_transitions",
                            lambda prev, curr: calls.append((prev, curr)))
        calls = []

        mb.obteniendo_ordenes_pendientes()
        mb.obteniendo_ordenes_pendientes()

        prev, curr = calls[-1]
        assert sorted(prev["orderId"]) == sorted(curr["orderId"]) == sorted(
            ["2103045465791528960", SL_A])

    def test_un_id_vacio_no_rompe_los_demas(self, isolated_workspace):
        pd.DataFrame([
            {"symbol": "ONDO-USDT", "orderId": SL_A, "type": "STOP_MARKET"},
            {"symbol": "BCH-USDT", "orderId": "", "type": "STOP_MARKET"},
        ]).to_csv(mb.ORDER_PENDING_PREV_CSV, index=False)

        df = mb._normalize_orders_df(mb._read_orders_csv(mb.ORDER_PENDING_PREV_CSV))
        assert df["orderId"].tolist() == [SL_A, ""]


class TestEntryWatch:
    def test_el_order_id_vuelve_exacto_con_otra_fila_sin_id(self, isolated_workspace):
        mb._append_entry_watch("AVAX-USDT", "SHORT", 3.0, request_id="r1",
                               order_id="2104654468594270208", side="SELL")
        mb._append_entry_watch("BNB-USDT", "SHORT", 0.04, request_id="r2",
                               order_id="", side="SELL")

        hit = mb._consume_entry_watch("AVAX-USDT", "SHORT")
        assert mb._norm_order_id(hit["order_id"]) == "2104654468594270208"
