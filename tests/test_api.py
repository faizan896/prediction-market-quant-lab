"""API tests via TestClient. No network: everything runs on synthetic data."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from quantlab.api import main as api_main
from quantlab.data.store import Store


@pytest.fixture
def client(tmp_path):
    store = Store(path=tmp_path / "api.db")
    api_main.app.dependency_overrides[api_main.get_store] = lambda: store
    with TestClient(api_main.app) as c:
        yield c
    api_main.app.dependency_overrides.clear()
    store.close()


class TestMeta:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_openapi_schema_builds(self, client):
        """Catches Pydantic models that can't be serialized — a class of bug
        that otherwise only shows up when the frontend calls the endpoint."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        for expected in ("/scan", "/backtest", "/strategies", "/paper", "/markets"):
            assert expected in paths


class TestStrategies:
    def test_lists_every_registered_strategy(self, client):
        r = client.get("/strategies")
        assert r.status_code == 200
        data = r.json()
        ids = {s["id"] for s in data}
        assert {"momentum", "meanrev", "longshot", "fairvalue"} <= ids

    def test_exposes_tunable_params_with_defaults(self, client):
        data = client.get("/strategies").json()
        momentum = next(s for s in data if s["id"] == "momentum")
        params = {p["name"]: p["default"] for p in momentum["params"]}
        assert params["lookback"] == 12
        assert momentum["summary"]


class TestBacktest:
    def test_synthetic_backtest_returns_a_leaderboard(self, client):
        r = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 30, "synthetic_steps": 60,
            "frictions": "frictionless",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["markets"] == 30
        assert len(body["leaderboard"]) >= 5
        assert body["warnings"]
        for row in body["leaderboard"]:
            assert 0 <= row["win_rate"] <= 1
            # JSON has no NaN; non-finite metrics must serialize as null.
            assert row["brier"] is None or isinstance(row["brier"], float)

    def test_leaderboard_is_sorted_by_sharpe(self, client):
        body = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 25, "synthetic_steps": 60,
        }).json()
        sharpes = [r["sharpe"] for r in body["leaderboard"]]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_specific_strategy_with_params(self, client):
        body = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 25, "synthetic_steps": 60,
            "strategies": [{"name": "momentum", "params": {"lookback": 6}}],
        }).json()
        assert len(body["leaderboard"]) == 1
        assert body["leaderboard"][0]["strategy"] == "Momentum"

    def test_unknown_strategy_warns_instead_of_500(self, client):
        body = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 20, "synthetic_steps": 40,
            "strategies": [{"name": "does_not_exist"}],
        }).json()
        assert body["leaderboard"] == []
        assert any("does_not_exist" in w for w in body["warnings"])

    def test_walk_forward_folds(self, client):
        body = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 40, "synthetic_steps": 120,
            "strategies": [{"name": "momentum"}], "walk_forward_folds": 3,
        }).json()
        assert len(body["folds"]) >= 2

    def test_empty_store_is_a_warning_not_an_error(self, client):
        r = client.post("/backtest", json={"synthetic": False})
        assert r.status_code == 200
        assert r.json()["markets"] == 0
        assert r.json()["warnings"]

    def test_equity_curve_is_downsampled_and_json_safe(self, client):
        body = client.post("/backtest", json={
            "synthetic": True, "synthetic_markets": 30, "synthetic_steps": 90,
            "strategies": [{"name": "buyhold"}],
        }).json()
        curve = body["equity"].get("Buy & Hold", [])
        assert len(curve) <= 400
        assert all(isinstance(p["equity"], (int, float)) for p in curve)


class TestScan:
    def test_scan_on_synthetic_markets(self, client):
        r = client.post("/scan", json={
            "venues": ["synthetic"], "limit": 40, "min_volume": 0,
            "min_edge": 0.0, "min_net_ev": -10, "min_confidence": 0.0,
            "frictions": "frictionless",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["scanned"] == 40
        for o in body["opportunities"]:
            assert o["side"] in ("YES", "NO")
            assert o["net_ev"] <= o["gross_ev"] + 1e-9  # costs must bite
            assert 0 <= o["confidence"] <= 1

    def test_one_dead_venue_does_not_sink_the_scan(self, client, monkeypatch):
        """Regression: an unreachable venue used to raise, the request 504'd,
        and the UI showed an empty scan with no explanation — indistinguishable
        from 'no opportunities', which is the one thing it must never be."""
        from quantlab.api import service as svc

        real = svc.PolymarketClient

        class Dead:
            def list_markets(self, **_kw):
                raise RuntimeError("connection timed out")

            def close(self):
                pass

        monkeypatch.setattr(svc, "PolymarketClient", lambda: Dead())
        try:
            body = client.post("/scan", json={
                "venues": ["polymarket", "synthetic"], "limit": 20, "min_volume": 0,
                "min_edge": 0.0, "min_net_ev": -10, "min_confidence": 0.0,
            }).json()
        finally:
            monkeypatch.setattr(svc, "PolymarketClient", real)

        # Synthetic markets still came through...
        assert body["scanned"] == 20
        # ...and the failure is reported rather than silently swallowed.
        assert any("polymarket" in w for w in body["warnings"])

    def test_empty_universe_is_explained(self, client):
        body = client.post("/scan", json={
            "venues": ["synthetic"], "limit": 5, "min_volume": 0,
            "min_edge": 0.9, "min_net_ev": 5.0,
        }).json()
        assert body["opportunities"] == []
        # scanned > 0 with 0 returned is a real result, not a data failure.
        assert body["scanned"] == 5
        assert body["warnings"] == []

    def test_strict_filters_return_nothing(self, client):
        body = client.post("/scan", json={
            "venues": ["synthetic"], "limit": 30, "min_volume": 0,
            "min_edge": 0.9, "min_net_ev": 5.0,
        }).json()
        assert body["opportunities"] == []

    def test_signal_toggles_and_weights_change_the_answer(self, client):
        """Adding a signal must move fair value, and raising its weight must
        move it further. This is the contract the strategy builder UI relies on."""
        base = {"venues": ["synthetic"], "limit": 30, "min_volume": 0,
                "min_edge": 0.0, "min_net_ev": -10, "min_confidence": 0.0,
                "market_anchor": 0.1, "frictions": "frictionless"}

        def max_edge(signals):
            body = client.post("/scan", json={**base, "signals": signals}).json()
            return max((o["edge"] for o in body["opportunities"]), default=0.0)

        # Market prior alone: fair value IS the price, so there is no edge.
        assert max_edge([{"name": "market"}]) < 0.005
        # Adding longshot moves it; at its default weight the market prior
        # (weight 3.0) still dominates, so the move is small but real.
        light = max_edge([{"name": "market"}, {"name": "longshot"}])
        assert light > 0.0005
        # Turning the weight up gives the signal real influence.
        heavy = max_edge([{"name": "market"}, {"name": "longshot", "weight": 30.0}])
        assert heavy > light * 3

    def test_external_model_moves_fair_value(self, client):
        markets = client.post("/scan", json={
            "venues": ["synthetic"], "limit": 5, "min_volume": 0,
            "min_edge": 0.0, "min_net_ev": -10, "min_confidence": 0.0,
        }).json()["opportunities"]
        assert markets
        key = markets[0]["key"]
        body = client.post("/scan", json={
            "venues": ["synthetic"], "limit": 5, "min_volume": 0,
            "min_edge": 0.0, "min_net_ev": -10, "min_confidence": 0.0,
            "market_anchor": 0.0, "signals": [{"name": "market"}],
            "external": {key: [0.99, 1.0]},
        }).json()
        target = next((o for o in body["opportunities"] if o["key"] == key), None)
        assert target is not None
        assert target["fair_value"] > target["price"]


class TestPaper:
    def _create(self, client, **kw):
        r = client.post("/paper", json={"name": "t", "capital": 50_000, **kw})
        assert r.status_code == 201
        return r.json()["run_id"]

    def test_create_and_fetch(self, client):
        run_id = self._create(client)
        r = client.get(f"/paper/{run_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["initial_capital"] == 50_000
        assert body["summary"]["equity"] == 50_000
        assert body["positions"] == []

    def test_missing_run_is_404(self, client):
        assert client.get("/paper/nope").status_code == 404
        assert client.post("/paper/nope/step", json={"synthetic": True}).status_code == 404

    def test_step_runs_a_cycle_and_persists(self, client):
        run_id = self._create(client, min_net_ev=-10, min_confidence=0.0)
        r = client.post(f"/paper/{run_id}/step",
                        json={"synthetic": True, "limit": 40})
        assert r.status_code == 200
        body = r.json()
        assert body["scanned"] == 40
        assert body["summary"]["run_id"] == run_id

        detail = client.get(f"/paper/{run_id}").json()
        assert len(detail["equity"]) >= 1
        assert len(detail["fills"]) == len(body["opened"]) + len(body["closed"]) + len(body["settled"])

    def test_equity_never_exceeds_capital_plus_pnl(self, client):
        run_id = self._create(client, min_net_ev=-10, min_confidence=0.0)
        client.post(f"/paper/{run_id}/step", json={"synthetic": True, "limit": 30})
        s = client.get(f"/paper/{run_id}").json()["summary"]
        assert s["equity"] == pytest.approx(
            s["cash"] + s["exposure"], rel=1e-6
        )

    def test_run_appears_in_the_list(self, client):
        run_id = self._create(client)
        runs = client.get("/paper").json()
        assert run_id in {r["run_id"] for r in runs}


class TestStoredMarkets:
    def test_empty_store_returns_empty_list(self, client):
        assert client.get("/markets").json() == []

    def test_stored_markets_are_serialized(self, client, tmp_path):
        from datetime import datetime, timedelta, timezone

        from quantlab.types import Market, Venue

        store = api_main.app.dependency_overrides[api_main.get_store]()
        store.upsert_markets([
            Market(venue=Venue.KALSHI, market_id="T1", event_id="E1", title="Test",
                   yes_bid=0.40, yes_ask=0.42, volume=5_000,
                   close_time=datetime.now(timezone.utc) + timedelta(days=1))
        ])
        rows = client.get("/markets?min_volume=1000").json()
        assert len(rows) == 1
        assert rows[0]["key"] == "kalshi:T1"
        assert rows[0]["mid"] == pytest.approx(0.41)
