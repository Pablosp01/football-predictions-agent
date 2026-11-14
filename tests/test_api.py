"""API tests: health, predict, aliases, unknown teams, batch, backtest.

Runs against the bundled offline sample data (FOOTBALL_DATA_API_KEY empty).
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402


@pytest.fixture()
def client():
    api.clear_cache()
    with TestClient(api.app) as c:
        yield c
    api.clear_cache()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Elo" in body["model"]
    assert "PL" in body["leagues"]


def test_predict_arsenal_chelsea(client):
    r = client.post("/predict", json={"home_team": "Arsenal", "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Arsenal" and body["away_team"] == "Chelsea"
    assert body["source"] == "model"
    total = body["p_home"] + body["p_draw"] + body["p_away"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert body["xg_home"] > 0 and body["xg_away"] > 0
    assert body["most_likely"] in ("H", "D", "A")
    assert len(body["top_scorelines"]) == 5
    assert body["top_scorelines"][0]["score"] and body["top_scorelines"][0]["prob"] > 0
    assert body["factors"]


def test_predict_aliases(client):
    # "Man United" / "Spurs" must resolve to the canonical sample names.
    r = client.post("/predict", json={"home_team": "Man United", "away_team": "Spurs"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Man Utd"
    assert body["away_team"] == "Tottenham"


def test_predict_unknown_team_returns_400_with_suggestions(client):
    r = client.post("/predict", json={"home_team": "Arsenl", "away_team": "Chelsea"})
    assert r.status_code == 400
    body = r.json()
    detail = body["detail"]
    assert "Unknown team" in detail["error"]
    assert "Arsenal" in detail["did_you_mean"]
    assert "Arsenal" in detail["known_teams"]


def test_predict_batch_ranked(client):
    r = client.post("/predict-batch", json={"fixtures": [
        {"home_team": "Arsenal", "away_team": "Chelsea"},
        {"home_team": "Liverpool", "away_team": "Man City"},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    confs = [p["confidence"] for p in body["predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert [p["rank"] for p in body["predictions"]] == [1, 2]


def test_fixtures_sample_data_has_no_future_fixtures(client):
    r = client.get("/fixtures", params={"league": "PL", "date": "2026-11-15"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fixtures"] == []
    assert "FOOTBALL_DATA_API_KEY" in body["message"]


def test_backtest_returns_metrics(client):
    r = client.get("/backtest", params={"league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_matches"] > 100
    assert 0 <= body["accuracy"] <= 1
    assert 0 <= body["brier"] <= 1
    assert body["log_loss"] > 0
