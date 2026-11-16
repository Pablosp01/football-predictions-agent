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


def test_fixtures_team_filter_sample_data_empty(client):
    # The bundled sample has no future fixtures, so a team filter
    # still returns the empty list (and the key hint), not an error.
    r = client.get("/fixtures", params={"league": "PL", "team": "Arsenal"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fixtures"] == []
    assert "FOOTBALL_DATA_API_KEY" in body["message"]


def test_fixtures_unknown_team_400(client):
    r = client.get("/fixtures", params={"league": "PL", "team": "Arsenl"})
    assert r.status_code == 400, r.text
    assert "Arsenal" in r.json()["detail"]["did_you_mean"]


def test_ratings_pl_sorted_descending(client):
    r = client.get("/ratings", params={"league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "PL"
    assert body["data_as_of"] == "2025-10-19"  # last sample match
    ratings = [row["rating"] for row in body["ratings"]]
    assert len(ratings) == 23
    assert ratings == sorted(ratings, reverse=True)
    assert body["ratings"][0]["team"] in {row["team"] for row in body["ratings"]}
    assert all(isinstance(row["rating"], float) for row in body["ratings"])


def test_team_arsenal_form_card(client):
    r = client.get("/team", params={"team": "Arsenal", "league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["team"] == "Arsenal"
    assert body["data_as_of"] == "2025-10-19"
    assert len(body["form"]) == 5
    assert all(f in "WDL" for f in body["form"])
    assert 0 <= body["points_per_game"] <= 3
    assert len(body["recent"]) == 5
    # Recent matches are newest first.
    dates = [m["date"] for m in body["recent"]]
    assert dates == sorted(dates, reverse=True)
    assert all(m["opponent"] and m["score"] and m["result"] in "WDL"
               for m in body["recent"])


def test_team_unknown_400_with_suggestions(client):
    r = client.get("/team", params={"team": "Arsenl", "league": "PL"})
    assert r.status_code == 400, r.text
    assert "Arsenal" in r.json()["detail"]["did_you_mean"]


def test_head_to_head_arsenal_chelsea(client):
    r = client.get("/head-to-head",
                   params={"home_team": "Arsenal", "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Arsenal" and body["away_team"] == "Chelsea"
    assert len(body["meetings"]) == 2  # two meetings in the sample
    assert (body["home_wins"] + body["draws"] + body["away_wins"]
            == len(body["meetings"]))
    dates = [m["date"] for m in body["meetings"]]
    assert dates == sorted(dates, reverse=True)
    assert all(m["score"] for m in body["meetings"])


def test_head_to_head_same_team_400(client):
    r = client.get("/head-to-head",
                   params={"home_team": "Arsenal", "away_team": "Arsenal"})
    assert r.status_code == 400, r.text


def test_backtest_returns_metrics(client):
    r = client.get("/backtest", params={"league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_matches"] > 100
    assert 0 <= body["accuracy"] <= 1
    assert 0 <= body["brier"] <= 1
    assert body["log_loss"] > 0
