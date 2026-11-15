"""League coverage tests: all 12 free-tier competitions, plus the
separate international (WC/EC) rating pool.

Runs against the bundled offline sample data (FOOTBALL_DATA_API_KEY empty).
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402
from football_agent.data.providers import (  # noqa: E402
    FREE_TIER_COMPETITIONS, INTERNATIONAL_LEAGUES,
    InternationalSampleProvider, SampleProvider, get_provider)
from football_agent.model.predict import Predictor  # noqa: E402


@pytest.fixture()
def client():
    api.clear_cache()
    with TestClient(api.app) as c:
        yield c
    api.clear_cache()


def test_every_free_tier_competition_resolves():
    """All 12 free-tier codes are listed, named, and have a provider."""
    assert len(api.LEAGUES) == 12
    assert set(api.LEAGUES) == set(FREE_TIER_COMPETITIONS)
    for code, name in FREE_TIER_COMPETITIONS.items():
        assert api._LEAGUE_NAMES[code] == name
        provider = get_provider(code)
        assert provider is not None
    # Internationals get their own provider; clubs get the sample one.
    assert isinstance(get_provider("WC"), InternationalSampleProvider)
    assert isinstance(get_provider("EC"), InternationalSampleProvider)
    assert isinstance(get_provider("PL"), SampleProvider)
    assert isinstance(get_provider("PD"), SampleProvider)


def test_health_lists_all_twelve(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["leagues"] == api.LEAGUES
    assert len(r.json()["leagues"]) == 12


def test_international_sample_loads():
    provider = InternationalSampleProvider()
    matches = provider.matches("WC")
    assert len(matches) == 480
    assert matches[0].date.isoformat() == "2024-01-12"
    assert matches[-1].date.isoformat() == "2026-10-06"
    teams = {m.home_team for m in matches} | {m.away_team for m in matches}
    assert len(teams) >= 150
    for team in ("Argentina", "Brazil", "France", "England", "Spain",
                 "Germany", "United States", "Morocco", "Japan"):
        assert team in teams
    # WC and EC share one pool: same matches either way.
    ec_matches = provider.matches("EC")
    assert len(ec_matches) == len(matches)
    assert {(m.date, m.home_team, m.away_team) for m in ec_matches} == \
        {(m.date, m.home_team, m.away_team) for m in matches}
    assert provider.fixtures("WC") == []


def test_wc_prediction_uses_international_pool():
    api.clear_cache()
    wc = api.get_predictor("WC")
    ec = api.get_predictor("EC")
    assert "Argentina" in wc.elo.ratings
    assert "Brazil" in wc.elo.ratings
    assert "Arsenal" not in wc.elo.ratings  # no club Elo in the pool
    assert "Arsenal" not in wc.dc.strengths
    # WC and EC are the same pool: identical ratings.
    assert wc.elo.ratings == ec.elo.ratings
    # Neutral venues: no Elo home edge for internationals.
    assert wc.elo.home_advantage == 0.0
    pred = wc.predict("Argentina", "Brazil")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert total == pytest.approx(1.0, abs=0.01)
    assert 0 < pred.p_home < 1 and 0 < pred.p_draw < 1 and 0 < pred.p_away < 1
    assert pred.xg_home > 0 and pred.xg_away > 0
    assert pred.totals and pred.top_pick and pred.match_pick


def test_club_pool_has_no_national_teams():
    api.clear_cache()
    pl = api.get_predictor("PL")
    assert "Arsenal" in pl.elo.ratings
    assert "Argentina" not in pl.elo.ratings
    assert "Brazil" not in pl.elo.ratings
    assert pl.elo.home_advantage == 60.0


def test_predict_argentina_brazil_via_api(client):
    r = client.post("/predict", json={"home_team": "Argentina",
                                      "away_team": "Brazil", "league": "WC"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Argentina" and body["away_team"] == "Brazil"
    assert body["league"] == "WC"
    total = body["p_home"] + body["p_draw"] + body["p_away"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert body["totals"] and body["top_pick"] and body["match_pick"]


def test_predict_usa_alias_resolves_to_united_states(client):
    r = client.post("/predict", json={"home_team": "USA",
                                      "away_team": "Mexico", "league": "WC"})
    assert r.status_code == 200, r.text
    assert r.json()["home_team"] == "United States"


def test_predict_euros_uses_international_pool(client):
    r = client.post("/predict", json={"home_team": "France",
                                      "away_team": "England", "league": "EC"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "EC"
    total = body["p_home"] + body["p_draw"] + body["p_away"]
    assert total == pytest.approx(1.0, abs=0.01)


def test_unknown_league_returns_400(client):
    r = client.post("/predict", json={"home_team": "Arsenal",
                                      "away_team": "Chelsea", "league": "XX"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Unknown league" in detail["error"]
    assert "PL" in detail["known_leagues"]
    assert "WC" in detail["known_leagues"]
    assert client.get("/fixtures", params={"league": "XX"}).status_code == 400
    assert client.get("/backtest", params={"league": "XX"}).status_code == 400


def test_non_pl_club_league_still_needs_key(client):
    # Existing behaviour: without FOOTBALL_DATA_API_KEY, only the
    # bundled PL sample (and the international sample) can be fitted.
    r = client.post("/predict", json={"home_team": "Real Madrid",
                                      "away_team": "Barcelona", "league": "PD"})
    assert r.status_code == 503
    assert "FOOTBALL_DATA_API_KEY" in r.json()["detail"]["hint"]


def test_predictor_home_advantage_param():
    assert Predictor().elo.home_advantage == 60.0
    assert Predictor(home_advantage=0.0).elo.home_advantage == 0.0


def test_page_dropdown_lists_all_competitions(client):
    html = client.get("/").text
    assert 'optgroup label="Club leagues"' in html
    assert 'optgroup label="International tournaments"' in html
    for code in ("PL", "PD", "SA", "BL1", "FL1", "ELC", "DED", "PPL",
                 "BSA", "CL", "WC", "EC"):
        assert f'value="{code}"' in html
    assert "World Cup and Euros work without a data key" in html
    assert "Only the Premier League works without a data key" not in html
    assert 'data-league="WC"' in html
    assert "Argentina vs Brazil" in html
