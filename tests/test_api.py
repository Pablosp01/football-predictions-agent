"""API tests: health, predict, aliases, unknown teams, batch, backtest.

Runs against the bundled offline sample data (FOOTBALL_DATA_API_KEY empty).
"""
import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402
from football_agent.data.providers import clear_espn_cache  # noqa: E402


def _espn_event(when, state, home, away):
    return {
        "date": when,
        "status": {"type": {"state": state}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"displayName": home}},
            {"homeAway": "away", "team": {"displayName": away}},
        ]}],
    }


def _espn_result(when, home, away, home_goals, away_goals):
    ev = _espn_event(when, "post", home, away)
    ev["competitions"][0]["competitors"][0]["score"] = str(home_goals)
    ev["competitions"][0]["competitors"][1]["score"] = str(away_goals)
    return ev


class _FakeEspnResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


# Mon 16 Nov 2026: Nations League day (two scheduled, one finished);
# no club league has a match that day.
_ESPN_NATIONS_DAY = {"events": [
    _espn_event("2026-11-16T17:00Z", "pre", "Iceland", "Luxembourg"),
    _espn_event("2026-11-16T19:45Z", "pre", "Germany", "Netherlands"),
    _espn_event("2026-11-16T15:00Z", "post", "Faroe Islands", "Moldova"),
]}


def _fake_espn_get(url, timeout=None):
    # Date-gated: each feed answers only on its day, so /today on a
    # different date sees nothing from it.
    if "uefa.nations" in url and "dates=20261116" in url:
        return _FakeEspnResponse(_ESPN_NATIONS_DAY)
    if "ned.1" in url and "dates=20261116" in url:
        return _FakeEspnResponse({"events": [
            _espn_event("2026-11-16T19:45Z", "pre", "Ajax", "PSV"),
        ]})
    if "eng.2" in url and "dates=20261116" in url:
        return _FakeEspnResponse({"events": [
            _espn_event("2026-11-16T19:45Z", "pre", "Leeds United",
                        "Norwich City"),
        ]})
    if "mex.1" in url and "dates=20261116" in url:
        return _FakeEspnResponse({"events": [
            _espn_event("2026-11-16T20:00Z", "pre", "Pachuca", "Toluca"),
        ]})
    if "mex.1" in url:
        # Past days: finished Liga MX matches with scores, so the
        # ESPN-history provider can fit MX1 (42 matches over 21 days).
        return _FakeEspnResponse({"events": [
            _espn_result("2026-11-08T20:00Z", "Pachuca", "Toluca", 2, 1),
            _espn_result("2026-11-08T22:00Z", "Toluca", "América", 1, 0),
        ]})
    if "fifa.world" in url and "dates=20260719" in url:
        return _FakeEspnResponse({"events": [
            _espn_event("2026-07-19T19:00Z", "pre", "Brazil", "Argentina"),
        ]})
    if "eng.1" in url and "dates=20261116" in url:
        # A real PL fixture, plus a phantom one: ESPN lists Coventry
        # City (a Championship team) in the eng.1 feed. The model
        # must not resolve Coventry to Man City and predict it.
        return _FakeEspnResponse({"events": [
            _espn_event("2026-11-16T15:00Z", "pre", "Arsenal", "Chelsea"),
            _espn_event("2026-11-16T17:30Z", "pre", "Coventry City",
                        "Crystal Palace"),
        ]})
    return _FakeEspnResponse({"events": []})


@pytest.fixture()
def client():
    api.clear_cache()
    clear_espn_cache()
    with TestClient(api.app) as c:
        yield c
    api.clear_cache()
    clear_espn_cache()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Elo" in body["model"]
    assert "PL" in body["leagues"]


def test_health_briefing_lists_configured_keys_only(monkeypatch):
    """GET /health names which briefing keys are set — never values.

    With only an OpenAI key set, the list is ["openai"]; the key
    itself appears nowhere in the response.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XINGCHI_API_KEY", raising=False)
    with TestClient(api.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["briefing"] == ["openai"]
    assert "openai-test-key" not in r.text


def test_health_briefing_empty_when_no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XINGCHI_API_KEY", raising=False)
    with TestClient(api.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["briefing"] == []


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


def test_predict_coventry_city_is_not_man_city(client):
    """A substring alias must not claim a team: "Coventry City" is
    not Man City, even though the alias "city" is a substring of it.
    ESPN lists Coventry vs Crystal Palace in the eng.1 feed, and the
    old alias match predicted it as Man City."""
    r = client.post("/predict", json={"home_team": "Coventry City",
                                      "away_team": "Crystal Palace"})
    assert r.status_code == 400, r.text
    assert "Unknown team" in r.json()["detail"]["error"]


def test_predict_city_alias_still_man_city(client):
    """Typing just "City" must still resolve to Man City."""
    r = client.post("/predict", json={"home_team": "City",
                                      "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    assert r.json()["home_team"] == "Man City"


def test_predict_same_team_400(client):
    """A team against itself is not a match — including through an
    alias ("City" and "Man City" are the same team)."""
    r = client.post("/predict", json={"home_team": "Arsenal",
                                      "away_team": "Arsenal"})
    assert r.status_code == 400, r.text
    assert "different" in r.json()["detail"]["error"]
    r = client.post("/predict", json={"home_team": "City",
                                      "away_team": "Man City"})
    assert r.status_code == 400, r.text


def test_resolve_team_word_match():
    """Names whose words are not contiguous still resolve:
    "Celta Vigo" -> "RC Celta de Vigo" (football-data.org's name).
    Without this, Real Madrid vs Celta Vigo on 2026-11-21 had no
    prediction at all."""
    from football_agent import api
    known = ["RC Celta de Vigo", "Real Madrid CF", "Sevilla FC"]
    with patch.object(api, "_known_teams", lambda league: known):
        assert api.resolve_team("Celta Vigo", "PD") == "RC Celta de Vigo"
        assert api.resolve_team("Real Madrid", "PD") == "Real Madrid CF"


def test_predict_fixtures_skips_same_team():
    """A fixture listing the same team twice gets no prediction
    (ESPN's "Alavés vs Deportivo" for 2026-11-21, both resolving to
    Deportivo Alavés) — it is listed, not predicted."""
    from datetime import date as _date
    from football_agent.api import _predict_fixtures
    from football_agent.data.types import Match
    out = _predict_fixtures(
        [Match(date=_date(2026, 11, 21), season="", league="PL",
               home_team="Arsenal", away_team="Arsenal")], "PL")
    assert len(out) == 1
    assert out[0]["prediction"] is None


def test_api_responses_are_not_cached(client):
    """Every answer is fresh: a deploy changes the answers, and a
    saved copy shows the old behaviour (the 2026-11-16 /today stale
    read)."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")


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


def test_predict_falls_back_to_the_league_the_teams_play_in(client):
    """The chatbot trap: the chatbot inherits the Predict tab's
    league, so a match asked with the tab on the wrong league came
    back "Unknown team". The API now finds the teams in the league
    they play in — Spain and France are in the international sample
    only, so with the tab on PL they fall back to the Nations League."""
    r = client.post("/predict", json={"home_team": "Spain",
                                      "away_team": "France",
                                      "league": "PL"})
    assert r.status_code == 200, r.text
    assert r.json()["league"] == "NL"


def test_predict_falls_back_to_espn_history_league(client):
    """Vincent's rule: a prediction asked in any league searches
    every league before answering — a Liga MX pair asked with the
    tab on PL predicts in MX1, not "Unknown team"."""
    from datetime import date as _date
    from football_agent.data.types import Match as _Match
    from football_agent.model.predict import Predictor as _Predictor
    real_known = api._known_teams
    real_predictor = api.get_predictor
    history = ([_Match(date=_date(2026, 11, 8), season="", league="MX1",
                       home_team="Pachuca", away_team="Toluca",
                       home_goals=2, away_goals=1),
                _Match(date=_date(2026, 11, 8), season="", league="MX1",
                       home_team="Toluca", away_team="América",
                       home_goals=1, away_goals=0)] * 6)
    mx1_predictor = _Predictor().fit(history)

    def fake_known(league):
        if league == "MX1":
            return ["Pachuca", "Toluca", "América"]
        return real_known(league)

    def fake_predictor(league):
        if league == "MX1":
            return mx1_predictor
        return real_predictor(league)

    with patch.object(api, "_known_teams", fake_known), \
         patch.object(api, "get_predictor", fake_predictor):
        r = client.post("/predict", json={"home_team": "Pachuca",
                                          "away_team": "Toluca",
                                          "league": "PL"})
    assert r.status_code == 200, r.text
    assert r.json()["league"] == "MX1"


def test_predict_fallback_keeps_the_original_error(client):
    """A pair no league can predict keeps the error for the league
    asked about — Derby County plays nowhere in the sample data."""
    r = client.post("/predict", json={"home_team": "Derby County",
                                      "away_team": "Southampton",
                                      "league": "PL"})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "Unknown team" in detail["error"]
    assert "Derby County" in detail["error"]


def test_predict_fallback_when_the_league_cannot_fit(client):
    """The reported bug, replayed: the Championship has too little
    history to fit, so Arsenal vs Chelsea asked with the tab on ELC
    must predict in the Premier League instead of failing — while a
    pair that only plays in the Championship keeps the
    Championship's own error."""
    from fastapi import HTTPException as _HTTPException

    real_known = api._known_teams
    real_predictor = api.get_predictor

    def fake_known(league):
        if league == "ELC":
            return ["Derby County", "Southampton"]
        return real_known(league)

    def fake_predictor(league):
        if league == "ELC":
            raise _HTTPException(status_code=503, detail={
                "error": "Not enough historical matches to fit the model for league 'ELC'",
                "matches_available": 6})
        return real_predictor(league)

    with patch.object(api, "_known_teams", fake_known), \
         patch.object(api, "get_predictor", fake_predictor):
        r = client.post("/predict", json={"home_team": "Arsenal",
                                          "away_team": "Chelsea",
                                          "league": "ELC"})
        assert r.status_code == 200, r.text
        assert r.json()["league"] == "PL"
        r = client.post("/predict", json={"home_team": "Derby County",
                                          "away_team": "Southampton",
                                          "league": "ELC"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["matches_available"] == 6


def test_team_falls_back_to_the_league_the_team_plays_in(client):
    """Spain's form asked with the tab on the Premier League must
    answer from the Nations League, not "Unknown team"."""
    r = client.get("/team", params={"team": "Spain", "league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["team"] == "Spain"
    assert body["league"] == "NL"


def test_resolve_team_live_names():
    """The live football-data.org names differ from the bundled
    sample names ("Manchester City FC" vs "Man City"): the short
    forms Vincent types must resolve to the live name when the
    live list is in use."""
    live = ["Manchester City FC", "Fulham FC", "Manchester United FC",
            "Liverpool FC", "Arsenal FC", "Chelsea FC",
            "Tottenham Hotspur FC", "Wolverhampton Wanderers FC",
            "Nottingham Forest FC", "Brighton & Hove Albion FC",
            "Leeds United FC", "Southampton FC"]
    with patch.object(api, "_known_teams", lambda league: live):
        assert api.resolve_team("Man City", "PL") == "Manchester City FC"
        assert api.resolve_team("Man Utd", "PL") == "Manchester United FC"
        assert api.resolve_team("Man United", "PL") == "Manchester United FC"
        assert api.resolve_team("Spurs", "PL") == "Tottenham Hotspur FC"
        assert api.resolve_team("Wolves", "PL") == "Wolverhampton Wanderers FC"
        assert api.resolve_team("Nott'm Forest", "PL") == "Nottingham Forest FC"
        assert api.resolve_team("Brighton", "PL") == "Brighton & Hove Albion FC"
        assert api.resolve_team("Leeds", "PL") == "Leeds United FC"
        assert api.resolve_team("Saints", "PL") == "Southampton FC"
        assert api.resolve_team("City", "PL") == "Manchester City FC"
        assert api.resolve_team("United", "PL") == "Manchester United FC"


def test_resolve_team_folds_accents():
    """Accents fold to plain letters: Vincent typing 'Alaves'
    (no accent) must match the live 'Deportivo Alavés', and
    'Atletico Madrid' must match 'Club Atlético de Madrid'.
    Articles are ignored in the word match, so 'RC deportivo
    de A coruna' still finds 'RC Deportivo La Coruña'."""
    live_pd = ["FC Barcelona", "Real Madrid CF", "Club Atlético de Madrid",
               "Real Betis Balompié", "Real Sociedad de Fútbol",
               "Athletic Club", "Sevilla FC", "RC Deportivo La Coruña",
               "Villarreal CF", "RC Celta de Vigo",
               "Rayo Vallecano de Madrid",
               "Real Racing Club de Santander", "Deportivo Alavés",
               "Levante UD", "Getafe CF", "CA Osasuna",
               "RCD Espanyol de Barcelona", "Elche CF", "Valencia CF",
               "Málaga CF"]
    with patch.object(api, "_known_teams", lambda league: live_pd):
        assert api.resolve_team("Alaves", "PD") == "Deportivo Alavés"
        assert api.resolve_team("Atletico Madrid", "PD") == "Club Atlético de Madrid"
        assert api.resolve_team("Malaga", "PD") == "Málaga CF"
        assert api.resolve_team("Real Sociedad", "PD") == "Real Sociedad de Fútbol"
        assert api.resolve_team("RC deportivo de A coruna", "PD") == "RC Deportivo La Coruña"


def test_resolve_team_english_names():
    """English names for teams whose live names are local:
    'Bayern Munich' for FC Bayern München, 'Inter Milan' for
    FC Internazionale Milano, 'Cologne' for 1. FC Köln,
    'Turin' for Torino FC."""
    live_bl1 = ["FC Bayern München", "Borussia Dortmund", "SC Freiburg",
                "RB Leipzig", "SV Werder Bremen", "Bayer 04 Leverkusen",
                "Eintracht Frankfurt", "SV 07 Elversberg", "VfB Stuttgart",
                "Hamburger SV", "TSG 1899 Hoffenheim", "SC Paderborn 07",
                "1. FSV Mainz 05", "FC Augsburg", "1. FC Köln",
                "1. FC Union Berlin", "FC Schalke 04",
                "Borussia Mönchengladbach"]
    with patch.object(api, "_known_teams", lambda league: live_bl1):
        assert api.resolve_team("Bayern Munich", "BL1") == "FC Bayern München"
        assert api.resolve_team("Bayern München", "BL1") == "FC Bayern München"
        assert api.resolve_team("Cologne", "BL1") == "1. FC Köln"
        assert api.resolve_team("Monchengladbach", "BL1") == "Borussia Mönchengladbach"
    live_sa = ["SS Lazio", "AS Roma", "FC Internazionale Milano", "AC Milan",
               "Como 1907", "Frosinone Calcio", "Juventus FC",
               "Bologna FC 1909", "SSC Napoli", "Cagliari Calcio",
               "Torino FC", "Atalanta BC", "Genoa CFC", "ACF Fiorentina",
               "US Lecce", "US Sassuolo Calcio", "Parma Calcio 1913",
               "AC Monza", "Venezia FC", "Udinese Calcio"]
    with patch.object(api, "_known_teams", lambda league: live_sa):
        assert api.resolve_team("Inter Milan", "SA") == "FC Internazionale Milano"
        assert api.resolve_team("Turin", "SA") == "Torino FC"
        assert api.resolve_team("Inter", "SA") == "FC Internazionale Milano"


def test_predict_survives_a_rate_limited_league(client):
    """The 2026-11-17 09:29 failure: a cold restart made the
    fallback search fetch nine leagues at once, the free feed
    capped at 10 requests a minute, and the 429 became an
    unhandled 500 ("Something went wrong"). A league whose data
    cannot be fetched is now skipped, and the request gets the
    handled error for the league asked about instead."""
    import requests

    real_known = api._known_teams

    def flaky_known(league):
        if league == "EC":
            raise requests.exceptions.HTTPError(
                "429 Client Error: Too Many Requests")
        return real_known(league)

    with patch.object(api, "_known_teams", flaky_known):
        r = client.post("/predict", json={"home_team": "Derby County",
                                          "away_team": "Southampton",
                                          "league": "PL"})
        assert r.status_code == 400, r.text
        assert "Unknown team" in r.json()["detail"]["error"]


def test_get_predictor_rate_limit_is_a_503():
    """A failed data fetch is a 503 with a plain message, never a 500."""
    import requests
    from fastapi import HTTPException as _HTTPException

    class _FailingProvider:
        def matches(self, league="PL"):
            raise requests.exceptions.HTTPError("429 Client Error")
        def fixtures(self, league="PL", on=None):
            raise requests.exceptions.HTTPError("429 Client Error")

    api.clear_cache()
    with patch.object(api, "get_provider",
                      lambda league="PL": _FailingProvider()):
        with pytest.raises(_HTTPException) as exc:
            api.get_predictor("PD")
        assert exc.value.status_code == 503
        assert "temporarily unavailable" in exc.value.detail["error"]


def test_predict_falls_back_when_the_requested_league_feed_fails(client):
    """A rate-limited feed for the league asked about must not 500:
    the request falls back to a league whose data is cached and
    predicts there."""
    import requests

    class _FailingProvider:
        def matches(self, league="PL"):
            raise requests.exceptions.HTTPError(
                "429 Client Error: Too Many Requests")
        def fixtures(self, league="PL", on=None):
            raise requests.exceptions.HTTPError(
                "429 Client Error: Too Many Requests")

    real_provider = api.get_provider

    def flaky_provider(league="PL"):
        if league == "PD":
            return _FailingProvider()
        return real_provider(league)

    with patch.object(api, "get_provider", flaky_provider):
        r = client.post("/predict", json={"home_team": "Arsenal",
                                          "away_team": "Chelsea",
                                          "league": "PD"})
        assert r.status_code == 200, r.text
        assert r.json()["league"] == "PL"


def test_team_unknown_everywhere_keeps_the_original_error(client):
    r = client.get("/team", params={"team": "Arsenl", "league": "PL"})
    assert r.status_code == 400, r.text
    assert "Arsenal" in r.json()["detail"]["did_you_mean"]


def test_head_to_head_falls_back_to_the_league_the_teams_play_in(client):
    r = client.get("/head-to-head", params={"home_team": "Spain",
                                            "away_team": "France",
                                            "league": "PL"})
    assert r.status_code == 200, r.text
    assert r.json()["league"] == "NL"


def test_predict_batch_league_follows_the_predictions(client):
    """A batch asked with the tab on PL, of national-team pairs,
    reports the Nations League, not the tab's league."""
    r = client.post("/predict-batch", json={"fixtures": [
        {"home_team": "Spain", "away_team": "France"},
        {"home_team": "Germany", "away_team": "Netherlands"},
    ], "league": "PL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "NL"
    assert all(p["league"] == "NL" for p in body["predictions"])


def test_fixtures_team_filter_finds_the_team_in_another_league(client):
    """Iceland's fixtures asked with the tab on PL must come from
    the Nations League fixtures, not "Unknown team"."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/fixtures", params={"league": "PL", "team": "Iceland",
                                            "date": "2026-11-16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "NL"
    assert len(body["fixtures"]) == 1
    assert body["fixtures"][0]["home_team"] == "Iceland"
    assert body["fixtures"][0]["prediction"] is not None


def test_fixtures_sample_data_has_no_future_fixtures(client):
    # ESPN is the fixtures source when no data key is set; it says
    # there are no PL matches on 2026-11-15 (international break).
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               return_value=_FakeEspnResponse({"events": []})):
        r = client.get("/fixtures", params={"league": "PL", "date": "2026-11-15"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fixtures"] == []
    assert body["message"] == "No PL matches on 2026-11-15."


def test_fixtures_espn_lists_real_fixture_with_prediction(client):
    """ESPN's fixture is listed with the model's prediction, and
    ESPN names are resolved to the model's canonical names."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    payload = {"events": [_espn_event(
        "2026-11-21T15:00Z", "pre", "Arsenal", "Newcastle United")]}
    with patch("football_agent.data.providers.requests.get",
               return_value=_FakeEspnResponse(payload)):
        r = client.get("/fixtures", params={"league": "PL", "date": "2026-11-21"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["fixtures"]) == 1
    fx = body["fixtures"][0]
    assert fx["home_team"] == "Arsenal"
    assert fx["away_team"] == "Newcastle"  # resolved, not "Newcastle United"
    assert fx["prediction"] is not None
    assert fx["prediction"]["match_pick"]["outcome"] == "home"


def test_fixtures_espn_failure_falls_back_to_sample_message(client):
    """If ESPN cannot be reached, the old sample-data message returns."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()

    def failing(url, timeout=None):
        raise ConnectionError("down")

    with patch("football_agent.data.providers.requests.get",
               side_effect=failing):
        r = client.get("/fixtures", params={"league": "PL", "date": "2026-11-15"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fixtures"] == []
    assert "FOOTBALL_DATA_API_KEY" in body["message"]


def test_fixtures_team_filter_espn(client):
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    payload = {"events": [
        _espn_event("2026-11-21T15:00Z", "pre", "Arsenal", "Newcastle United"),
        _espn_event("2026-11-21T17:30Z", "pre", "Chelsea", "Liverpool"),
    ]}
    with patch("football_agent.data.providers.requests.get",
               return_value=_FakeEspnResponse(payload)):
        r = client.get("/fixtures", params={
            "league": "PL", "team": "Arsenal", "date": "2026-11-21"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["fixtures"]) == 1
    assert body["fixtures"][0]["home_team"] == "Arsenal"


def test_today_lists_nations_league_matches(client):
    """Mon 16 Nov 2026: no club football, but 10 Nations League
    matches — /today lists the real ones with predictions."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/today", params={"date": "2026-11-16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["date"] == "2026-11-16"
    group = next(g for g in body["groups"] if g["league"] == "NL")
    assert group["label"] == "UEFA Nations League"
    assert len(group["fixtures"]) == 2  # finished match excluded
    fx = group["fixtures"][0]
    assert fx["home_team"] == "Iceland" and fx["away_team"] == "Luxembourg"
    assert fx["prediction"] is not None
    assert fx["prediction"]["match_pick"] is not None
    assert fx["prediction"]["league"] == "NL"


def test_fixtures_nations_league(client):
    """The Nations League is a real league in /fixtures, not an error."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/fixtures", params={"league": "NL", "date": "2026-11-16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "NL"
    assert len(body["fixtures"]) == 2
    assert body["fixtures"][0]["home_team"] == "Iceland"
    assert body["fixtures"][0]["prediction"] is not None


def test_ratings_and_team_nations_league(client):
    r = client.get("/ratings", params={"league": "NL"})
    assert r.status_code == 200, r.text
    teams = [row["team"] for row in r.json()["ratings"]]
    assert "Iceland" in teams and "Germany" in teams
    r = client.get("/team", params={"team": "Iceland", "league": "NL"})
    assert r.status_code == 200, r.text
    assert r.json()["team"] == "Iceland"
    assert r.json()["league"] == "NL"


def test_today_searches_all_leagues(client, tmp_path, monkeypatch):
    """Today's prediction searches every league, not just the top
    ones: on Mon 16 Nov 2026 the Eredivisie and Championship also
    have a match. They have no prediction data yet (no key), so they
    are listed as unavailable instead of silently dropped. A Liga MX
    match is predicted too — its history comes from ESPN, not the
    football-data.org key."""
    from football_agent.data.providers import clear_espn_cache
    monkeypatch.setattr("football_agent.data.providers.CACHE_DIR",
                        tmp_path)
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/today", params={"date": "2026-11-16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [g["league"] for g in body["groups"]] == ["PL", "MX1", "NL"]
    pl = body["groups"][0]
    # The phantom Coventry fixture is dropped, not predicted as
    # Man City: only the real Arsenal vs Chelsea is listed.
    assert len(pl["fixtures"]) == 1
    assert pl["fixtures"][0]["home_team"] == "Arsenal"
    assert pl["fixtures"][0]["away_team"] == "Chelsea"
    assert pl["fixtures"][0]["prediction"] is not None
    mx1 = body["groups"][1]
    assert mx1["label"] == "Liga MX"
    assert len(mx1["fixtures"]) == 1
    assert mx1["fixtures"][0]["home_team"] == "Pachuca"
    assert mx1["fixtures"][0]["away_team"] == "Toluca"
    assert mx1["fixtures"][0]["prediction"] is not None
    assert mx1["fixtures"][0]["prediction"]["league"] == "MX1"
    assert [u["league"] for u in body["unavailable"]] == ["ELC", "DED"]
    assert body["unavailable"][0] == {
        "league": "ELC", "label": "Championship", "fixtures": 1}
    assert body["unavailable"][1] == {
        "league": "DED", "label": "Eredivisie", "fixtures": 1}


def test_today_includes_world_cup(client):
    """A World Cup match day is searched too, and predicted with the
    international pool."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/today", params={"date": "2026-07-19"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [g["league"] for g in body["groups"]] == ["WC"]
    assert body["unavailable"] == []
    fx = body["groups"][0]["fixtures"][0]
    assert fx["home_team"] == "Brazil" and fx["away_team"] == "Argentina"
    assert fx["prediction"] is not None


def test_fixtures_world_cup(client):
    """The World Cup is a real league in /fixtures, via its own ESPN
    feed — not an error, and not Nations League matches."""
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_espn_get):
        r = client.get("/fixtures", params={"league": "WC", "date": "2026-07-19"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "WC"
    assert len(body["fixtures"]) == 1
    assert body["fixtures"][0]["home_team"] == "Brazil"
    assert body["fixtures"][0]["prediction"] is not None


def test_today_no_matches_message(client):
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               return_value=_FakeEspnResponse({"events": []})):
        r = client.get("/today", params={"date": "2026-11-16"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == []
    assert body["unavailable"] == []
    assert body["message"].startswith("No matches on 2026-11-16")


def test_today_invalid_date_400(client):
    r = client.get("/today", params={"date": "16-11-2026"})
    assert r.status_code == 400, r.text


def test_fixtures_team_filter_espn_no_date(client):
    # No date: ESPN's next-10-days scan. The fake answers only on
    # 2026-11-21, five days after the fixed "today".
    from datetime import date as _date
    from football_agent.data.providers import clear_espn_cache
    clear_espn_cache()

    class _FixedDate(_date):
        @classmethod
        def today(cls):
            return _date(2026, 11, 16)

    payload = {"events": [_espn_event(
        "2026-11-21T15:00Z", "pre", "Arsenal", "Newcastle United")]}

    def dated(url, timeout=None):
        if "dates=20261121" in url:
            return _FakeEspnResponse(payload)
        return _FakeEspnResponse({"events": []})

    with (patch("football_agent.data.providers.requests.get",
                side_effect=dated),
          patch("football_agent.data.providers.date", _FixedDate)):
        r = client.get("/fixtures", params={"league": "PL", "team": "Arsenal"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["fixtures"]) == 1
    assert body["fixtures"][0]["home_team"] == "Arsenal"
    assert body["fixtures"][0]["away_team"] == "Newcastle"


def test_fixtures_unknown_team_400(client):
    with patch("football_agent.data.providers.requests.get",
               return_value=_FakeEspnResponse({"events": []})):
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
    # Each row carries the data behind the ranking, not just a number.
    for row in body["ratings"]:
        assert isinstance(row["form"], list) and len(row["form"]) <= 5
        assert all(letter in "WDL" for letter in row["form"])
        assert isinstance(row["points_per_game"], (int, float))


def test_ratings_nations_league_has_form(client):
    r = client.get("/ratings", params={"league": "NL"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["league"] == "NL"
    top = body["ratings"][0]
    assert top["team"] == "Spain"
    assert isinstance(top["form"], list) and len(top["form"]) == 5
    assert top["points_per_game"] > 0


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
