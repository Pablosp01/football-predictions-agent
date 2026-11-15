"""Goals-total markets and model picks: totals math, top pick, match pick.

Runs against the bundled offline sample data (FOOTBALL_DATA_API_KEY empty).
Totals come from the Dixon-Coles scoreline matrix, so over + under must
sum to 1 for every line and the top pick must be the max of the six sides.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402
from football_agent.data.providers import SampleProvider  # noqa: E402
from football_agent.model.predict import (  # noqa: E402
    Predictor,
    match_pick_from_probs,
    top_pick_from_totals,
    totals_from_matrix,
)


@pytest.fixture()
def client():
    api.clear_cache()
    with TestClient(api.app) as c:
        yield c
    api.clear_cache()


def _fitted_predictor() -> Predictor:
    return Predictor().fit(SampleProvider().matches())


# -- totals math -----------------------------------------------------------

def test_totals_over_under_sum_to_one_and_in_range():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    assert set(pred.totals) == {"1.5", "2.5", "3.5"}
    for line, sides in pred.totals.items():
        assert set(sides) == {"over", "under"}
        assert sides["over"] + sides["under"] == pytest.approx(1.0, abs=1e-9)
        assert 0 <= sides["over"] <= 1 and 0 <= sides["under"] <= 1


def test_totals_over_decreases_with_higher_line():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    assert (pred.totals["1.5"]["over"] > pred.totals["2.5"]["over"]
            > pred.totals["3.5"]["over"])


def test_totals_match_matrix_directly():
    # A hand-built matrix: P(0-0)=0.5, P(1-0)=0.25, P(2-1)=0.25.
    matrix = np.zeros((11, 11))
    matrix[0, 0] = 0.5
    matrix[1, 0] = 0.25
    matrix[2, 1] = 0.25
    totals = totals_from_matrix(matrix)
    # Totals: 0 (0.5), 1 (0.25), 3 (0.25).
    assert totals["1.5"]["over"] == pytest.approx(0.25)   # only the 3-goal game
    assert totals["1.5"]["under"] == pytest.approx(0.75)
    assert totals["2.5"]["over"] == pytest.approx(0.25)
    assert totals["3.5"]["over"] == pytest.approx(0.0)
    assert totals["3.5"]["under"] == pytest.approx(1.0)


def test_p_over_25_matches_totals_line():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    assert pred.p_over_25 == pytest.approx(pred.totals["2.5"]["over"], abs=1e-9)


# -- top pick ---------------------------------------------------------------

def test_top_pick_is_max_of_six_sides():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    sides = []
    for line, s in pred.totals.items():
        sides.append((f"Over {line}", s["over"]))
        sides.append((f"Under {line}", s["under"]))
    best_market, best_prob = max(sides, key=lambda t: t[1])
    assert pred.top_pick["market"] == best_market
    assert pred.top_pick["probability"] == pytest.approx(best_prob, abs=1e-9)


def test_top_pick_tie_prefers_lower_line_then_over():
    totals = {
        "1.5": {"over": 0.80, "under": 0.20},
        "2.5": {"over": 0.50, "under": 0.50},
        "3.5": {"over": 0.20, "under": 0.80},  # ties Over 1.5
    }
    assert top_pick_from_totals(totals) == {"market": "Over 1.5",
                                            "probability": 0.80}
    # A three-way 0.50 tie on every line: the lowest line wins, Over first.
    tied = {line: {"over": 0.50, "under": 0.50}
            for line in ("1.5", "2.5", "3.5")}
    assert top_pick_from_totals(tied) == {"market": "Over 1.5",
                                          "probability": 0.50}


# -- match pick --------------------------------------------------------------

def test_match_pick_is_max_1x2_outcome():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    probs = {"home": pred.p_home, "draw": pred.p_draw, "away": pred.p_away}
    best = max(probs, key=probs.get)
    assert pred.match_pick["outcome"] == best
    assert pred.match_pick["probability"] == pytest.approx(probs[best], abs=1e-9)
    assert pred.match_pick["probability"] == pytest.approx(pred.confidence, abs=1e-9)


def test_match_pick_tie_prefers_home_then_draw():
    assert match_pick_from_probs(0.4, 0.4, 0.2) == {"outcome": "home",
                                                    "probability": 0.4}
    assert match_pick_from_probs(0.2, 0.4, 0.4) == {"outcome": "draw",
                                                    "probability": 0.4}


# -- API ---------------------------------------------------------------------

def test_predict_returns_totals_top_pick_match_pick(client):
    r = client.post("/predict", json={"home_team": "Arsenal",
                                      "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Legacy fields unchanged.
    assert body["p_btts"] > 0 and body["p_over_25"] > 0
    assert body["p_over_25"] == pytest.approx(body["totals"]["2.5"]["over"],
                                              abs=0.001)
    assert set(body["totals"]) == {"1.5", "2.5", "3.5"}
    for line, sides in body["totals"].items():
        assert sides["over"] + sides["under"] == pytest.approx(1.0, abs=0.002)
        assert 0 <= sides["over"] <= 1 and 0 <= sides["under"] <= 1
    # Top pick is the max of the six sides.
    sides = [(f"Over {line}", s["over"]) for line, s in body["totals"].items()]
    sides += [(f"Under {line}", s["under"]) for line, s in body["totals"].items()]
    best_market, best_prob = max(sides, key=lambda t: t[1])
    assert body["top_pick"]["market"] == best_market
    assert body["top_pick"]["probability"] == pytest.approx(best_prob, abs=0.001)
    # Match pick is the max 1X2 outcome.
    probs = {"home": body["p_home"], "draw": body["p_draw"],
             "away": body["p_away"]}
    best = max(probs, key=probs.get)
    assert body["match_pick"] == {"outcome": best,
                                  "probability": pytest.approx(probs[best],
                                                             abs=0.001)}
    assert body["match_pick"]["probability"] == pytest.approx(
        body["confidence"], abs=0.001)


def test_predict_batch_includes_new_fields(client):
    r = client.post("/predict-batch", json={"fixtures": [
        {"home_team": "Arsenal", "away_team": "Chelsea"},
        {"home_team": "Liverpool", "away_team": "Man City"},
    ]})
    assert r.status_code == 200, r.text
    for p in r.json()["predictions"]:
        assert set(p["totals"]) == {"1.5", "2.5", "3.5"}
        assert p["top_pick"]["market"] in (
            "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
            "Over 3.5", "Under 3.5")
        assert p["match_pick"]["outcome"] in ("home", "draw", "away")
        assert p["p_over_25"] == pytest.approx(p["totals"]["2.5"]["over"],
                                               abs=0.001)


# -- web page -----------------------------------------------------------------

def test_page_has_picks_section(client):
    html = client.get("/").text
    assert "Model’s picks" in html
    assert "Model’s pick" in html  # the badge on the top pick
    # The totals rows are built at runtime from the line list + "Over "/"Under ".
    assert '"1.5", "2.5", "3.5"' in html
    assert '"Over " + line' in html
    assert '"Under " + line' in html
    assert "Both teams to score" in html
    assert "renderPicks" in html
    # Disclaimer still present — picks are model probabilities, not advice.
    assert "not betting advice" in html
    assert "guaranteed win" not in html.lower()
