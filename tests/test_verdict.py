"""Plain-language verdict: the model must TELL the pick in words.

Every prediction states three picks as sentences — the match winner,
BTTS yes/no, and the goals total — each starting with "Pick: " and
carrying its probability in brackets. btts_pick is the more likely
BTTS side (ties go to "yes").
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402
from football_agent.data.providers import SampleProvider  # noqa: E402
from football_agent.model.predict import (  # noqa: E402
    Predictor,
    btts_pick_from_prob,
)


@pytest.fixture()
def client():
    api.clear_cache()
    with TestClient(api.app) as c:
        yield c
    api.clear_cache()


def _fitted_predictor() -> Predictor:
    return Predictor().fit(SampleProvider().matches())


# -- btts_pick ---------------------------------------------------------------

def test_btts_pick_yes_when_likely():
    assert btts_pick_from_prob(0.7) == {"side": "yes", "probability": 0.7}
    assert btts_pick_from_prob(0.9) == {"side": "yes", "probability": 0.9}


def test_btts_pick_no_when_unlikely():
    assert btts_pick_from_prob(0.3) == {"side": "no", "probability": 0.7}
    assert btts_pick_from_prob(0.1) == {"side": "no", "probability": 0.9}


def test_btts_pick_tie_goes_to_yes():
    assert btts_pick_from_prob(0.5) == {"side": "yes", "probability": 0.5}


def test_btts_pick_matches_p_btts_on_real_prediction():
    pred = _fitted_predictor().predict("Arsenal", "Chelsea")
    if pred.p_btts >= 0.5:
        assert pred.btts_pick == {"side": "yes",
                                  "probability": pytest.approx(pred.p_btts)}
    else:
        assert pred.btts_pick == {"side": "no",
                                  "probability": pytest.approx(1 - pred.p_btts)}


# -- API ---------------------------------------------------------------------

def test_predict_returns_btts_pick(client):
    r = client.post("/predict", json={"home_team": "Arsenal",
                                      "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Arsenal vs Chelsea: BTTS is the likelier side.
    assert body["btts_pick"] == {"side": "yes",
                                 "probability": pytest.approx(body["p_btts"],
                                                            abs=0.001)}
    assert body["btts_pick"]["probability"] == pytest.approx(0.525, abs=0.01)
    # The three picks are all present and consistent.
    assert body["match_pick"]["outcome"] == "home"
    assert body["top_pick"]["market"] == "Over 1.5"


def test_predict_batch_returns_btts_pick(client):
    r = client.post("/predict-batch", json={"fixtures": [
        {"home_team": "Arsenal", "away_team": "Chelsea"},
        {"home_team": "Liverpool", "away_team": "Man City"},
    ]})
    assert r.status_code == 200, r.text
    for p in r.json()["predictions"]:
        assert p["btts_pick"]["side"] in ("yes", "no")
        expected = p["p_btts"] if p["p_btts"] >= 0.5 else 1 - p["p_btts"]
        assert p["btts_pick"]["probability"] == pytest.approx(expected,
                                                             abs=0.001)


# -- web page ------------------------------------------------------------------

def test_page_has_pick_verdict_sentences(client):
    html = client.get("/").text
    # The verdict lines are built at runtime by renderPicks; the page
    # must carry the "Pick: " prefix and the three sentence shapes.
    assert '"Pick: "' in html
    assert "Both teams to score — " in html
    assert '" goals</b> ("' in html
    # Supporting detail and disclaimer are still there.
    assert "Model’s picks" in html
    assert "Both teams to score" in html
    assert "Statistical model output, not betting advice" in html
