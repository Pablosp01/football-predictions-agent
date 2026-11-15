"""Plain-language verdict: the model must TELL the pick in words.

Every prediction states three picks as sentences — the match winner,
BTTS yes/no, and the goals total — each starting with "Pick: " and
carrying its probability in brackets. btts_pick is the more likely
BTTS side (ties go to "yes").

The 1X2 percentages are displayed with largest-remainder rounding
(tenths1x2 in the page): rounding each probability on its own can
add up to 100.1%, so the three displayed values always sum to
exactly 100.0% at one decimal.
"""
import json
import os
import re
import subprocess
from pathlib import Path

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


# -- 1X2 rounding ------------------------------------------------------------

def _run_tenths1x2(cases):
    """Run the page's tenths1x2 helper under node; return its results.

    The helper is extracted from index.html so the test exercises the
    exact code the browser runs, not a Python reimplementation.
    """
    html = (Path(__file__).resolve().parents[1]
            / "src" / "football_agent" / "static" / "index.html").read_text()
    match = re.search(r"function tenths1x2\(ph, pd, pa\) \{.*?\n  \}\n",
                      html, re.S)
    assert match, "tenths1x2 helper missing from the page"
    driver = match.group(0) + (
        "console.log(JSON.stringify("
        "CASES.map(function (c) { return tenths1x2(c[0], c[1], c[2]); })));"
    ).replace("CASES", json.dumps(cases))
    out = subprocess.run(["node", "-e", driver],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_1x2_displayed_percentages_sum_to_100():
    cases = [
        # Arsenal vs Chelsea, as reported live: independent rounding
        # gave 47.6 + 27.4 + 25.1 = 100.1.
        [0.476, 0.2735, 0.2505],
        [1 / 3, 1 / 3, 1 / 3],          # thirds
        [0.5, 0.25, 0.25],
        [0.333, 0.333, 0.334],
        [0.1, 0.2, 0.7],
        [0.999, 0.0005, 0.0005],
        [0.5177, 0.2896, 0.1927],       # Argentina vs Brazil (WC)
    ]
    results = _run_tenths1x2(cases)
    assert len(results) == len(cases)
    for tenths in results:
        assert len(tenths) == 3
        assert all(isinstance(t, int) for t in tenths)
        assert sum(tenths) == 1000, tenths  # exactly 100.0% at one decimal
    # The Arsenal case: the away side drops from 25.1 to 25.0.
    assert results[0] == [476, 274, 250]
    # Thirds split the extra tenth to the home side (tie-break order).
    assert results[1] == [334, 333, 333]


def test_render_prediction_uses_largest_remainder(client):
    html = client.get("/").text
    assert "tenths1x2(d.p_home, d.p_draw, d.p_away)" in html
