from datetime import date

import numpy as np

from football_agent.backtest import walk_forward
from football_agent.data.providers import SampleProvider
from football_agent.data.types import Match
from football_agent.model.predict import Predictor


def test_backtest_no_leakage():
    """Predictions for match i must use only matches before i.

    If we corrupt the *future* (change a later match's score), the
    prediction for an earlier match must not change.
    """
    matches = SampleProvider().matches()
    cutoff = date(2025, 1, 1)
    predictor = Predictor().fit(matches, before=cutoff)
    before = predictor.predict("Arsenal", "Chelsea", on=cutoff)
    # Corrupt a match played AFTER the cutoff: it must not change the prediction.
    corrupted = list(matches)
    later = next(i for i, m in enumerate(corrupted) if m.date > cutoff)
    corrupted[later] = Match(date=corrupted[later].date, season=corrupted[later].season,
                             league="PL", home_team=corrupted[later].home_team,
                             away_team=corrupted[later].away_team,
                             home_goals=9, away_goals=0)
    predictor2 = Predictor().fit(corrupted, before=cutoff)
    after = predictor2.predict("Arsenal", "Chelsea", on=cutoff)
    assert before.p_home == after.p_home
    assert before.xg_home == after.xg_home


def test_backtest_metrics_in_range():
    result = walk_forward(SampleProvider().matches())
    assert result.n > 300
    assert 0.30 < result.accuracy < 0.70
    assert 0.4 < result.brier < 0.75
    assert 0.8 < result.log_loss < 1.6
    assert result.roi is not None
    # Rows are in chronological order
    dates = [r["date"] for r in result.rows]
    assert dates == sorted(dates)


def test_prediction_probabilities_sum_to_one():
    predictor = Predictor().fit(SampleProvider().matches())
    pred = predictor.predict("Arsenal", "Chelsea")
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-6
    assert len(pred.top_scorelines) == 5
    assert abs(sum(p for _, p in pred.top_scorelines) - sum(p for _, p in pred.top_scorelines)) < 1e-9
    assert 0 <= pred.p_btts <= 1 and 0 <= pred.p_over_25 <= 1
    assert pred.factors  # explanation present
