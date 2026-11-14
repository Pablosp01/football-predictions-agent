import math

import numpy as np

from football_agent.data.providers import SampleProvider
from football_agent.model.poisson import DixonColesModel, poisson_pmf


def test_poisson_pmf_sums_to_one():
    lam = 1.5
    total = sum(poisson_pmf(k, lam) for k in range(0, 40))
    assert abs(total - 1.0) < 1e-6


def test_score_matrix_sums_to_one_and_outcomes_sum_to_one():
    provider = SampleProvider()
    model = DixonColesModel().fit(provider.matches())
    xh, xa = model.expected_goals("Arsenal", "Chelsea")
    assert xh > 0 and xa > 0
    matrix = model.score_matrix(xh, xa)
    assert abs(matrix.sum() - 1.0) < 1e-9
    p_home, p_draw, p_away = model.outcome_probs("Arsenal", "Chelsea")
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert all(0 < p < 1 for p in (p_home, p_draw, p_away))


def test_dixon_coles_rho_adjusts_low_scores():
    # tau(0,0) = 1 - rho, so rho > 0 shrinks P(0-0) vs independent Poisson.
    m = DixonColesModel(rho=0.1)
    base = m.score_matrix(1.5, 1.2)
    indep = poisson_pmf(0, 1.5) * poisson_pmf(0, 1.2)
    assert base[0, 0] < indep
    assert abs(base.sum() - 1.0) < 1e-9


def test_stronger_team_gets_higher_xg_at_home():
    model = DixonColesModel().fit(SampleProvider().matches())
    xh_strong, _ = model.expected_goals("Liverpool", "Southampton")
    xh_weak, _ = model.expected_goals("Southampton", "Liverpool")
    assert xh_strong > xh_weak
