from datetime import date

from football_agent.data.types import Match
from football_agent.model.elo import EloRatings

M = Match(date=date(2025, 1, 1), season="2024-25", league="PL",
          home_team="Arsenal", away_team="Chelsea", home_goals=2, away_goals=0)


def test_new_teams_start_at_1500():
    elo = EloRatings()
    assert elo.rating("Arsenal") == 1500.0
    assert elo.rating("Unknown FC") == 1500.0


def test_home_win_raises_home_rating_and_lowers_away():
    elo = EloRatings()
    elo.update(M)
    assert elo.rating("Arsenal") > 1500.0
    assert elo.rating("Chelsea") < 1500.0
    # Zero-sum update
    assert elo.rating("Arsenal") + elo.rating("Chelsea") == 3000.0


def test_draw_moves_ratings_towards_each_other():
    elo = EloRatings()
    elo.ratings["Arsenal"] = 1600.0
    elo.ratings["Chelsea"] = 1400.0
    draw = Match(date=date(2025, 1, 2), season="2024-25", league="PL",
                 home_team="Arsenal", away_team="Chelsea", home_goals=1, away_goals=1)
    elo.update(draw)
    assert elo.rating("Arsenal") < 1600.0
    assert elo.rating("Chelsea") > 1400.0


def test_expected_home_score_above_half_for_equal_teams():
    elo = EloRatings()
    assert elo.expected_home_score("Arsenal", "Chelsea") > 0.5  # home advantage
