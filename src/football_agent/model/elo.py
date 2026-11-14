"""Elo ratings with home advantage.

Standard Elo: expected score E = 1 / (1 + 10^(-(R_home + H - R_away)/400)),
update R += K * (actual - E), K=20, H=60 Elo points (~ home edge).
New teams start at 1500.
"""
from __future__ import annotations

from ..data.types import Match

K_FACTOR = 20.0
HOME_ADVANTAGE = 60.0
INITIAL_RATING = 1500.0


class EloRatings:
    def __init__(self, k: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, INITIAL_RATING)

    def expected_home_score(self, home: str, away: str) -> float:
        diff = self.rating(home) + self.home_advantage - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, match: Match) -> None:
        """Update ratings from a played match. Draw = 0.5, win = 1 / loss = 0."""
        if not match.played or match.result is None:
            return
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[match.result]
        expected = self.expected_home_score(match.home_team, match.away_team)
        delta = self.k * (actual - expected)
        self.ratings[match.home_team] = self.rating(match.home_team) + delta
        self.ratings[match.away_team] = self.rating(match.away_team) - delta

    def fit(self, matches: list[Match]) -> "EloRatings":
        for m in sorted(matches, key=lambda x: x.date):
            self.update(m)
        return self

    def win_probability(self, home: str, away: str) -> float:
        """Home win probability implied by Elo (draws folded in via expected score)."""
        return self.expected_home_score(home, away)
