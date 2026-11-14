"""Core data types shared by providers, model and backtest."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Result = Literal["H", "D", "A"]


@dataclass(frozen=True)
class Match:
    date: date
    season: str
    league: str
    home_team: str
    away_team: str
    home_goals: int | None = None  # None => fixture not yet played
    away_goals: int | None = None
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> Result | None:
        if not self.played:
            return None
        if self.home_goals > self.away_goals:  # type: ignore[operator]
            return "H"
        if self.home_goals < self.away_goals:  # type: ignore[operator]
            return "A"
        return "D"
