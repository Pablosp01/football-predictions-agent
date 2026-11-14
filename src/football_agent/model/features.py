"""Form, head-to-head and rest-day features from match history.

All features are computed strictly from matches played *before* the
fixture date, so backtesting cannot leak future information.
"""
from __future__ import annotations

from datetime import date

from ..data.types import Match


def _team_matches(matches: list[Match], team: str, before: date, n: int | None = None) -> list[Match]:
    out = [m for m in matches
           if m.played and m.date < before and team in (m.home_team, m.away_team)]
    out.sort(key=lambda m: m.date)
    return out[-n:] if n else out


def form_points(matches: list[Match], team: str, before: date, n: int = 5) -> float:
    """Points per game over the last n matches (W=3, D=1). 1.5 = no data."""
    recent = _team_matches(matches, team, before, n)
    if not recent:
        return 1.5
    pts = 0
    for m in recent:
        if m.home_team == team:
            pts += 3 if m.home_goals > m.away_goals else (1 if m.home_goals == m.away_goals else 0)  # type: ignore[operator]
        else:
            pts += 3 if m.away_goals > m.home_goals else (1 if m.home_goals == m.away_goals else 0)  # type: ignore[operator]
    return pts / len(recent)


def goals_per_game(matches: list[Match], team: str, before: date, n: int = 5) -> tuple[float, float]:
    """(scored, conceded) per game over the last n matches."""
    recent = _team_matches(matches, team, before, n)
    if not recent:
        return 1.4, 1.4
    scored = conceded = 0
    for m in recent:
        if m.home_team == team:
            scored += m.home_goals or 0  # type: ignore[operator]
            conceded += m.away_goals or 0  # type: ignore[operator]
        else:
            scored += m.away_goals or 0  # type: ignore[operator]
            conceded += m.home_goals or 0  # type: ignore[operator]
    return scored / len(recent), conceded / len(recent)


def head_to_head(matches: list[Match], home: str, away: str, before: date, n: int = 5) -> float:
    """Home team's points per game vs this opponent (1.5 = no meetings)."""
    h2h = [m for m in matches if m.played and m.date < before
           and {m.home_team, m.away_team} == {home, away}]
    h2h.sort(key=lambda m: m.date)
    h2h = h2h[-n:]
    if not h2h:
        return 1.5
    pts = 0
    for m in h2h:
        if m.home_team == home:
            pts += 3 if m.home_goals > m.away_goals else (1 if m.home_goals == m.away_goals else 0)  # type: ignore[operator]
        else:
            pts += 3 if m.away_goals > m.home_goals else (1 if m.home_goals == m.away_goals else 0)  # type: ignore[operator]
    return pts / len(h2h)


def rest_days(matches: list[Match], team: str, before: date) -> int:
    """Days since the team's last match before `before` (7 = no recent match)."""
    past = _team_matches(matches, team, before)
    if not past:
        return 7
    return (before - past[-1].date).days
