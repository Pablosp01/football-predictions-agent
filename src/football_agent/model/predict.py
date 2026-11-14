"""Prediction: blend Dixon-Coles Poisson + Elo + form/H2H/rest adjustments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..data.types import Match
from .calibration import Calibrator
from .elo import EloRatings
from .features import form_points, goals_per_game, head_to_head, rest_days
from .poisson import DixonColesModel


@dataclass
class Prediction:
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    xg_home: float
    xg_away: float
    top_scorelines: list[tuple[str, float]]
    p_btts: float
    p_over_25: float
    confidence: float
    factors: list[str]

    @property
    def most_likely(self) -> str:
        return max(("H", "D", "A"), key=lambda r: {"H": self.p_home, "D": self.p_draw, "A": self.p_away}[r])

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team, "away_team": self.away_team,
            "p_home": round(self.p_home, 4), "p_draw": round(self.p_draw, 4),
            "p_away": round(self.p_away, 4), "xg_home": round(self.xg_home, 2),
            "xg_away": round(self.xg_away, 2),
            "most_likely": self.most_likely,
            "p_btts": round(self.p_btts, 4), "p_over_25": round(self.p_over_25, 4),
            "confidence": round(self.confidence, 4),
            "top_scorelines": [(s, round(p, 4)) for s, p in self.top_scorelines],
            "factors": self.factors,
        }


class Predictor:
    """Fits Elo + Dixon-Coles on history, then predicts fixtures."""

    def __init__(self) -> None:
        self.elo = EloRatings()
        self.dc = DixonColesModel()
        self.calibrator = Calibrator()
        self.history: list[Match] = []
        self.fitted = False

    def fit(self, matches: list[Match], before: date | None = None) -> "Predictor":
        """Fit only on matches played before `before` (default: all).

        Pass a cutoff when predicting a past date so no future match can
        leak into Elo / Dixon-Coles strengths. Walk-forward backtesting
        relies on this by fitting on played[:i] for each step.
        """
        self.history = sorted([m for m in matches if m.played
                               and (before is None or m.date < before)],
                              key=lambda m: m.date)
        self.elo.fit(self.history)
        self.dc.fit(self.history)
        self.fitted = True
        return self

    def _adjust_xg(self, home: str, away: str, on: date) -> tuple[float, float, list[str]]:
        xh, xa = self.dc.expected_goals(home, away)
        factors: list[str] = []
        fh, fa = form_points(self.history, home, on), form_points(self.history, away, on)
        form_diff = fh - fa
        if abs(form_diff) >= 0.6:
            xh *= 1.0 + 0.10 * form_diff
            xa *= 1.0 - 0.10 * form_diff
            factors.append(f"Form: {home} {fh:.1f} pts/game vs {away} {fa:.1f} (last 5)")
        h2h = head_to_head(self.history, home, away, on)
        if h2h != 1.5:
            xh *= 1.0 + 0.05 * (h2h - 1.5)
            xa *= 1.0 - 0.05 * (h2h - 1.5)
            factors.append(f"Head-to-head: {home} {h2h:.1f} pts/game vs {away}")
        rh, ra = rest_days(self.history, home, on), rest_days(self.history, away, on)
        if rh <= 3 < ra:
            xa *= 1.05
            factors.append(f"Rest: {away} had {ra} days' rest vs {home}'s {rh}")
        elif ra <= 3 < rh:
            xh *= 1.05
            factors.append(f"Rest: {home} had {rh} days' rest vs {away}'s {ra}")
        elo_diff = self.elo.rating(home) - self.elo.rating(away)
        if abs(elo_diff) >= 50:
            stronger = home if elo_diff > 0 else away
            factors.append(f"Elo: {stronger} rated {abs(elo_diff):.0f} points higher")
        gh, ga = goals_per_game(self.history, home, on), goals_per_game(self.history, away, on)
        factors.append(f"Recent goals: {home} {gh[0]:.1f} scored / {gh[1]:.1f} conceded; "
                       f"{away} {ga[0]:.1f} / {ga[1]:.1f}")
        return max(xh, 0.05), max(xa, 0.05), factors

    def predict(self, home: str, away: str, on: date | None = None) -> Prediction:
        if not self.fitted:
            raise RuntimeError("Predictor must be fitted before predict()")
        on = on or (self.history[-1].date if self.history else date.today())
        xh, xa, factors = self._adjust_xg(home, away, on)
        matrix = self.dc.score_matrix(xh, xa)
        p_home = float(sum(matrix[h, a] for h in range(11) for a in range(11) if h > a))
        p_draw = float(sum(matrix[i, i] for i in range(11)))
        p_away = float(sum(matrix[h, a] for h in range(11) for a in range(11) if h < a))
        # Elo enters the calibration features below as an independent signal.
        elo_share = self.elo.win_probability(home, away)
        raw = np.array([p_home, p_draw, p_away])
        raw = raw / raw.sum()
        feats = np.array([raw[0], raw[1], raw[2], elo_share,
                          form_points(self.history, home, on) - form_points(self.history, away, on)])
        cal = self.calibrator.calibrate(feats)
        if cal is not None:
            raw = cal / cal.sum()
        p_home, p_draw, p_away = (float(v) for v in raw)
        flat = sorted(((f"{h}-{a}", float(matrix[h, a])) for h in range(7) for a in range(7)),
                      key=lambda t: t[1], reverse=True)[:5]
        p_btts = float(sum(matrix[h, a] for h in range(1, 11) for a in range(1, 11)))
        p_over = float(sum(matrix[h, a] for h in range(11) for a in range(11) if h + a > 2.5))
        confidence = float(max(p_home, p_draw, p_away))
        return Prediction(home, away, p_home, p_draw, p_away, xh, xa,
                          flat, p_btts, p_over, confidence, factors[:3])
