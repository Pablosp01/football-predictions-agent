"""Walk-forward backtest: expanding window, no look-ahead.

For each match (after a warm-up of `min_history` matches), fit the model
only on matches played before it, predict it, then move on. This mirrors
real use: no future information ever enters a prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.types import Match
from .model.predict import Predictor


@dataclass
class BacktestResult:
    n: int
    accuracy: float
    brier: float
    log_loss: float
    home_win_rate: float
    roi: float | None  # flat 1-unit stakes on most-likely outcome, decimal odds
    rows: list[dict]

    def table(self) -> str:
        lines = [
            f"Matches backtested : {self.n}",
            f"Accuracy (1X2)     : {self.accuracy:.3f}",
            f"Brier score        : {self.brier:.4f}  (lower is better; 0.667 = always 1/3)",
            f"Log loss           : {self.log_loss:.4f}  (lower is better)",
            f"Home win rate      : {self.home_win_rate:.3f}",
        ]
        if self.roi is not None:
            lines.append(f"ROI vs odds        : {self.roi:+.1%}  (flat 1u on most-likely outcome)")
        else:
            lines.append("ROI vs odds        : n/a (no odds in data)")
        return "\n".join(lines)


def _brier(probs: np.ndarray, outcome: int) -> float:
    one_hot = np.zeros(3)
    one_hot[outcome] = 1.0
    return float(np.sum((probs - one_hot) ** 2))


def walk_forward(matches: list[Match], min_history: int = 100,
                 refit_every: int = 20) -> BacktestResult:
    played = sorted([m for m in matches if m.played], key=lambda m: m.date)
    if len(played) <= min_history:
        raise ValueError(f"Need > {min_history} played matches, got {len(played)}")
    rows: list[dict] = []
    predictor: Predictor | None = None
    next_refit = min_history
    for i in range(min_history, len(played)):
        if predictor is None or i >= next_refit:
            predictor = Predictor().fit(played[:i])
            next_refit = i + refit_every
        m = played[i]
        pred = predictor.predict(m.home_team, m.away_team, on=m.date)  # type: ignore[union-attr]
        probs = np.array([pred.p_home, pred.p_draw, pred.p_away])
        outcome = {"H": 0, "D": 1, "A": 2}[m.result]  # type: ignore[index]
        roi_row: float | None = None
        if m.home_odds and m.draw_odds and m.away_odds:
            odds = [m.home_odds, m.draw_odds, m.away_odds]
            pick = int(np.argmax(probs))
            roi_row = odds[pick] - 1.0 if pick == outcome else -1.0
        rows.append({
            "date": m.date.isoformat(), "home_team": m.home_team, "away_team": m.away_team,
            "p_home": pred.p_home, "p_draw": pred.p_draw, "p_away": pred.p_away,
            "predicted": ["H", "D", "A"][int(np.argmax(probs))],
            "actual": m.result, "brier": _brier(probs, outcome),
            "log_loss": -float(np.log(max(probs[outcome], 1e-9))),
            "roi": roi_row,
        })
    acc = float(np.mean([r["predicted"] == r["actual"] for r in rows]))
    brier = float(np.mean([r["brier"] for r in rows]))
    ll = float(np.mean([r["log_loss"] for r in rows]))
    hw = float(np.mean([r["actual"] == "H" for r in rows]))
    rois = [r["roi"] for r in rows if r["roi"] is not None]
    roi = float(np.mean(rois)) if rois else None
    return BacktestResult(len(rows), acc, brier, ll, hw, roi, rows)
