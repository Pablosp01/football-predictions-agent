"""Agent loop: rank a fixture list by edge/confidence and write outputs."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .data.types import Match
from .model.predict import Predictor


def rank_fixtures(predictor: Predictor, fixtures: list[Match], on: date | None = None) -> list[dict]:
    """Predict each fixture and rank by confidence, then by edge over 1/3."""
    out = []
    for fx in fixtures:
        p = predictor.predict(fx.home_team, fx.away_team, on=on or fx.date)
        edge = p.confidence - (1 / 3)
        out.append({"fixture": fx, "prediction": p, "edge": edge})
    out.sort(key=lambda r: (r["prediction"].confidence, r["edge"]), reverse=True)
    return out


def run_agent(matches: list[Match], fixtures: list[Match],
              out_dir: str | Path = "output") -> tuple[Path, Path]:
    """Fit on history, predict fixtures, write predictions.csv + predictions.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor().fit(matches)
    ranked = rank_fixtures(predictor, fixtures)
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                    "xg_home", "xg_away", "most_likely", "confidence", "p_btts", "p_over_25"])
        for r in ranked:
            p, fx = r["prediction"], r["fixture"]
            w.writerow([fx.date.isoformat(), fx.home_team, fx.away_team,
                        f"{p.p_home:.3f}", f"{p.p_draw:.3f}", f"{p.p_away:.3f}",
                        f"{p.xg_home:.2f}", f"{p.xg_away:.2f}", p.most_likely,
                        f"{p.confidence:.3f}", f"{p.p_btts:.3f}", f"{p.p_over_25:.3f}"])
    md_path = out_dir / "predictions.md"
    lines = ["# Football Predictions", "",
             f"Ranked by confidence. Model: Elo (K=20, H=60) + Dixon-Coles Poisson, "
             f"form / H2H / rest adjustments. Not betting advice.", ""]
    for r in ranked:
        p, fx = r["prediction"], r["fixture"]
        lines.append(f"## {fx.date.isoformat()} — {fx.home_team} vs {fx.away_team}")
        lines.append(f"**Pick: {p.most_likely}** — H {p.p_home:.0%} / D {p.p_draw:.0%} / "
                     f"A {p.p_away:.0%} (confidence {p.confidence:.0%})")
        lines.append(f"xG {p.xg_home:.2f} – {p.xg_away:.2f} · BTTS {p.p_btts:.0%} · "
                     f"Over 2.5 {p.p_over_25:.0%}")
        lines.append("Most likely scorelines: " +
                     ", ".join(f"{s} ({prob:.0%})" for s, prob in p.top_scorelines))
        for f in p.factors:
            lines.append(f"- {f}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
