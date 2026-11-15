"""CLI: football-predict predict | fixtures | backtest | evaluate | fetch

Works offline by default (bundled sample data). With FOOTBALL_DATA_API_KEY
set, the live football-data.org provider is used instead.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_agent.agent import run_agent  # noqa: E402
from football_agent.backtest import walk_forward  # noqa: E402
from football_agent.data.providers import (  # noqa: E402
    FootballDataOrgProvider, SampleProvider, get_provider)
from football_agent.model.predict import Predictor  # noqa: E402


def _provider(league: str):
    return get_provider(league)


def cmd_predict(args: argparse.Namespace) -> int:
    provider = _provider(args.league)
    matches = provider.matches(args.league)
    predictor = Predictor().fit(matches)
    pred = predictor.predict(args.home, args.away)
    print(f"{pred.home_team} vs {pred.away_team}")
    print(f"  Home win: {pred.p_home:.1%}   Draw: {pred.p_draw:.1%}   Away win: {pred.p_away:.1%}")
    print(f"  Expected goals: {pred.xg_home:.2f} - {pred.xg_away:.2f}")
    print(f"  Most likely: {pred.most_likely} (confidence {pred.confidence:.0%})")
    print("  Top scorelines: " + ", ".join(f"{s} ({p:.1%})" for s, p in pred.top_scorelines))
    print(f"  BTTS: {pred.p_btts:.1%}   Over 2.5 goals: {pred.p_over_25:.1%}")
    print("  Goals totals: " + "  ".join(
        f"Over {line}: {sides['over']:.1%} / Under {line}: {sides['under']:.1%}"
        for line, sides in pred.totals.items()))
    match_label = {"home": f"{pred.home_team} win",
                   "away": f"{pred.away_team} win",
                   "draw": "Draw"}[pred.match_pick["outcome"]]
    print(f"  Pick: {match_label} ({pred.match_pick['probability']:.1%})")
    print(f"  Pick: Both teams to score — "
          f"{'Yes' if pred.btts_pick['side'] == 'yes' else 'No'} "
          f"({pred.btts_pick['probability']:.1%})")
    print(f"  Pick: {pred.top_pick['market']} goals "
          f"({pred.top_pick['probability']:.1%})")
    print("  Key factors:")
    for f in pred.factors:
        print(f"    - {f}")
    return 0


def cmd_fixtures(args: argparse.Namespace) -> int:
    provider = _provider(args.league)
    on = date.fromisoformat(args.date) if args.date else None
    fixtures = provider.fixtures(args.league, on)
    if args.date and not fixtures:
        # Sample data has no future fixtures; live data needs an API key.
        matches = provider.matches(args.league)
        last = sorted({m.date for m in matches})[-1]
        print(f"No fixtures on {args.date}. Sample data ends {last}; "
              "set FOOTBALL_DATA_API_KEY for real upcoming fixtures.")
        return 0
    predictor = Predictor().fit(provider.matches(args.league))
    for fx in fixtures:
        p = predictor.predict(fx.home_team, fx.away_team, on=fx.date)
        print(f"{fx.date}  {fx.home_team} vs {fx.away_team}: "
              f"H {p.p_home:.0%} / D {p.p_draw:.0%} / A {p.p_away:.0%}  "
              f"pick {p.most_likely} ({p.confidence:.0%})  xG {p.xg_home:.2f}-{p.xg_away:.2f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    provider = _provider(args.league)
    result = walk_forward(provider.matches(args.league))
    print(result.table())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Alias of backtest with a plain-language verdict."""
    rc = cmd_backtest(args)
    print("\nEvaluation: accuracy above ~45% and Brier below ~0.60 on 1X2 beats "
          "always picking the home team (~47% accuracy here is mostly the home "
          "bias) — judge the model on Brier/log-loss, not raw accuracy alone.")
    return rc


def cmd_fetch(args: argparse.Namespace) -> int:
    provider = FootballDataOrgProvider()  # raises if no key
    matches = provider.matches(args.league)
    print(f"Fetched {len(matches)} finished {args.league} matches from football-data.org "
          f"(cached in data/cache/).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="football-predict", description="Football predictions agent")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("predict", help="Predict one match")
    pr.add_argument("--home", required=True)
    pr.add_argument("--away", required=True)
    pr.add_argument("--league", default="PL")
    pr.set_defaults(func=cmd_predict)

    fx = sub.add_parser("fixtures", help="List/predict fixtures")
    fx.add_argument("--league", default="PL")
    fx.add_argument("--date", default=None, help="YYYY-MM-DD")
    fx.set_defaults(func=cmd_fixtures)

    bt = sub.add_parser("backtest", help="Walk-forward backtest on history")
    bt.add_argument("--league", default="PL")
    bt.set_defaults(func=cmd_backtest)

    ev = sub.add_parser("evaluate", help="Backtest with a plain-language verdict")
    ev.add_argument("--league", default="PL")
    ev.set_defaults(func=cmd_evaluate)

    ft = sub.add_parser("fetch", help="Fetch live data (needs FOOTBALL_DATA_API_KEY)")
    ft.add_argument("--league", default="PL")
    ft.set_defaults(func=cmd_fetch)

    ag = sub.add_parser("agent", help="Rank fixtures and write predictions.csv/.md")
    ag.add_argument("--league", default="PL")
    ag.add_argument("--home", nargs="*", default=[], help="Home teams (paired with --away)")
    ag.add_argument("--away", nargs="*", default=[])
    ag.add_argument("--out-dir", default="output")
    ag.set_defaults(func=cmd_agent)
    return p


def cmd_agent(args: argparse.Namespace) -> int:
    from football_agent.data.types import Match
    provider = _provider(args.league)
    matches = provider.matches(args.league)
    if args.home and args.away and len(args.home) == len(args.away):
        fixtures = [Match(date=date.today(), season="", league=args.league,
                          home_team=h, away_team=a) for h, a in zip(args.home, args.away)]
    else:
        # Default demo fixtures: next-round style pairings from sample teams.
        teams = sorted({m.home_team for m in matches})
        fixtures = [Match(date=date.today(), season="", league=args.league,
                          home_team=teams[i], away_team=teams[-1 - i])
                    for i in range(5)]
    csv_path, md_path = run_agent(matches, fixtures, out_dir=args.out_dir)
    print(f"Wrote {csv_path} and {md_path} ({len(fixtures)} fixtures ranked).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
