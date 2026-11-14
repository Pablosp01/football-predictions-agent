"""FastAPI wrapper around the Elo + Dixon-Coles model.

This is what ChatGPT talks to: a user types team names in ChatGPT, the
Custom GPT calls these endpoints, and the numbers that come back are the
model's numbers — ChatGPT must not invent or override them.

Run locally:
    uvicorn football_agent.api:app --reload
    python -m football_agent.api            # same, on $PORT (default 8000)
"""
from __future__ import annotations

import difflib
import os
import re
from datetime import date as _date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .backtest import walk_forward
from .data.providers import get_provider
from .model.predict import Predictor

MODEL_NAME = "Elo + Dixon-Coles Poisson"
LEAGUES = ["PL", "PD", "SA", "BL1", "FL1", "CL"]
DISCLAIMER = (
    "Statistical model output, not betting advice. No guarantee of profit; "
    "probabilities are estimates and football is noisy."
)

# ---------------------------------------------------------------------------
# Team-name normalisation
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Lowercase, strip punctuation/FC suffix, collapse whitespace."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    s = re.sub(r"\b(fc|football club|afc)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Keys are normalised; values are the canonical names used by the model.
ALIASES = {
    "man utd": "Man Utd", "man united": "Man Utd", "manchester united": "Man Utd",
    "manchester utd": "Man Utd", "united": "Man Utd",
    "man city": "Man City", "manchester city": "Man City", "city": "Man City",
    "tottenham": "Tottenham", "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "arsenal": "Arsenal", "the gunners": "Arsenal",
    "aston villa": "Aston Villa", "villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton", "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace", "palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich", "ipswich town": "Ipswich",
    "leeds": "Leeds", "leeds united": "Leeds",
    "leicester": "Leicester", "leicester city": "Leicester",
    "liverpool": "Liverpool",
    "newcastle": "Newcastle", "newcastle united": "Newcastle",
    "nottm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest",
    "forest": "Nott'm Forest",
    "southampton": "Southampton", "saints": "Southampton",
    "sunderland": "Sunderland",
    "west ham": "West Ham", "west ham united": "West Ham",
    "wolves": "Wolves", "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
}


def _known_teams(league: str) -> list[str]:
    provider = get_provider(league)
    teams = {m.home_team for m in provider.matches(league)}
    teams |= {m.away_team for m in provider.matches(league)}
    return sorted(teams)


def resolve_team(name: str, league: str) -> str:
    """Resolve a free-text team name to a canonical team, or raise 400."""
    known = _known_teams(league)
    by_norm = {_norm(t): t for t in known}
    key = _norm(name)
    if key in by_norm:
        return by_norm[key]
    if key in ALIASES and ALIASES[key] in known:
        return ALIASES[key]
    # Substring match either way ("Newcastle" in "Newcastle United", etc.)
    for normed, canonical in by_norm.items():
        if key and (key in normed or normed in key):
            return canonical
    for alias, canonical in ALIASES.items():
        if canonical in known and (key in alias or alias in key):
            return canonical
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.55)
    if not suggestions:
        suggestions = difflib.get_close_matches(key, [_norm(t) for t in known],
                                                n=3, cutoff=0.55)
        suggestions = [by_norm[s] for s in suggestions if s in by_norm]
    raise HTTPException(status_code=400, detail={
        "error": f"Unknown team: '{name}'",
        "did_you_mean": suggestions,
        "known_teams": known,
    })


# ---------------------------------------------------------------------------
# Cached fitted model (fit once per league, reuse for every request)
# ---------------------------------------------------------------------------

_predictors: dict[str, Predictor] = {}
_matches_cache: dict[str, list] = {}


def get_predictor(league: str) -> Predictor:
    if league not in _predictors:
        provider = get_provider(league)
        matches = provider.matches(league)
        if len(matches) < 10:
            raise HTTPException(status_code=503, detail={
                "error": f"Not enough historical matches to fit the model for league '{league}'",
                "matches_available": len(matches),
                "hint": ("Set FOOTBALL_DATA_API_KEY for live data; the bundled sample "
                         "data only covers the Premier League (PL)."),
            })
        _matches_cache[league] = matches
        _predictors[league] = Predictor().fit(matches)
    return _predictors[league]


def clear_cache() -> None:
    """For tests: drop cached predictors so a fresh fit happens."""
    _predictors.clear()
    _matches_cache.clear()


# ---------------------------------------------------------------------------
# Auth (optional): if API_KEY is set, every endpoint except /health requires
# an X-API-Key header with that value.
# ---------------------------------------------------------------------------

def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401,
                            detail="Missing or invalid X-API-Key header")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Chelsea"])
    league: str = Field(default="PL", examples=["PL"])


class FixtureInput(BaseModel):
    home_team: str
    away_team: str


class BatchRequest(BaseModel):
    fixtures: list[FixtureInput] = Field(..., min_length=1, max_length=50)
    league: str = Field(default="PL")


class PredictionOut(BaseModel):
    home_team: str
    away_team: str
    league: str
    p_home: float
    p_draw: float
    p_away: float
    xg_home: float
    xg_away: float
    most_likely: str
    confidence: float
    top_scorelines: list[dict]
    p_btts: float
    p_over_25: float
    factors: list[str]
    source: str = "model"
    disclaimer: str = DISCLAIMER


def _prediction_out(pred, league: str) -> PredictionOut:
    d = pred.to_dict()
    return PredictionOut(
        home_team=d["home_team"], away_team=d["away_team"], league=league,
        p_home=d["p_home"], p_draw=d["p_draw"], p_away=d["p_away"],
        xg_home=d["xg_home"], xg_away=d["xg_away"],
        most_likely=d["most_likely"], confidence=d["confidence"],
        top_scorelines=[{"score": s, "prob": p} for s, p in d["top_scorelines"]],
        p_btts=d["p_btts"], p_over_25=d["p_over_25"], factors=d["factors"],
    )


def _predict_one(home: str, away: str, league: str) -> PredictionOut:
    predictor = get_predictor(league)
    home_c = resolve_team(home, league)
    away_c = resolve_team(away, league)
    if home_c == away_c:
        raise HTTPException(status_code=400, detail={
            "error": "Home and away teams must be different",
            "home_team": home_c, "away_team": away_c,
        })
    return _prediction_out(predictor.predict(home_c, away_c), league)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up at startup, not on the first prediction: imports (pandas /
    # scikit-learn) and the PL model fit happen here, while Render is still
    # running its health checks. On Render's free tier the first request
    # after a cold start is already slow; without this, /predict would pay
    # the whole warm-up cost inside ChatGPT's action timeout.
    try:
        get_predictor("PL")
    except Exception:
        pass
    yield


app = FastAPI(
    title="Football Predictions Agent API",
    description=("Elo + Dixon-Coles Poisson predictions. The probabilities, xG and "
                 "scorelines returned here are the model's authoritative output."),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "leagues": LEAGUES}


@app.post("/predict", response_model=PredictionOut,
          dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictionOut:
    return _predict_one(req.home_team, req.away_team, req.league)


@app.post("/predict-batch", dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchRequest) -> dict:
    results = [_predict_one(f.home_team, f.away_team, req.league)
               for f in req.fixtures]
    results.sort(key=lambda r: r.confidence, reverse=True)
    ranked = [dict(r.model_dump(), rank=i + 1) for i, r in enumerate(results)]
    return {"league": req.league, "count": len(ranked),
            "predictions": ranked, "source": "model",
            "disclaimer": DISCLAIMER}


@app.get("/fixtures", dependencies=[Depends(require_api_key)])
def fixtures(league: str = "PL",
             date: str | None = Query(default=None, examples=["2026-11-15"])) -> dict:
    on = None
    if date:
        try:
            on = _date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail={
                "error": f"Invalid date '{date}': use YYYY-MM-DD"})
    provider = get_provider(league)
    days_fixtures = provider.fixtures(league, on)
    out: list[dict] = []
    predictor = None
    for fx in days_fixtures:
        try:
            predictor = predictor or get_predictor(league)
            pred = _prediction_out(predictor.predict(fx.home_team, fx.away_team,
                                                     on=fx.date), league)
            out.append({"date": fx.date.isoformat(), "home_team": fx.home_team,
                        "away_team": fx.away_team,
                        "prediction": pred.model_dump()})
        except HTTPException:
            out.append({"date": fx.date.isoformat(), "home_team": fx.home_team,
                        "away_team": fx.away_team, "prediction": None})
    if not out:
        if os.environ.get("FOOTBALL_DATA_API_KEY"):
            message = f"No fixtures scheduled on {date or 'that date'} for {league}."
        else:
            matches = _matches_cache.get(league) or provider.matches(league)
            last = max((m.date for m in matches), default=None)
            message = (f"No fixtures on {date or 'that date'}. The bundled sample data "
                       f"has no future fixtures (it ends {last}); set "
                       "FOOTBALL_DATA_API_KEY for real upcoming fixtures.")
        return {"league": league, "date": date, "fixtures": [],
                "message": message, "source": "model"}
    return {"league": league, "date": date, "fixtures": out, "source": "model"}


@app.get("/backtest", dependencies=[Depends(require_api_key)])
def backtest(league: str = "PL") -> dict:
    provider = get_provider(league)
    matches = provider.matches(league)
    try:
        result = walk_forward(matches)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc),
                            "matches_available": len(matches)})
    return {"league": league, "accuracy": round(result.accuracy, 4),
            "brier": round(result.brier, 4), "log_loss": round(result.log_loss, 4),
            "n_matches": result.n, "home_win_rate": round(result.home_win_rate, 4),
            "roi": (round(result.roi, 4) if result.roi is not None else None),
            "source": "model", "disclaimer": DISCLAIMER}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
