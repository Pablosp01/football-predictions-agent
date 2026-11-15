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
import json
import os
import re
import time
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from pathlib import Path
from typing import Annotated

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .backtest import walk_forward
from .data.providers import INTERNATIONAL_LEAGUES, get_provider
from .model.predict import Predictor

MODEL_NAME = "Elo + Dixon-Coles Poisson"
# Every football-data.org free-tier (TIER_ONE) competition.
LEAGUES = ["PL", "PD", "SA", "BL1", "FL1", "ELC", "DED", "PPL", "BSA",
           "CL", "WC", "EC"]
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
    # International teams (names as in the bundled international sample).
    "usa": "United States", "us": "United States", "america": "United States",
    "united states of america": "United States",
    "cote d'ivoire": "Ivory Coast", "c te d'ivoire": "Ivory Coast",
    "czechia": "Czech Republic", "ir iran": "Iran",
    "korea republic": "South Korea", "turkiye": "Turkey",
    "holland": "Netherlands",
}


def _require_known_league(league: str) -> str:
    """Normalise a league code, or raise a clear 400 for unknown ones."""
    code = (league or "PL").strip().upper()
    if code not in LEAGUES:
        raise HTTPException(status_code=400, detail={
            "error": f"Unknown league: '{league}'",
            "known_leagues": LEAGUES,
            "hint": ("Supported competitions are the football-data.org free tier: "
                     "PL, PD, SA, BL1, FL1, ELC, DED, PPL, BSA, CL, WC, EC."),
        })
    return code


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
    league = _require_known_league(league)
    if league not in _predictors:
        provider = get_provider(league)
        matches = provider.matches(league)
        if len(matches) < 10:
            raise HTTPException(status_code=503, detail={
                "error": f"Not enough historical matches to fit the model for league '{league}'",
                "matches_available": len(matches),
                "hint": ("Set FOOTBALL_DATA_API_KEY for live data; the bundled sample "
                         "data only covers the Premier League (PL) and the "
                         "international tournaments (WC, EC)."),
            })
        _matches_cache[league] = matches
        if league in INTERNATIONAL_LEAGUES:
            # National teams only, and mostly neutral venues: no Elo
            # home edge, and a rating pool that never mixes with clubs.
            _predictors[league] = Predictor(home_advantage=0.0).fit(matches)
        else:
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
    # Goals-total markets: {"1.5": {"over": p, "under": 1-p}, ...} for
    # lines 1.5 / 2.5 / 3.5, summed from the scoreline distribution.
    totals: dict
    # The model's single most confident totals call, e.g.
    # {"market": "Over 1.5", "probability": 0.83}.
    top_pick: dict
    # The most likely 1X2 outcome, e.g. {"outcome": "home", "probability": 0.476}.
    match_pick: dict
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
        totals=d["totals"], top_pick=d["top_pick"], match_pick=d["match_pick"],
    )


def _predict_one(home: str, away: str, league: str) -> PredictionOut:
    league = _require_known_league(league)
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
# AI match briefing (AgentRouter only, live web search)
#
# Context only: recent form, injuries/suspensions, team news found by a
# live web search. The model's numbers from /predict are authoritative
# and are never touched by this section.
#
# The briefing uses AgentRouter's built-in web search: the /v1/responses
# call carries "tools": [{"type": "web_search"}], so the model searches
# the web for the fixture and cites what it finds. The prompt tells it
# to use ONLY the search results, to cite every factual claim with a
# markdown source link, and never to invent players, results, or
# injuries. Sources are extracted from the answer's markdown links and
# from any urls in the response's web_search_call output items, and the
# response carries live_search=true. The web page says the briefing is a
# live web search and lists the cited sources below it.
#
# Needs one key on the server:
#   AGENTROUTER_API_KEY — free AgentRouter key, the only summariser.
# Without it /ai-brief answers 503 and the web page hides the briefing.
#
# API shape (verified 2026-11-15 against the live key):
#   POST https://agentrouter.org/v1/responses
#   headers: Authorization Bearer, User-Agent: claude-cli/2.1.220
#   (external, claude-desktop), anthropic-version: 2023-06-01
#   body: {"model": "gpt-5.6-sol", "input": ...,
#          "tools": [{"type": "web_search"}]}
#   response: {"output_text": "..."} (or output[].content[] text), with
#   web_search_call output items and url citations.
# The gateway rejects requests without the fingerprint headers (HTTP
# 401 "unauthorized client detected"), and its models do NOT support
# /v1/chat/completions (HTTP 400) — use /v1/responses only.
# The key is sent as a header, never in a URL, and never logged or
# returned.
#
# Timing: the search itself is slow — a live web-searched call took
# 57s against the real key, while plain calls answer in ~1s. The
# AgentRouter timeout is therefore 90s and the whole-briefing budget
# 95s; the page shows a loading state for that long, and the 15-minute
# cache makes repeat requests instant.
# ---------------------------------------------------------------------------

AGENTROUTER_MODEL = os.environ.get("AGENTROUTER_MODEL", "gpt-5.6-sol")
AGENTROUTER_RESPONSES_URL = "https://agentrouter.org/v1/responses"
# The gateway rejects clients without a supported harness fingerprint.
AGENTROUTER_FINGERPRINT_HEADERS = {
    "User-Agent": "claude-cli/2.1.220 (external, claude-desktop)",
    "anthropic-version": "2023-06-01",
}
AGENTROUTER_TIMEOUT_S = 90
BRIEF_TOTAL_BUDGET_S = 95  # wall-clock budget for the whole briefing
BRIEF_CACHE_TTL_S = 15 * 60  # save free-tier quota: one briefing per pair/15 min
BRIEF_CACHE_MAX = 200

_brief_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}


class _BriefError(Exception):
    """The AgentRouter call failed (HTTP error, timeout, bad answer)."""


_LEAGUE_NAMES = {
    "PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
    "BL1": "Bundesliga", "FL1": "Ligue 1", "ELC": "Championship",
    "DED": "Eredivisie", "PPL": "Primeira Liga",
    "BSA": "Campeonato Brasileiro Série A", "CL": "Champions League",
    "WC": "FIFA World Cup", "EC": "European Championship",
}


def _brief_prompt(home: str, away: str, league: str, today: str) -> str:
    league_name = _LEAGUE_NAMES.get(league.upper(), league)
    return (
        f"Today is {today}. The football match is {home} vs {away} "
        f"({league_name}).\n\n"
        "Use the web search tool to search the web for this fixture, "
        "then return a JSON object (no other text) with exactly these keys:\n"
        '{"summary": "2-3 sentence overview of the match context", '
        '"form": ["<team>: last-5 results, e.g. W W D L W, with scores if found"], '
        '"injuries": ["<player> (<team>) — injury or suspension detail, or empty if none found"], '
        '"news": ["any other relevant team news"]}\n\n'
        "Rules:\n"
        "- Use ONLY what you find in the web search results. If a section "
        "has no reliable information, use an empty list.\n"
        "- Cite every factual claim with a source link in the text, like "
        "[ESPN](https://www.espn.com/soccer/...).\n"
        "- Do NOT invent players, results, or injuries.\n"
        "- Do NOT predict a winner, scoreline, or probabilities. Do NOT mention odds.\n"
        "- Keep every item to one short sentence. No betting language."
    )


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _strip_md_links(text: str) -> str:
    """Turn markdown links into plain titles for display.

    The page escapes all AI text, so a literal "[title](url)" would show
    as-is; the URL itself is kept separately in the sources list.
    """
    return _MD_LINK_RE.sub(r"\1", text)


def _parse_briefing(text: str) -> dict:
    """Parse the model's JSON answer; fall back to plain text as summary."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            def _items(key: str) -> list[str]:
                return [_strip_md_links(str(x).strip())
                        for x in (data.get(key) or []) if str(x).strip()]
            return {
                "summary": _strip_md_links(str(data.get("summary") or "").strip()),
                "form": _items("form"),
                "injuries": _items("injuries"),
                "news": _items("news"),
            }
    except (ValueError, TypeError):
        pass
    return {"summary": _strip_md_links(text.strip()),
            "form": [], "injuries": [], "news": []}


def _extract_agentrouter_text(data: dict) -> str:
    """Return the answer text from an AgentRouter /v1/responses body.

    Primary shape: top-level "output_text". Fallback: output[] items
    with content[] chunks of type "output_text".
    """
    text = data.get("output_text") or ""
    if isinstance(text, str) and text.strip():
        return text.strip()
    texts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for chunk in item.get("content") or []:
            if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                if chunk.get("text"):
                    texts.append(str(chunk["text"]))
    return "\n".join(texts).strip()


def _extract_sources(text: str, data: dict) -> list[dict]:
    """Collect the briefing's live sources as [{title, url}, ...].

    Two origins, both deduped by URL and restricted to http(s):
    markdown links [title](url) in the answer text, and any url values
    found in the response body — which covers web_search_call output
    items and url_citation annotations, whatever their exact shape.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    def _add(title, url) -> None:
        if not url or not str(url).startswith(("http://", "https://")):
            return
        url = str(url).strip()
        if url in seen:
            return
        seen.add(url)
        title = (str(title).strip() if title else "") or url
        sources.append({"title": title, "url": url})

    for title, url in _MD_LINK_RE.findall(text or ""):
        _add(title, url)

    def _walk(node, title_hint=None) -> None:
        if isinstance(node, dict):
            title = node.get("title") if isinstance(node.get("title"), str) \
                else title_hint
            url = node.get("url")
            if isinstance(url, str):
                _add(title, url)
            for value in node.values():
                _walk(value, title)
        elif isinstance(node, list):
            for value in node:
                _walk(value, title_hint)

    _walk(data)
    return sources


def _summarise_with_agentrouter(home: str, away: str, league: str,
                                api_key: str, timeout: float) -> dict:
    """Ask AgentRouter (/v1/responses, web_search tool) for the briefing.

    The model searches the web for the fixture and cites what it finds.
    Returns {"summary", "form", "injuries", "news", "sources"}.
    Raises _BriefError on any failure. The key is never logged.
    The gateway rejects requests without the Claude Code fingerprint
    headers, and its models do not support /v1/chat/completions.
    """
    today = _datetime.now(_timezone.utc).date().isoformat()
    body = {
        "model": AGENTROUTER_MODEL,
        "input": _brief_prompt(home, away, league, today),
        "tools": [{"type": "web_search"}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(AGENTROUTER_FINGERPRINT_HEADERS)
    try:
        resp = requests.post(
            AGENTROUTER_RESPONSES_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise _BriefError(f"AgentRouter request failed: {exc}") from exc
    if resp.status_code != 200:
        detail = ""
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                err = err_json.get("error") or {}
                detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except Exception:
            detail = (resp.text or "")[:200]
        raise _BriefError(
            f"AgentRouter returned HTTP {resp.status_code}"
            + (f": {detail}" if detail else "")
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise _BriefError(
            "AgentRouter returned an unreadable response") from exc
    text = _extract_agentrouter_text(data)
    if not text:
        raise _BriefError("AgentRouter returned no answer text")
    parsed = _parse_briefing(text)
    parsed["sources"] = _extract_sources(text, data)
    return parsed


def _build_brief(home: str, away: str, league: str,
                 agentrouter_key: str) -> dict:
    """Ask AgentRouter (live web search) for the briefing.

    Returns {"summary", "form", "injuries", "news", "sources",
    "live_search"} where sources are the live pages the model cited and
    live_search is always True. Raises _BriefError on any failure.
    Total wall-clock budget is BRIEF_TOTAL_BUDGET_S.
    """
    deadline = time.monotonic() + BRIEF_TOTAL_BUDGET_S

    def _timeout(cap: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _BriefError("AI search time budget exhausted")
        return min(cap, remaining)

    parsed = _summarise_with_agentrouter(
        home, away, league, agentrouter_key,
        _timeout(AGENTROUTER_TIMEOUT_S))
    parsed["live_search"] = True
    return parsed


def clear_brief_cache() -> None:
    """For tests: drop cached briefings."""
    _brief_cache.clear()


class BriefRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Chelsea"])
    league: str = Field(default="PL", examples=["PL"])


class BriefSource(BaseModel):
    title: str
    url: str


class BriefOut(BaseModel):
    home_team: str
    away_team: str
    league: str
    summary: str
    form: list[str]
    injuries: list[str]
    news: list[str]
    sources: list[BriefSource]
    live_search: bool = False
    searched_at: str


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
    try:
        get_predictor("WC")
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


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@app.get("/", response_class=FileResponse)
def web_chat():
    """Free web chat page: type two teams, get a model prediction.

    Served from the same app, so it works at / on Render and locally.
    No ChatGPT plan needed.
    """
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML, media_type="text/html")
    return PlainTextResponse(
        "Football Predictions Agent — web page file is missing. "
        "The API still works: GET /health, POST /predict.",
        status_code=200, media_type="text/plain")


@app.post("/predict", response_model=PredictionOut,
          dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictionOut:
    return _predict_one(req.home_team, req.away_team, req.league)


@app.post("/ai-brief", response_model=BriefOut,
          dependencies=[Depends(require_api_key)])
def ai_brief(req: BriefRequest) -> BriefOut:
    """AI match briefing: live web search via AgentRouter, never
    predictions. The AI searches the web for the fixture and cites
    its sources.

    Team names are used as given, so this works for any league. 503 if
    AGENTROUTER_API_KEY is not set; 502 if AgentRouter fails or times
    out; never 500 and never hangs (95s total budget — the search
    itself is slow).
    """
    home = req.home_team.strip()
    away = req.away_team.strip()
    league = (req.league or "PL").strip().upper()
    if not home or not away:
        raise HTTPException(status_code=400, detail={
            "error": "home_team and away_team must not be empty"})
    if _norm(home) == _norm(away):
        raise HTTPException(status_code=400, detail={
            "error": "Home and away teams must be different"})
    agentrouter_key = os.environ.get("AGENTROUTER_API_KEY")
    if not agentrouter_key:
        raise HTTPException(status_code=503, detail={
            "error": "AI search is not configured",
            "hint": "Set AGENTROUTER_API_KEY on the server to enable the AI match briefing.",
        })
    key = (_norm(home), _norm(away), league)
    cached = _brief_cache.get(key)
    if cached and time.monotonic() - cached[0] < BRIEF_CACHE_TTL_S:
        return BriefOut(**cached[1])
    try:
        parsed = _build_brief(home, away, league, agentrouter_key)
    except _BriefError as exc:
        raise HTTPException(status_code=502, detail={
            "error": "AI search is unavailable right now",
            "hint": str(exc)[:200],
        }) from exc
    except Exception:  # never 500: any unexpected failure is a 502
        raise HTTPException(status_code=502, detail={
            "error": "AI search is unavailable right now",
            "hint": "Unexpected error talking to the AI search service.",
        })
    searched_at = _datetime.now(_timezone.utc).isoformat(timespec="seconds")
    payload = {
        "home_team": home, "away_team": away, "league": league,
        "summary": parsed["summary"], "form": parsed["form"],
        "injuries": parsed["injuries"], "news": parsed["news"],
        "sources": parsed["sources"], "live_search": parsed["live_search"],
        "searched_at": searched_at,
    }
    if len(_brief_cache) >= BRIEF_CACHE_MAX:
        _brief_cache.clear()
    _brief_cache[key] = (time.monotonic(), payload)
    return BriefOut(**payload)


@app.post("/predict-batch", dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchRequest) -> dict:
    req.league = _require_known_league(req.league)
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
    league = _require_known_league(league)
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
            if last is None:
                message = (f"No fixtures on {date or 'that date'}. The bundled sample "
                           f"data has no matches for {league}; set "
                           "FOOTBALL_DATA_API_KEY for real upcoming fixtures.")
            else:
                message = (f"No fixtures on {date or 'that date'}. The bundled sample data "
                           f"has no future fixtures (it ends {last}); set "
                           "FOOTBALL_DATA_API_KEY for real upcoming fixtures.")
        return {"league": league, "date": date, "fixtures": [],
                "message": message, "source": "model"}
    return {"league": league, "date": date, "fixtures": out, "source": "model"}


@app.get("/backtest", dependencies=[Depends(require_api_key)])
def backtest(league: str = "PL") -> dict:
    league = _require_known_league(league)
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
