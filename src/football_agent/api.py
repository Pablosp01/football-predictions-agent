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
import logging
import os
import re
import time
import unicodedata
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
from .data.providers import (ESPN_LEAGUE_SLUGS, INTERNATIONAL_LEAGUES,
                             EspnProvider, get_provider)
from .model.predict import Predictor

MODEL_NAME = "Elo + Dixon-Coles Poisson"
# Server log for AI-gateway diagnostics. Only gateway response shapes
# are logged (status, content type, body length and a short head) —
# never an API key, and never a full body.
logger = logging.getLogger("football_agent.api")
# Every football-data.org free-tier (TIER_ONE) competition, plus the
# UEFA Nations League (NL) — not on football-data.org, it uses the
# international sample pool and ESPN fixtures instead.
LEAGUES = ["PL", "PD", "SA", "BL1", "FL1", "ELC", "DED", "PPL", "BSA",
           "CL", "WC", "EC", "NL"]
DISCLAIMER = (
    "Statistical model output, not betting advice. No guarantee of profit; "
    "probabilities are estimates and football is noisy."
)

# ---------------------------------------------------------------------------
# Team-name normalisation
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Lowercase, fold accents to plain letters, strip punctuation
    and FC suffixes, collapse whitespace.

    Accents are folded, not dropped: "Alavés" becomes "alaves", so
    Vincent typing "Alaves" (no accent) still matches. The old
    version replaced é/ñ with a space, so "Alavés" became "alav s"
    and "Alaves" matched nothing (2026-11-17).
    """
    folded = unicodedata.normalize("NFKD", name.strip().lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9' ]", " ", folded)
    s = re.sub(r"\b(fc|football club|afc)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Articles that carry no meaning in a team name: "RC Deportivo de
# A Coruña" and "RC Deportivo La Coruña" are the same team, so the
# word match ignores them on both sides.
_ARTICLES = {"a", "la", "el", "los", "las", "the", "de", "del",
             "di", "des", "les", "van", "von"}


# Keys are normalised; values are the canonical names the key can
# mean, live name first, sample name second — the first one present
# in the league's team list wins. The live football-data.org names
# ("Manchester City FC") differ from the bundled sample names
# ("Man City"), so one alias must cover both: "Man City" resolves
# to nothing in the live data (the re-test on 2026-11-17 showed
# "Unknown team: 'Man City'"), and "Manchester City FC" resolves
# to nothing in the sample data.
ALIASES = {
    "man utd": ["Manchester United FC", "Man Utd"],
    "man united": ["Manchester United FC", "Man Utd"],
    "manchester united": ["Manchester United FC", "Man Utd"],
    "manchester utd": ["Manchester United FC", "Man Utd"],
    "united": ["Manchester United FC", "Man Utd"],
    "man city": ["Manchester City FC", "Man City"],
    "manchester city": ["Manchester City FC", "Man City"],
    "city": ["Manchester City FC", "Man City"],
    "tottenham": ["Tottenham Hotspur FC", "Tottenham"],
    "tottenham hotspur": ["Tottenham Hotspur FC", "Tottenham"],
    "spurs": ["Tottenham Hotspur FC", "Tottenham"],
    "arsenal": ["Arsenal FC", "Arsenal"],
    "the gunners": ["Arsenal FC", "Arsenal"],
    "aston villa": ["Aston Villa FC", "Aston Villa"],
    "villa": ["Aston Villa FC", "Aston Villa"],
    "bournemouth": ["AFC Bournemouth", "Bournemouth"],
    "brentford": ["Brentford FC", "Brentford"],
    "brighton": ["Brighton & Hove Albion FC", "Brighton"],
    "brighton and hove albion": ["Brighton & Hove Albion FC", "Brighton"],
    "brighton hove albion": ["Brighton & Hove Albion FC", "Brighton"],
    "chelsea": ["Chelsea FC", "Chelsea"],
    "crystal palace": ["Crystal Palace FC", "Crystal Palace"],
    "palace": ["Crystal Palace FC", "Crystal Palace"],
    "everton": ["Everton FC", "Everton"],
    "fulham": ["Fulham FC", "Fulham"],
    "ipswich": ["Ipswich Town FC", "Ipswich"],
    "ipswich town": ["Ipswich Town FC", "Ipswich"],
    "leeds": ["Leeds United FC", "Leeds"],
    "leeds united": ["Leeds United FC", "Leeds"],
    "leicester": ["Leicester City FC", "Leicester"],
    "leicester city": ["Leicester City FC", "Leicester"],
    "liverpool": ["Liverpool FC", "Liverpool"],
    "newcastle": ["Newcastle United FC", "Newcastle"],
    "newcastle united": ["Newcastle United FC", "Newcastle"],
    "nottm forest": ["Nottingham Forest FC", "Nott'm Forest"],
    "nott'm forest": ["Nottingham Forest FC", "Nott'm Forest"],
    "nottingham forest": ["Nottingham Forest FC", "Nott'm Forest"],
    "forest": ["Nottingham Forest FC", "Nott'm Forest"],
    "southampton": ["Southampton FC", "Southampton"],
    "saints": ["Southampton FC", "Southampton"],
    "sunderland": ["Sunderland AFC", "Sunderland"],
    "west ham": ["West Ham United FC", "West Ham"],
    "west ham united": ["West Ham United FC", "West Ham"],
    "wolves": ["Wolverhampton Wanderers FC", "Wolves"],
    "wolverhampton": ["Wolverhampton Wanderers FC", "Wolves"],
    "wolverhampton wanderers": ["Wolverhampton Wanderers FC", "Wolves"],
    # English names for teams whose live names are local:
    # "Bayern Munich" for FC Bayern München, "Inter Milan" for
    # FC Internazionale Milano, "Cologne" for 1. FC Köln,
    # "Turin" for Torino FC.
    "bayern munich": ["FC Bayern München", "Bayern Munich"],
    "inter milan": ["FC Internazionale Milano", "Inter"],
    "cologne": ["1. FC Köln", "Cologne"],
    "turin": ["Torino FC", "Torino"],
    # International teams (names as in the bundled international sample).
    "usa": ["United States"], "us": ["United States"],
    "america": ["United States"],
    "united states of america": ["United States"],
    "cote d'ivoire": ["Ivory Coast"], "c te d'ivoire": ["Ivory Coast"],
    "czechia": ["Czech Republic"], "ir iran": ["Iran"],
    "korea republic": ["South Korea"], "turkiye": ["Turkey"],
    "holland": ["Netherlands"],
}


def _require_known_league(league: str) -> str:
    """Normalise a league code, or raise a clear 400 for unknown ones."""
    code = (league or "PL").strip().upper()
    if code not in LEAGUES:
        raise HTTPException(status_code=400, detail={
            "error": f"Unknown league: '{league}'",
            "known_leagues": LEAGUES,
            "hint": ("Supported competitions are the football-data.org free tier: "
                     "PL, PD, SA, BL1, FL1, ELC, DED, PPL, BSA, CL, WC, EC, "
                     "plus the UEFA Nations League (NL)."),
        })
    return code


def _known_teams(league: str) -> list[str]:
    """Canonical team names for a league, cached per process.

    The underlying data does not change while the server runs (a new
    deploy or a changed data key restarts it); tests reset it with
    clear_cache().
    """
    if league not in _known_teams_cache:
        provider = get_provider(league)
        matches = provider.matches(league)
        _known_teams_cache[league] = sorted(
            {m.home_team for m in matches} | {m.away_team for m in matches})
    return _known_teams_cache[league]


def resolve_team(name: str, league: str) -> str:
    """Resolve a free-text team name to a canonical team, or raise 400."""
    known = _known_teams(league)
    by_norm = {_norm(t): t for t in known}
    key = _norm(name)
    if key in by_norm:
        return by_norm[key]
    # Aliases cover both name eras (live "Manchester City FC" and
    # sample "Man City"): the first candidate present in this
    # league's team list wins.
    if key in ALIASES:
        for candidate in ALIASES[key]:
            if candidate in known:
                return candidate
    # Substring match either way ("Newcastle" in "Newcastle United", etc.)
    for normed, canonical in by_norm.items():
        if key and (key in normed or normed in key):
            return canonical
    # Word match: every non-article word of the name appears in
    # the canonical name ("Celta Vigo" -> "RC Celta de Vigo",
    # whose words are not contiguous). Articles are ignored on
    # both sides, so "RC deportivo de A coruna" still finds "RC
    # Deportivo La Coruña" — "a" and "la" are the same word for
    # this purpose. Only for multi-word names — a single word is
    # too ambiguous ("Real" matches half the league).
    key_words = {w for w in key.split() if w not in _ARTICLES}
    if len(key_words) > 1:
        for normed, canonical in by_norm.items():
            rest = {w for w in normed.split() if w not in _ARTICLES}
            if key_words <= rest:
                return canonical
    # Aliases are exact only: a substring alias ("city" for Man
    # City) would otherwise claim any team whose name contains it
    # — "Coventry City" resolved to Man City, and the model then
    # predicted a match that does not exist (ESPN lists Coventry
    # vs Crystal Palace in the eng.1 feed for 2026-11-21).
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


# Order to search when a team is not in the league asked for: the big
# domestic leagues first, then the Champions League, then the
# international pool (Nations League first — the chatbot's convention
# for national teams). A club team also plays in its domestic league,
# so domestic leagues come before the Champions League.
_LEAGUE_SEARCH_ORDER = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL",
                        "BSA", "ELC", "CL", "NL", "WC", "EC"]


def _resolve_team_quiet(name: str, league: str) -> str | None:
    """resolve_team without raising: the canonical name, or None.

    Any failure counts as "not in this league" — including the data
    feed failing. A rate limit returns 429, which is not an
    HTTPException, so catching only that would let one league's
    failed fetch 500 the whole request: that is what broke
    predictions at 09:29 on 2026-11-17, when a cold restart made
    the fallback search fetch nine leagues at once and the free
    feed capped at 10 requests a minute.
    """
    try:
        return resolve_team(name, league)
    except Exception:
        return None


def resolve_team_anywhere(name: str, league: str) -> tuple[str, str]:
    """Resolve a team name, searching every league.

    Returns (canonical_name, league_found). The league asked for is
    tried first; then the other leagues in _LEAGUE_SEARCH_ORDER.
    Raises the asked-for league's 400 (with did_you_mean) when the
    team is found nowhere — so a genuinely unknown name still gets
    useful suggestions for the league asked about.

    This is the chatbot trap fix: the chatbot inherits the Predict
    tab's league, so "Arsenal's form" asked with the tab on the
    Championship came back "Unknown team" — the model has Arsenal in
    the Premier League, and the answer should come from there.
    """
    first_error = None
    try:
        return resolve_team(name, league), league
    except HTTPException as err:
        first_error = err
    except Exception:
        # The requested league's data could not be fetched at all
        # (the feed rate-limited us, or the network failed): there
        # is no "unknown team" error to keep — try the other leagues,
        # which may be cached, and say the feed is unavailable if
        # none of them resolves the team either.
        first_error = None
    for other in _LEAGUE_SEARCH_ORDER:
        if other == league:
            continue
        found = _resolve_team_quiet(name, other)
        if found is not None:
            return found, other
    if first_error is not None:
        raise first_error
    raise HTTPException(status_code=503, detail={
        "error": "The live data feed is temporarily unavailable — try again in a minute.",
    })


# ---------------------------------------------------------------------------
# Cached fitted model (fit once per league, reuse for every request)
# ---------------------------------------------------------------------------

_predictors: dict[str, Predictor] = {}
_matches_cache: dict[str, list] = {}
_known_teams_cache: dict[str, list[str]] = {}


def get_predictor(league: str) -> Predictor:
    league = _require_known_league(league)
    if league not in _predictors:
        provider = get_provider(league)
        try:
            matches = provider.matches(league)
        except Exception as err:
            # A failed fetch (the free feed rate-limits at 10
            # requests a minute) is a temporary 503, never a 500:
            # the request can be retried in a minute.
            logger.warning("data feed failed for league %s: %r", league, err)
            raise HTTPException(status_code=503, detail={
                "error": f"The live data feed is temporarily unavailable for league '{league}'",
                "hint": "The free data feed allows 10 requests a minute — try again in a minute.",
            })
        if len(matches) < 10:
            if os.environ.get("FOOTBALL_DATA_API_KEY"):
                hint = (f"The live feed returned only {len(matches)} finished "
                        f"matches for league '{league}'; the model needs at "
                        "least 10 to fit.")
            else:
                hint = ("Set FOOTBALL_DATA_API_KEY for live data; the bundled sample "
                        "data only covers the Premier League (PL) and the "
                        "international tournaments (WC, EC, NL).")
            raise HTTPException(status_code=503, detail={
                "error": f"Not enough historical matches to fit the model for league '{league}'",
                "matches_available": len(matches),
                "hint": hint,
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
    _known_teams_cache.clear()


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
    # The more likely BTTS side, e.g. {"side": "yes", "probability": 0.525}.
    btts_pick: dict
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
        btts_pick=d["btts_pick"],
    )


def _predict_one(home: str, away: str, league: str) -> PredictionOut:
    league = _require_known_league(league)
    try:
        predictor = get_predictor(league)
        home_c = resolve_team(home, league)
        away_c = resolve_team(away, league)
    except HTTPException as first_error:
        # The league asked for cannot predict this pair: it may have
        # too little history (the Championship), or the teams may play
        # in another league (the chatbot inherits the Predict tab's
        # league). Predict in the league the teams play in instead;
        # if none works, the original error stands.
        fallback = _predict_one_anywhere(home, away, league)
        if fallback is not None:
            return fallback
        raise first_error
    if home_c == away_c:
        raise HTTPException(status_code=400, detail={
            "error": "Home and away teams must be different",
            "home_team": home_c, "away_team": away_c,
        })
    return _prediction_out(predictor.predict(home_c, away_c), league)


def _predict_one_anywhere(home: str, away: str,
                          requested_league: str) -> PredictionOut | None:
    """Predict a pair in the first league where both teams play and
    the model fits. Teams are resolved before fitting, so a league is
    only fitted when both teams are in it. Returns None when no
    league works."""
    for league in _LEAGUE_SEARCH_ORDER:
        if league == requested_league:
            continue
        home_c = _resolve_team_quiet(home, league)
        if home_c is None:
            continue
        away_c = _resolve_team_quiet(away, league)
        if away_c is None or away_c == home_c:
            continue
        try:
            predictor = get_predictor(league)
        except Exception:
            # The league's data could not be fetched (rate limit):
            # skip it, do not fail the whole request.
            continue
        return _prediction_out(predictor.predict(home_c, away_c), league)
    return None


# ---------------------------------------------------------------------------
# AI match briefing (AgentRouter primary, Xingchi fallback)
#
# Context only: recent form, injuries/suspensions, team news found by a
# live web search. The model's numbers from /predict are authoritative
# and are never touched by this section.
#
# Primary: AgentRouter's built-in web search. The /v1/responses call
# carries "tools": [{"type": "web_search"}], so the model searches
# the web for the fixture and cites what it finds. The prompt tells it
# to use ONLY the search results, to cite every factual claim with a
# markdown source link, and never to invent players, results, or
# injuries. Sources are extracted from the answer's markdown links and
# from any urls in the response's web_search_call output items, and the
# response carries live_search=true. The web page says the briefing is a
# live web search and lists the cited sources below it.
#
# Fallback: if the AgentRouter call fails (HTTP error, timeout, empty
# or unparseable answer) and a Xingchi key is set, the briefing falls
# back to Xingchi's /v1/chat/completions (model gpt-6.1-sol, the only
# one that answers — gpt-6-astra returns 503). Xingchi has no web
# search (its /v1/responses returns 400 "Upstream request failed"),
# so the fallback is knowledge-only: the prompt states the model has
# no web access and must only say what it is confident about, and the
# response carries live_search=false with no sources. The web page
# says so ("From the AI's own knowledge, not a live web search").
#
# Needs keys on the server (either one enables the briefing, neither
# is required for predictions):
#   AGENTROUTER_API_KEY — free AgentRouter key, the primary summariser.
#   XINGCHI_API_KEY — pay-as-you-go Xingchi key, the fallback only.
# Without either, /ai-brief answers 503 and the page hides the briefing.
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
# Timing: the search itself is slow — plain calls answer in ~3s and a
# trivial web-searched call in ~5s, but the briefing's web search is
# heavier: the simple single-search prompt typically answers in
# ~40-70s against the real key (the earlier open-ended prompt took
# 130s+ and timed out). The AgentRouter timeout is therefore 90s and
# the whole-briefing budget 95s, with one bounded retry for a fast
# failure only; the page shows a loading state for that long, and the
# 15-minute cache makes repeat requests instant.
# ---------------------------------------------------------------------------

AGENTROUTER_MODEL = os.environ.get("AGENTROUTER_MODEL", "gpt-5.6-sol")
AGENTROUTER_RESPONSES_URL = "https://agentrouter.org/v1/responses"
# The gateway rejects clients without a supported harness fingerprint.
AGENTROUTER_FINGERPRINT_HEADERS = {
    "User-Agent": "claude-cli/2.1.220 (external, claude-desktop)",
    "anthropic-version": "2023-06-01",
}
AGENTROUTER_TIMEOUT_S = 90
# A fast AgentRouter failure (e.g. a transient unreadable gateway
# response) is retried once; a failure that took this long or more
# has already spent the budget and is not retried.
AGENTROUTER_FAST_RETRY_S = 30
BRIEF_TOTAL_BUDGET_S = 95  # wall-clock budget for the whole briefing
BRIEF_CACHE_TTL_S = 15 * 60  # save free-tier quota: one briefing per pair/15 min
BRIEF_CACHE_MAX = 200

# Xingchi fallback (knowledge-only): the gateway has no web search,
# so it is used only when AgentRouter fails. Verified 2026-11-15
# against the live key: POST https://api.xingchiapi.com/v1/chat/completions
# with the same Claude Code fingerprint headers and a Bearer key,
# model gpt-6.1-sol (gpt-6-astra returns 503 on every call), body
# {"model": ..., "messages": [{"role": "user", "content": ...}]}.
# Plain requests without the fingerprint get HTTP 403
# "client_not_allowed"; /v1/responses returns 400 "Upstream request
# failed" — never use it here.
XINGCHI_MODEL = os.environ.get("XINGCHI_MODEL", "gpt-6.1-sol")
XINGCHI_CHAT_URL = "https://api.xingchiapi.com/v1/chat/completions"
XINGCHI_FINGERPRINT_HEADERS = {
    "User-Agent": "claude-cli/2.1.220 (external, claude-desktop)",
    "anthropic-version": "2023-06-01",
}
XINGCHI_TIMEOUT_S = 30

_brief_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}


class _BriefError(Exception):
    """The AgentRouter call failed (HTTP error, timeout, bad answer).

    ``retryable`` says whether a second attempt could help. A web
    firewall page or an empty body will not change on an immediate
    retry, so those are not retryable; a transport error or a
    transient bad reply is. ``waf`` marks a web firewall challenge
    page: the gateway may still answer a plain call without the
    search tool, so that is worth one try.
    """

    def __init__(self, message: str, retryable: bool = True,
                 waf: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.waf = waf


_LEAGUE_NAMES = {
    "PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
    "BL1": "Bundesliga", "FL1": "Ligue 1", "ELC": "Championship",
    "DED": "Eredivisie", "PPL": "Primeira Liga",
    "BSA": "Campeonato Brasileiro Série A", "CL": "Champions League",
    "WC": "FIFA World Cup", "EC": "European Championship",
    "NL": "UEFA Nations League",
}


def _brief_prompt_knowledge(home: str, away: str, league: str,
                            today: str) -> str:
    """Same briefing shape as _brief_prompt, but the model has no web
    access (Xingchi fallback): it must answer only from what it knows
    with confidence, and say nothing it is unsure about.
    """
    league_name = _LEAGUE_NAMES.get(league.upper(), league)
    return (
        f"Today is {today}. The football match is {home} vs {away} "
        f"({league_name}).\n\n"
        "You have no web access. Answer only from what you know with "
        "confidence, then return a JSON object (no other text) with exactly these keys:\n"
        '{"summary": "2-3 sentence overview of the match context", '
        '"form": ["<team>: last-5 results, e.g. W W D L W, with scores if known"], '
        '"injuries": ["<player> (<team>) — injury or suspension detail, or empty if none known"], '
        '"news": ["any other relevant team news"]}\n\n'
        "Rules:\n"
        "- Use ONLY what you are confident you know. If a section "
        "has no reliable information, use an empty list.\n"
        "- If there is no upcoming fixture between these teams on or "
        "near this date, do NOT leave everything empty: give the "
        "latest news and recent form for each team from what you "
        "know, and say so in the summary (e.g. 'No fixture between "
        "these teams is scheduled — latest on each team:').\n"
        "- You DO know the teams themselves: their league, recent "
        "league position, key players and head-to-head history. Use "
        "that general context when fixture-specific news is "
        "unavailable, and mark anything dated as approximate.\n"
        "- Do NOT invent players, results, or injuries.\n"
        "- Do NOT predict a winner, scoreline, or probabilities. Do NOT mention odds.\n"
        "- Keep every item to one short sentence. No betting language."
    )


def _brief_prompt(home: str, away: str, league: str, today: str) -> str:
    """Single-search briefing prompt.

    The earlier open-ended prompt ("search the web for this fixture")
    let the model run search after search and took 130s+ against the
    real key, past the endpoint's budget. This version asks for ONE
    search with a fixed query and answers from that result only; it
    was verified live at ~40-70s. Same JSON keys and rules as before.
    """
    league_name = _LEAGUE_NAMES.get(league.upper(), league)
    return (
        f"Today is {today}. The football match is {home} vs {away} "
        f"({league_name}). Use the web search tool ONCE to search for "
        f"'{home} vs {away} latest news injuries form' — one search only, "
        "do not run more searches — then, using ONLY that search result, "
        "return a JSON object (no other text) with exactly these keys:\n"
        '{"summary": "2-3 sentence overview of the match context", '
        '"form": ["<team>: last-5 results, e.g. W W D L W, with scores if found"], '
        '"injuries": ["<player> (<team>) — injury or suspension detail, or empty if none found"], '
        '"news": ["any other relevant team news"]}\n\n'
        "Rules:\n"
        "- Use ONLY what you find in the web search result. If a section "
        "has no reliable information, use an empty list.\n"
        "- If there is no upcoming fixture between these teams, use the "
        "latest news for each team instead of leaving everything empty.\n"
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


def _http_error_detail(resp) -> str:
    """Best-effort error text from a failed gateway response."""
    try:
        err_json = resp.json()
        if isinstance(err_json, dict):
            err = err_json.get("error") or {}
            if isinstance(err, dict):
                return str(err.get("message", ""))
            return str(err)
    except Exception:
        pass
    return (resp.text or "")[:200]


def _summarise_with_agentrouter(home: str, away: str, league: str,
                                api_key: str, timeout: float,
                                use_search: bool = True) -> dict:
    """Ask AgentRouter (/v1/responses) for the briefing.

    With ``use_search`` (default) the call carries the web_search
    tool: the model searches the web for the fixture and cites what
    it finds. Without it the call is plain and the knowledge prompt
    is used instead — the answer is from what the model knows, with
    no live search. That plain form is the fallback when the
    gateway's web firewall blocks the search call from this server
    (it answers HTTP 200 with an HTML challenge page): the firewall
    may still let a plain call through, and a knowledge-only
    briefing beats no briefing.

    Returns {"summary", "form", "injuries", "news", "sources"}.
    Raises _BriefError on any failure. The key is never logged.
    The gateway rejects requests without the Claude Code fingerprint
    headers, and its models do not support /v1/chat/completions.
    """
    today = _datetime.now(_timezone.utc).date().isoformat()
    if use_search:
        prompt = _brief_prompt(home, away, league, today)
    else:
        prompt = _brief_prompt_knowledge(home, away, league, today)
    body = {
        "model": AGENTROUTER_MODEL,
        "input": prompt,
    }
    if use_search:
        body["tools"] = [{"type": "web_search"}]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # The gateway sits behind a web firewall (Alibaba Cloud WAF):
        # from Render's servers it answers HTTP 200 with an HTML
        # challenge page instead of JSON (verified 2026-11-16 in the
        # Render logs). A plain browser-style Accept header is sent so
        # the request does not look like a bare script call; the
        # Claude Code fingerprint headers are still required by the
        # gateway itself.
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
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
        detail = _http_error_detail(resp)
        logger.warning(
            "AgentRouter HTTP %s: content-type=%s body-head=%r",
            resp.status_code,
            (getattr(resp, "headers", None) or {}).get("Content-Type", ""),
            (resp.text or "")[:200])
        raise _BriefError(
            f"AgentRouter returned HTTP {resp.status_code}"
            + (f": {detail}" if detail else "")
        )
    try:
        data = resp.json()
    except ValueError as exc:
        body_text = resp.text or ""
        content_type = (getattr(resp, "headers", None) or {}).get(
            "Content-Type", "")
        logger.warning(
            "AgentRouter unreadable response: status=%s content-type=%s "
            "len=%s body-head=%r",
            resp.status_code, content_type,
            len(body_text), body_text[:300])
        if content_type.startswith("text/html"):
            # Web firewall challenge page, not a gateway answer: an
            # immediate retry gets the same page, so do not retry.
            raise _BriefError(
                "AgentRouter is blocking requests from this server "
                "(web firewall)", retryable=False, waf=True) from exc
        if not body_text.strip():
            raise _BriefError(
                "AgentRouter returned an empty response",
                retryable=False) from exc
        raise _BriefError(
            "AgentRouter returned an unreadable response") from exc
    text = _extract_agentrouter_text(data)
    if not text:
        logger.warning(
            "AgentRouter no answer text: top-level keys=%s output items=%s",
            sorted(data.keys()) if isinstance(data, dict) else type(data),
            [it.get("type") for it in (data.get("output") or [])
             if isinstance(it, dict)] if isinstance(data, dict) else "?")
        raise _BriefError("AgentRouter returned no answer text")
    parsed = _parse_briefing(text)
    parsed["sources"] = _extract_sources(text, data)
    return parsed


def _extract_xingchi_text(data: dict) -> str:
    """Return the answer text from a Xingchi /v1/chat/completions body.

    Shape: choices[0].message.content.
    """
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"].strip()
    return ""


def _summarise_with_xingchi(home: str, away: str, league: str,
                            api_key: str, timeout: float) -> dict:
    """Ask Xingchi (/v1/chat/completions, gpt-6.1-sol) for the briefing.

    Knowledge-only fallback: Xingchi has no web search, so the model
    answers from what it knows and the result carries live_search
    False and no sources. Returns {"summary", "form", "injuries",
    "news", "sources": []}. Raises _BriefError on any failure.
    The key is never logged. The gateway rejects requests without
    the Claude Code fingerprint headers (HTTP 403), and gpt-6-astra
    is unavailable (HTTP 503) — only gpt-6.1-sol is used.
    """
    today = _datetime.now(_timezone.utc).date().isoformat()
    body = {
        "model": XINGCHI_MODEL,
        "messages": [{"role": "user",
                      "content": _brief_prompt_knowledge(
                          home, away, league, today)}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(XINGCHI_FINGERPRINT_HEADERS)
    try:
        resp = requests.post(
            XINGCHI_CHAT_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise _BriefError(f"Xingchi request failed: {exc}") from exc
    if resp.status_code != 200:
        detail = _http_error_detail(resp)
        logger.warning(
            "Xingchi HTTP %s: content-type=%s body-head=%r",
            resp.status_code,
            (getattr(resp, "headers", None) or {}).get("Content-Type", ""),
            (resp.text or "")[:200])
        raise _BriefError(
            f"Xingchi returned HTTP {resp.status_code}"
            + (f": {detail}" if detail else "")
        )
    try:
        data = resp.json()
    except ValueError as exc:
        body_text = resp.text or ""
        logger.warning(
            "Xingchi unreadable response: status=%s content-type=%s "
            "len=%s body-head=%r",
            resp.status_code,
            (getattr(resp, "headers", None) or {}).get("Content-Type", ""),
            len(body_text), body_text[:300])
        if not body_text.strip():
            raise _BriefError("Xingchi returned an empty response") from exc
        raise _BriefError(
            "Xingchi returned an unreadable response") from exc
    text = _extract_xingchi_text(data)
    if not text:
        raise _BriefError("Xingchi returned no answer text")
    parsed = _parse_briefing(text)
    parsed["sources"] = []
    return parsed


def _build_brief(home: str, away: str, league: str,
                 agentrouter_key: str | None,
                 xingchi_key: str | None = None) -> dict:
    """Ask AgentRouter first (live web search); fall back to Xingchi.

    Returns {"summary", "form", "injuries", "news", "sources",
    "live_search"} — live_search True with the cited sources when
    AgentRouter answers, False with no sources when the Xingchi
    knowledge-only fallback answers. Raises _BriefError on any
    failure. Total wall-clock budget is BRIEF_TOTAL_BUDGET_S; the
    fallback only runs if time is left in that budget.

    A fast, retryable AgentRouter failure (elapsed <
    AGENTROUTER_FAST_RETRY_S, e.g. a transient unreadable gateway
    response) is retried once before falling back; a slow failure
    has already spent the budget and is not retried, and a
    non-retryable one (web firewall page, empty body) is never
    retried — the immediate second attempt gets the same page.
    The exception is a firewall block: the search call is retried
    once WITHOUT the search tool (knowledge prompt), because the
    firewall may still let a plain call through, and a
    knowledge-only briefing beats none.
    """
    started = time.monotonic()
    deadline = started + BRIEF_TOTAL_BUDGET_S

    def _timeout(cap: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _BriefError("AI search time budget exhausted")
        return min(cap, remaining)

    if agentrouter_key:
        try:
            parsed = _summarise_with_agentrouter(
                home, away, league, agentrouter_key,
                _timeout(AGENTROUTER_TIMEOUT_S))
            parsed["live_search"] = True
            return parsed
        except _BriefError as first_error:
            if first_error.waf:
                # The firewall blocked the search call; try once
                # without the search tool (knowledge prompt). The
                # firewall may let a plain call through, and the
                # result is honestly labelled knowledge-only.
                try:
                    parsed = _summarise_with_agentrouter(
                        home, away, league, agentrouter_key,
                        _timeout(AGENTROUTER_TIMEOUT_S), use_search=False)
                    parsed["live_search"] = False
                    parsed["sources"] = []
                    return parsed
                except _BriefError:
                    pass
            elif (first_error.retryable and
                    time.monotonic() - started < AGENTROUTER_FAST_RETRY_S):
                try:
                    parsed = _summarise_with_agentrouter(
                        home, away, league, agentrouter_key,
                        _timeout(AGENTROUTER_TIMEOUT_S))
                    parsed["live_search"] = True
                    return parsed
                except _BriefError:
                    pass
            if not xingchi_key:
                raise
    if not xingchi_key:
        raise _BriefError("No AI briefing key is configured")
    parsed = _summarise_with_xingchi(
        home, away, league, xingchi_key,
        _timeout(XINGCHI_TIMEOUT_S))
    parsed["live_search"] = False
    parsed["sources"] = []
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


@app.middleware("http")
async def _no_store(request, call_next):
    """Never let a browser cache any response: a deploy changes the
    page and the answers, and a saved copy shows the old behaviour
    after an update. The page itself was already no-store; the
    2026-11-16 /today check showed a saved JSON answer doing the
    same thing, so this covers every response."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


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
        # Never cache the page: it is the whole app (HTML + JS inline),
        # and a deploy changes it. Without this, a browser keeps
        # serving the previous deploy's page — the old dropdown, the
        # old chatbot — after a new version is live. That was the
        # 2026-11-16 false failure: the live site was already new,
        # but the checking browser showed its saved old copy.
        return FileResponse(INDEX_HTML, media_type="text/html",
                            headers={"Cache-Control": "no-store"})
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
    its sources. If AgentRouter's web firewall blocks the search
    call from this server, it is retried once without the search
    tool (knowledge-only answer, labelled as such); if that is
    blocked too and a Xingchi key is set, the briefing falls back
    to Xingchi's knowledge-only answer (live_search false, no
    sources).

    Team names are used as given, so this works for any league. 503 if
    neither key is set; 502 if both services fail or time out; never
    500 and never hangs (95s total budget — the search itself is
    slow).
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
    xingchi_key = os.environ.get("XINGCHI_API_KEY")
    if not agentrouter_key and not xingchi_key:
        raise HTTPException(status_code=503, detail={
            "error": "AI search is not configured",
            "hint": "Set AGENTROUTER_API_KEY or XINGCHI_API_KEY on the server "
                    "to enable the AI match briefing.",
        })
    key = (_norm(home), _norm(away), league)
    cached = _brief_cache.get(key)
    if cached and time.monotonic() - cached[0] < BRIEF_CACHE_TTL_S:
        return BriefOut(**cached[1])
    try:
        parsed = _build_brief(home, away, league,
                              agentrouter_key, xingchi_key)
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
    # The pairs may have been predicted in another league (a pair
    # whose teams play elsewhere falls back there). The top-level
    # league is the predictions' league when they all agree.
    leagues = {r.league for r in results}
    league = leagues.pop() if len(leagues) == 1 else req.league
    return {"league": league, "count": len(ranked),
            "predictions": ranked, "source": "model",
            "disclaimer": DISCLAIMER}


def _team_last5(predictor, team: str) -> tuple[list[str], float]:
    """A team's last-5 form letters (oldest first) and points per game,
    from the fitted history. Shared by /ratings (and the same logic
    lives in /team) so a league-winner answer is data — rating plus
    recent form and points — not just one favourite name."""
    played = [m for m in predictor.history
              if m.home_team == team or m.away_team == team]
    recent = sorted(played, key=lambda m: m.date)[-5:]
    form: list[str] = []
    points = 0
    for m in recent:
        if m.home_team == team:
            gf, ga = m.home_goals or 0, m.away_goals or 0
        else:
            gf, ga = m.away_goals or 0, m.home_goals or 0
        if gf > ga:
            form.append("W")
            points += 3
        elif gf == ga:
            form.append("D")
            points += 1
        else:
            form.append("L")
    n = len(recent)
    return form, round(points / n, 2) if n else 0.0


@app.get("/ratings", dependencies=[Depends(require_api_key)])
def ratings(league: str = "PL") -> dict:
    """Elo table for a league, strongest first, with each team's
    recent form and points per game.

    The model's answer to "who is the best team / who will win the
    league": the highest Elo rating, backed by the last-5 form and
    points — not a favourite name on its own. Ratings come from the
    fitted history, so data_as_of is the last match date in that
    history.
    """
    league = _require_known_league(league)
    predictor = get_predictor(league)

    def _row(team: str) -> dict:
        form, ppg = _team_last5(predictor, team)
        return {"team": team,
                "rating": round(predictor.elo.rating(team), 1),
                "form": form, "points_per_game": ppg}

    rows = sorted((_row(t) for t in _known_teams(league)),
                  key=lambda r: -r["rating"])
    last = predictor.history[-1].date if predictor.history else None
    return {"league": league, "ratings": rows,
            "data_as_of": last.isoformat() if last else None,
            "source": "model"}


@app.get("/team", dependencies=[Depends(require_api_key)])
def team_card(team: str, league: str = "PL") -> dict:
    """A team's form card: Elo rating, last-5 results, goals.

    Form is the last five played matches in the fitted history,
    oldest first; recent is the same five, newest first.
    data_as_of is the last match date in that history. Unknown teams
    get the same 400 + did_you_mean as /predict.
    """
    league = _require_known_league(league)
    try:
        canonical = resolve_team(team, league)
    except HTTPException:
        # The team plays in another league (the chatbot inherits the
        # Predict tab's league): answer from the league it plays in.
        canonical, league = resolve_team_anywhere(team, league)
    predictor = get_predictor(league)
    played = [m for m in predictor.history
              if m.home_team == canonical or m.away_team == canonical]
    recent = sorted(played, key=lambda m: m.date)[-5:]
    form, scored, conceded, results = [], 0, 0, []
    for m in recent:
        if m.home_team == canonical:
            gf, ga, opponent = m.home_goals, m.away_goals, m.away_team
        else:
            gf, ga, opponent = m.away_goals, m.home_goals, m.home_team
        gf, ga = gf or 0, ga or 0
        scored += gf
        conceded += ga
        letter = "W" if gf > ga else "D" if gf == ga else "L"
        form.append(letter)
        results.append({"date": m.date.isoformat(), "opponent": opponent,
                        "score": f"{gf}-{ga}",  # the team's goals first
                        "result": letter,
                        "home": m.home_team == canonical})
    n = len(recent)
    last = predictor.history[-1].date if predictor.history else None
    return {"team": canonical, "league": league,
            "rating": round(predictor.elo.rating(canonical), 1),
            "form": form,
            "points_per_game": round(sum(3 if f == "W" else 1 if f == "D" else 0
                                         for f in form) / n, 2) if n else 0.0,
            "goals_scored_per_game": round(scored / n, 2) if n else 0.0,
            "goals_conceded_per_game": round(conceded / n, 2) if n else 0.0,
            "recent": list(reversed(results)),
            "data_as_of": last.isoformat() if last else None,
            "source": "model"}


def _head_to_head_result(home: str, away: str, predictor: Predictor,
                         league: str) -> dict:
    """The head-to-head response body for resolved teams and a fitted
    predictor: the last five meetings, newest first, with the tally."""
    pair = {home, away}
    meetings = sorted(
        (m for m in predictor.history
         if {m.home_team, m.away_team} == pair),
        key=lambda m: m.date, reverse=True)[:5]
    hw = draws = aw = 0
    out = []
    for m in meetings:
        hg, ag = m.home_goals or 0, m.away_goals or 0
        if hg > ag:
            winner = m.home_team
        elif hg < ag:
            winner = m.away_team
        else:
            winner = None
        if winner == home:
            hw += 1
        elif winner == away:
            aw += 1
        else:
            draws += 1
        out.append({"date": m.date.isoformat(), "home_team": m.home_team,
                    "away_team": m.away_team, "score": f"{hg}-{ag}"})
    last = predictor.history[-1].date if predictor.history else None
    return {"home_team": home, "away_team": away, "league": league,
            "meetings": out, "home_wins": hw, "draws": draws,
            "away_wins": aw,
            "data_as_of": last.isoformat() if last else None,
            "source": "model"}


def _head_to_head_in_league(home_team: str, away_team: str,
                            league: str) -> dict | None:
    """Head-to-head in one league: None when the teams do not both
    play there or the model cannot fit it."""
    home = _resolve_team_quiet(home_team, league)
    if home is None:
        return None
    away = _resolve_team_quiet(away_team, league)
    if away is None or away == home:
        return None
    try:
        predictor = get_predictor(league)
    except Exception:
        # The league's data could not be fetched (rate limit):
        # skip it, do not fail the whole request.
        return None
    return _head_to_head_result(home, away, predictor, league)


@app.get("/head-to-head", dependencies=[Depends(require_api_key)])
def head_to_head_endpoint(home_team: str, away_team: str,
                          league: str = "PL") -> dict:
    """Last five meetings between two teams, newest first."""
    league = _require_known_league(league)
    try:
        home = resolve_team(home_team, league)
        away = resolve_team(away_team, league)
        if home == away:
            raise HTTPException(status_code=400, detail={
                "error": "Home and away teams must be different."})
        predictor = get_predictor(league)
    except HTTPException as first_error:
        # The teams may play in another league (the chatbot inherits
        # the Predict tab's league): answer from the league they play
        # in. If none, the original error stands.
        for other in _LEAGUE_SEARCH_ORDER:
            if other == league:
                continue
            out = _head_to_head_in_league(home_team, away_team, other)
            if out is not None:
                return out
        raise first_error
    return _head_to_head_result(home, away, predictor, league)


def _predict_fixtures(days_fixtures, league: str) -> list[dict]:
    """Predict a fixture list; fixtures the model cannot resolve or
    predict keep prediction None.

    Team names are resolved to the model's canonical names first
    (ESPN says "Manchester City", the model knows "Man City"), and
    the resolved names are what the output carries. A league the
    model cannot fit without a data key (every club league except
    PL) yields prediction None for every fixture.
    """
    out: list[dict] = []
    try:
        predictor = get_predictor(league)
    except HTTPException:
        predictor = None
    for fx in days_fixtures:
        item: dict = {"date": fx.date.isoformat(), "home_team": fx.home_team,
                      "away_team": fx.away_team, "prediction": None}
        if predictor is not None:
            try:
                home = resolve_team(fx.home_team, league)
                away = resolve_team(fx.away_team, league)
                if home == away:
                    # Same team on both sides: bad feed data, never a
                    # real match. Leave it unpredicted rather than
                    # predicting a team against itself (ESPN listed
                    # "Alavés vs Deportivo" for 2026-11-21, both
                    # resolving to Deportivo Alavés).
                    pass
                else:
                    pred = _prediction_out(
                        predictor.predict(home, away, on=fx.date), league)
                    item["home_team"] = home
                    item["away_team"] = away
                    item["prediction"] = pred.model_dump()
            except HTTPException:
                pass
        out.append(item)
    return out


@app.get("/fixtures", dependencies=[Depends(require_api_key)])
def fixtures(league: str = "PL",
             date: str | None = Query(default=None, examples=["2026-11-15"]),
             team: str | None = Query(default=None, examples=["Arsenal"])) -> dict:
    league = _require_known_league(league)
    on = None
    if date:
        try:
            on = _date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail={
                "error": f"Invalid date '{date}': use YYYY-MM-DD"})
    canonical = None
    if team:
        # A team's fixtures only (e.g. its next match): find the team
        # wherever it plays (the chatbot inherits the Predict tab's
        # league) and fetch that league's fixtures. Unknown names get
        # the same 400 + did_you_mean as /predict.
        canonical, league = resolve_team_anywhere(team, league)
    provider = get_provider(league)
    espn_ok = False
    if os.environ.get("FOOTBALL_DATA_API_KEY"):
        try:
            days_fixtures = provider.fixtures(league, on)
        except Exception as err:
            logger.warning("live fixtures failed for league %s: %r", league, err)
            raise HTTPException(status_code=503, detail={
                "error": f"The live data feed is temporarily unavailable for league '{league}'",
                "hint": "The free data feed allows 10 requests a minute — try again in a minute.",
            })
    else:
        # Live fixtures from ESPN's free scoreboard (no key). On
        # failure fall back to the bundled sample, which has no
        # future fixtures — and says so below.
        try:
            days_fixtures = EspnProvider().fixtures(league, on)
            espn_ok = True
        except Exception:
            days_fixtures = provider.fixtures(league, on)
    if team:
        # Fixture names are resolved first — ESPN says "Manchester
        # City", the model knows "Man City".
        kept = []
        for fx in days_fixtures:
            try:
                h = resolve_team(fx.home_team, league)
                a = resolve_team(fx.away_team, league)
            except HTTPException:
                continue
            if h == canonical or a == canonical:
                kept.append(fx)
        days_fixtures = kept
    out = _predict_fixtures(days_fixtures, league)
    if not out:
        if espn_ok:
            # ESPN answered: there really are no matches that day.
            message = f"No {league} matches on {date or 'that date'}."
        elif os.environ.get("FOOTBALL_DATA_API_KEY"):
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


@app.get("/today", dependencies=[Depends(require_api_key)])
def today(date: str | None = Query(default=None, examples=["2026-11-16"])) -> dict:
    """What matches are actually on today (or a given day), with
    the model's prediction for each.

    Every league the site knows is searched, on ESPN's free
    scoreboard — no key needed: all ten club leagues (Premier League,
    La Liga, Serie A, Bundesliga, Ligue 1, Championship, Eredivisie,
    Primeira Liga, Brasileirão, Champions League) plus the World Cup,
    the Euros and the Nations League. Leagues with matches that day
    but no prediction are listed under "unavailable": the model has
    no history for them yet, so they need FOOTBALL_DATA_API_KEY for
    live history before it can predict them. Internationals are
    predicted with the international pool (WC/EC/NL).
    """
    on = None
    if date:
        try:
            on = _date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail={
                "error": f"Invalid date '{date}': use YYYY-MM-DD"})
    on = on or _date.today()
    espn = EspnProvider()
    groups: list[dict] = []
    unavailable: list[dict] = []
    for league in (*ESPN_LEAGUE_SLUGS, "WC", "EC", "NL"):
        try:
            fixtures = espn.fixtures(league, on)
        except Exception:
            continue
        if not fixtures:
            continue
        predicted = [f for f in _predict_fixtures(fixtures, league)
                     if f["prediction"] is not None]
        if predicted:
            groups.append({"league": league,
                           "label": _LEAGUE_NAMES.get(league, league),
                           "fixtures": predicted})
        else:
            # Fixtures exist, but the model cannot predict this
            # league yet — say so instead of silently dropping it.
            unavailable.append({"league": league,
                                "label": _LEAGUE_NAMES.get(league, league),
                                "fixtures": len(fixtures)})
    if not groups and not unavailable:
        return {"date": on.isoformat(), "groups": [], "unavailable": [],
                "message": (f"No matches on {on.isoformat()} in any of the "
                            "covered competitions (Premier League, La Liga, "
                            "Serie A, Bundesliga, Ligue 1, Championship, "
                            "Eredivisie, Primeira Liga, Brasileirão, "
                            "Champions League, World Cup, Euros, Nations "
                            "League)."),
                "source": "espn"}
    return {"date": on.isoformat(), "groups": groups,
            "unavailable": unavailable, "source": "espn"}


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
