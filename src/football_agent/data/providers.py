"""Pluggable data providers.

Primary: football-data.org (free tier, key via FOOTBALL_DATA_API_KEY).
Fallback: bundled sample CSV + generic CSV loader, so everything works offline.
API responses are cached under data/cache/ (JSON, 6h TTL).
"""
from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import requests

from .types import Match

LEAGUE_CODES = {"PL": "PL", "La Liga": "PD", "Serie A": "SA",
                "Bundesliga": "BL1", "Ligue 1": "FL1", "Championship": "ELC",
                "Eredivisie": "DED", "Primeira Liga": "PPL",
                "Campeonato Brasileiro Série A": "BSA", "UCL": "CL",
                "European Championship": "EC", "FIFA World Cup": "WC"}
CODE_TO_LEAGUE = {v: k for k, v in LEAGUE_CODES.items()}

# football-data.org free tier (TIER_ONE) covers exactly these 12
# competitions: the top 5 leagues + Championship, Eredivisie, Primeira
# Liga, Brasileirão Série A, Champions League, and the two international
# tournaments below. Verified 2026-11-15 against the official docs
# (WC/EC/BSA are TIER_ONE) and the free-plan lists published by
# freeapihub (2026-11-08) and thestatsapi/highlightly (2026).
FREE_TIER_COMPETITIONS = {
    "PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
    "BL1": "Bundesliga", "FL1": "Ligue 1", "ELC": "Championship",
    "DED": "Eredivisie", "PPL": "Primeira Liga",
    "BSA": "Campeonato Brasileiro Série A", "CL": "Champions League",
    "EC": "European Championship", "WC": "FIFA World Cup",
}

# International tournaments: national teams only. They share one rating
# pool (the bundled international sample) and are never mixed with
# club-team Elo. NL is the UEFA Nations League — it is not a
# football-data.org competition, so it has no live history source;
# the sample pool and ESPN fixtures are its only data.
INTERNATIONAL_LEAGUES = {"WC", "EC", "NL"}

DEFAULT_SAMPLE = Path(__file__).resolve().parents[3] / "data" / "sample" / "pl_matches.csv"
DEFAULT_INTERNATIONAL_SAMPLE = (Path(__file__).resolve().parents[3]
                                / "data" / "sample" / "international_matches.csv")
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class Provider(Protocol):
    def matches(self, league: str = "PL") -> list[Match]: ...
    def fixtures(self, league: str = "PL", on: date | None = None) -> list[Match]: ...


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _row_to_match(row: dict) -> Match:
    def _f(key: str) -> float | None:
        v = row.get(key)
        return float(v) if v not in (None, "", "nan") else None

    def _g(key: str) -> int | None:
        v = row.get(key)
        return int(float(v)) if v not in (None, "", "nan") else None

    return Match(
        date=_parse_date(str(row["date"])),
        season=str(row.get("season", "")),
        league=str(row.get("league", "PL")),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_goals=_g("home_goals"),
        away_goals=_g("away_goals"),
        home_odds=_f("home_odds"),
        draw_odds=_f("draw_odds"),
        away_odds=_f("away_odds"),
    )


class CsvProvider:
    """Load matches/fixtures from a CSV with the sample-file columns."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def matches(self, league: str = "PL") -> list[Match]:
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = [_row_to_match(r) for r in csv.DictReader(fh)]
        rows = [m for m in rows if m.league == league or league in ("PL", "all")]
        return sorted([m for m in rows if m.played], key=lambda m: m.date)

    def fixtures(self, league: str = "PL", on: date | None = None) -> list[Match]:
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = [_row_to_match(r) for r in csv.DictReader(fh)]
        out = [m for m in rows if not m.played]
        if on is not None:
            out = [m for m in out if m.date == on]
        return sorted(out, key=lambda m: m.date)


class SampleProvider(CsvProvider):
    """Bundled realistic Premier League sample dataset (offline fallback)."""

    def __init__(self, path: str | Path | None = None):
        super().__init__(path or DEFAULT_SAMPLE)


class InternationalSampleProvider(CsvProvider):
    """Bundled senior men's international results (offline fallback).

    One shared rating pool for the World Cup (WC) and the Euros (EC):
    national teams are never mixed with club teams. The file IS the
    pool, so matches() returns every played match in it regardless of
    the requested league code. Source: martj42/international_results
    (results.csv, public domain), filtered to competitive senior men's
    internationals — finals of the World Cup, Euros, Copa América,
    AFCON, Gold Cup and Asian Cup from 2024 onward, plus the most
    recent qualifiers / Nations League matches — 480 matches,
    2024-01-12 to 2026-10-06, 168 national teams.
    """

    def __init__(self, path: str | Path | None = None):
        super().__init__(path or DEFAULT_INTERNATIONAL_SAMPLE)

    def matches(self, league: str = "WC") -> list[Match]:
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = [_row_to_match(r) for r in csv.DictReader(fh)]
        return sorted([m for m in rows if m.played], key=lambda m: m.date)

    def fixtures(self, league: str = "WC", on: date | None = None) -> list[Match]:
        # The bundled sample holds results only; no future fixtures.
        return []


class FootballDataOrgProvider:
    """football-data.org v4, free tier (10 req/min). Key via env var.

    Endpoint: GET /v4/competitions/{code}/matches?season=YYYY
    Cached for 6 hours in data/cache/.
    """

    BASE = "https://api.football-data.org/v4"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = None):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise ValueError("FOOTBALL_DATA_API_KEY is not set (see .env.example)")
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, params: dict | None = None) -> dict:
        cache_key = (path + "|" + json.dumps(params or {}, sort_keys=True)).replace("/", "_")
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            stamp = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - stamp < timedelta(hours=6):
                return json.loads(cache_file.read_text(encoding="utf-8"))
        resp = requests.get(f"{self.BASE}{path}", headers={"X-Auth-Token": self.api_key},
                            params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def matches(self, league: str = "PL") -> list[Match]:
        code = LEAGUE_CODES.get(league, league)
        data = self._get(f"/competitions/{code}/matches", {"status": "FINISHED"})
        return sorted((self._api_to_match(m) for m in data.get("matches", [])),
                      key=lambda m: m.date)

    def fixtures(self, league: str = "PL", on: date | None = None) -> list[Match]:
        code = LEAGUE_CODES.get(league, league)
        params: dict = {"status": "SCHEDULED"}
        if on is not None:
            params["dateFrom"] = on.isoformat()
            params["dateTo"] = on.isoformat()
        data = self._get(f"/competitions/{code}/matches", params)
        return sorted((self._api_to_match(m) for m in data.get("matches", [])),
                      key=lambda m: m.date)

    @staticmethod
    def _api_to_match(m: dict) -> Match:
        score = m.get("score", {}).get("fullTime", {}) or {}
        return Match(
            date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).date(),
            season=str(m.get("season", {}).get("startDate", "")[:4]),
            league=CODE_TO_LEAGUE.get(m.get("competition", {}).get("code", ""), "PL"),
            home_team=m["homeTeam"]["name"],
            away_team=m["awayTeam"]["name"],
            home_goals=score.get("home"),
            away_goals=score.get("away"),
        )


# ESPN scoreboard (free, no key): the day's fixtures per competition.
# Verified 2026-11-16: the scoreboard returns the day's events with
# state "pre" (scheduled), "in" (live) or "post" (finished); only
# "pre" events are fixtures. Club leagues map to our league codes;
# the UEFA Nations League is the international competition that
# actually plays on international-break days (e.g. Mon 16 Nov 2026:
# 10 Nations League matches, no club football at all). ESPN team
# names are display names ("Manchester City", "Newcastle United") —
# the API's resolve_team() maps them to the model's canonical names.
ESPN_LEAGUE_SLUGS = {
    "PL": "eng.1", "PD": "esp.1", "SA": "ita.1", "BL1": "ger.1",
    "FL1": "fra.1", "DED": "ned.1", "PPL": "por.1", "BSA": "bra.1",
    "CL": "uefa.champions",
}
ESPN_INTL_SLUGS = ("uefa.nations",)
ESPN_CACHE_TTL = timedelta(minutes=30)
_espn_cache: dict[str, tuple[datetime, list[Match]]] = {}


def clear_espn_cache() -> None:
    """For tests: drop cached ESPN scoreboard days."""
    _espn_cache.clear()


def _espn_day(slug: str, on: date, league: str) -> list[Match]:
    """One ESPN scoreboard day: scheduled fixtures only, cached 30 min.

    Raises on transport failure so the caller can fall back to the
    bundled sample (which has no future fixtures and says so).
    """
    key = f"{slug}|{on.isoformat()}"
    now = datetime.now()
    hit = _espn_cache.get(key)
    if hit and now - hit[0] < ESPN_CACHE_TTL:
        return hit[1]
    url = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
           f"{slug}/scoreboard?dates={on.strftime('%Y%m%d')}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    out: list[Match] = []
    # Kickoff order: the day's matches in the order they happen.
    events = sorted(resp.json().get("events", []),
                    key=lambda e: e.get("date", ""))
    for event in events:
        if event.get("status", {}).get("type", {}).get("state") != "pre":
            continue  # live or finished: not a fixture to predict
        comps = event.get("competitions", [{}])[0]
        teams = comps.get("competitors", [])
        home = next((t["team"].get("displayName") for t in teams
                     if t.get("homeAway") == "home"), None)
        away = next((t["team"].get("displayName") for t in teams
                     if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        day = datetime.fromisoformat(
            event["date"].replace("Z", "+00:00")).date()
        out.append(Match(date=day, season="", league=league,
                         home_team=home, away_team=away))
    # Stable by date: within a day the kickoff order above is kept.
    out.sort(key=lambda m: m.date)
    _espn_cache[key] = (now, out)
    return out


class EspnProvider:
    """Free ESPN scoreboard fixtures (no key) — fixtures only.

    History still comes from the sample or football-data.org; ESPN
    only answers "what matches are on this day".
    """

    def fixtures(self, league: str = "PL",
                 on: date | None = None) -> list[Match]:
        if league == "NL":
            # The Nations League is an international competition, so
            # its fixtures come from the intl feed, not a club slug.
            if on is not None:
                return self.intl_fixtures(on, league="NL")
            out: list[Match] = []
            today = date.today()
            for i in range(10):
                out.extend(self.intl_fixtures(
                    today + timedelta(days=i), league="NL"))
            return sorted(out, key=lambda m: m.date)
        slug = ESPN_LEAGUE_SLUGS.get(league)
        if slug is None:
            return []
        if on is not None:
            return _espn_day(slug, on, league)
        # No date: the next 10 days' scheduled fixtures (the "next
        # match" flow) — a league plays roughly weekly, so this is
        # a couple of matchdays, not a whole season.
        out: list[Match] = []
        today = date.today()
        for i in range(10):
            out.extend(_espn_day(slug, today + timedelta(days=i), league))
        # Stable by date: within a day the kickoff order is kept.
        return sorted(out, key=lambda m: m.date)

    def intl_fixtures(self, on: date, league: str = "WC") -> list[Match]:
        """International matches on a day (UEFA Nations League)."""
        out: list[Match] = []
        for slug in ESPN_INTL_SLUGS:
            out.extend(_espn_day(slug, on, league))
        return sorted(out, key=lambda m: m.date)


def get_provider(league: str = "PL") -> Provider:
    """Prefer the live API when a key is set, else the bundled sample data.

    WC/EC use the international sample pool (national teams only);
    club leagues use the Premier League sample file. NL (Nations
    League) always uses the international sample pool: it is not a
    football-data.org competition, so no key can fetch its history.
    """
    if league == "NL":
        return InternationalSampleProvider()
    if os.environ.get("FOOTBALL_DATA_API_KEY"):
        try:
            return FootballDataOrgProvider()
        except ValueError:
            pass
    if league in INTERNATIONAL_LEAGUES:
        return InternationalSampleProvider()
    return SampleProvider()
