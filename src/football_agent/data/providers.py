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
# club-team Elo.
INTERNATIONAL_LEAGUES = {"WC", "EC"}

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


def get_provider(league: str = "PL") -> Provider:
    """Prefer the live API when a key is set, else the bundled sample data.

    WC/EC use the international sample pool (national teams only);
    club leagues use the Premier League sample file.
    """
    if os.environ.get("FOOTBALL_DATA_API_KEY"):
        try:
            return FootballDataOrgProvider()
        except ValueError:
            pass
    if league in INTERNATIONAL_LEAGUES:
        return InternationalSampleProvider()
    return SampleProvider()
