"""ESPN scoreboard provider tests: live fixtures with no key.

The provider reads the public ESPN scoreboard (no API key) for one
day per competition. Only state "pre" (scheduled) events are
fixtures; live and finished events are not. Results are cached
30 minutes per competition-day.
"""
from datetime import date, timedelta
from unittest.mock import patch

from football_agent.data.providers import (
    EspnProvider, clear_espn_cache,
)
from football_agent.data.types import Match


def _event(when, state, home, away):
    return {
        "date": when,
        "status": {"type": {"state": state}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"displayName": home}},
            {"homeAway": "away", "team": {"displayName": away}},
        ]}],
    }


# Mon 16 Nov 2026, UEFA Nations League: two scheduled, one finished.
ESPN_NATIONS_DAY = {"events": [
    _event("2026-11-16T17:00Z", "pre", "Iceland", "Luxembourg"),
    _event("2026-11-16T19:45Z", "pre", "Germany", "Netherlands"),
    _event("2026-11-16T15:00Z", "post", "Faroe Islands", "Moldova"),
]}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _fake_get(url, timeout=None):
    if "uefa.nations" in url:
        return _FakeResponse(ESPN_NATIONS_DAY)
    return _FakeResponse({"events": []})


def test_espn_day_returns_only_scheduled_fixtures():
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_get):
        intl = EspnProvider().intl_fixtures(date(2026, 11, 16))
    assert len(intl) == 2  # the finished match is not a fixture
    assert intl[0] == Match(date=date(2026, 11, 16), season="",
                            league="WC", home_team="Iceland",
                            away_team="Luxembourg")
    assert intl[1].home_team == "Germany"
    assert intl[1].away_team == "Netherlands"
    assert all(not m.played for m in intl)


def test_espn_day_is_cached():
    clear_espn_cache()
    calls = {"n": 0}

    def counting(url, timeout=None):
        calls["n"] += 1
        return _fake_get(url, timeout)

    with patch("football_agent.data.providers.requests.get",
               side_effect=counting):
        EspnProvider().intl_fixtures(date(2026, 11, 16))
        EspnProvider().intl_fixtures(date(2026, 11, 16))
    assert calls["n"] == 1


def test_espn_fixtures_without_date_scans_ten_days():
    """No date: the next 10 days are scanned for a league's fixtures."""
    clear_espn_cache()

    def dated(url, timeout=None):
        # A fixture only on 2026-11-21 (five days out).
        if "dates=20261121" in url:
            return _FakeResponse({"events": [
                _event("2026-11-21T15:00Z", "pre",
                       "Arsenal", "Newcastle United")]})
        return _FakeResponse({"events": []})

    class _FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 11, 16)

    with (patch("football_agent.data.providers.requests.get",
                side_effect=dated),
          patch("football_agent.data.providers.date", _FixedDate)):
        fixtures = EspnProvider().fixtures("PL")
    assert len(fixtures) == 1
    assert fixtures[0].date == date(2026, 11, 21)
    assert fixtures[0].home_team == "Arsenal"
    assert fixtures[0].away_team == "Newcastle United"


def test_espn_unknown_league_returns_empty():
    clear_espn_cache()
    with patch("football_agent.data.providers.requests.get",
               side_effect=_fake_get):
        assert EspnProvider().fixtures("WC", date(2026, 11, 16)) == []


def test_espn_transport_failure_raises():
    clear_espn_cache()

    def failing(url, timeout=None):
        raise ConnectionError("down")

    with patch("football_agent.data.providers.requests.get",
               side_effect=failing):
        try:
            EspnProvider().intl_fixtures(date(2026, 11, 16))
        except ConnectionError:
            return
        raise AssertionError("expected the transport failure to raise")
