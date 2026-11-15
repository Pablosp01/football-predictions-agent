"""AI match briefing tests: POST /ai-brief with AgentRouter only.

Live web search: the /v1/responses call carries the web_search tool and
sources are extracted from the answer's markdown links and from the
web_search_call output items. No real keys and no network:
football_agent.api.requests.post (AgentRouter) is patched.
"""
import json
import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402


def _client():
    api.clear_cache()
    api.clear_brief_cache()
    return TestClient(api.app)


BRIEFING_JSON = (
    '{"summary": "Arsenal host Chelsea with both sides in good form, '
    'per [ESPN](https://www.espn.com/soccer/team/_/id/359/arsenal).", '
    '"form": ["Arsenal: W W D W W", "Chelsea: W D W W L"], '
    '"injuries": ["Bukayo Saka (Arsenal) — hamstring, doubt"], '
    '"news": ["Chelsea welcome back their captain."]}'
)

# The live shape: output_text plus web_search_call output items. The
# ESPN link also appears in the answer text, so it must be deduped.
FAKE_AGENTROUTER_OK = {
    "output_text": BRIEFING_JSON,
    "output": [
        {"type": "web_search_call",
         "action": {"type": "search",
                    "query": "Arsenal vs Chelsea form injuries"}},
        {"type": "message", "content": [
            {"type": "output_text",
             "text": BRIEFING_JSON,
             "annotations": [
                 {"type": "url_citation",
                  "title": "Arsenal - ESPN",
                  "url": "https://www.espn.com/soccer/team/_/id/359/arsenal"},
                 {"type": "url_citation",
                  "title": "Chelsea - ESPN",
                  "url": "https://www.espn.com/soccer/team/_/id/363/chelsea"},
             ]},
        ]},
    ],
}

FAKE_AGENTROUTER_OUTPUT_SHAPE = {
    "output": [{"type": "message", "content": [
        {"type": "output_text", "text": BRIEFING_JSON},
    ]}],
}


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def _set_key(monkeypatch):
    monkeypatch.setenv("AGENTROUTER_API_KEY", "agentrouter-test-key")


def _post_brief(client):
    return client.post("/ai-brief",
                       json={"home_team": "Arsenal",
                             "away_team": "Chelsea"})


def _assert_briefing_body(r, sources=None):
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Arsenal"
    assert body["away_team"] == "Chelsea"
    assert body["league"] == "PL"
    assert "good form" in body["summary"]
    assert body["form"] == ["Arsenal: W W D W W", "Chelsea: W D W W L"]
    assert body["injuries"] == ["Bukayo Saka (Arsenal) — hamstring, doubt"]
    assert body["news"] == ["Chelsea welcome back their captain."]
    # Live web search: sources are the cited pages, deduped, and the
    # markdown link is shown as plain text in the summary.
    assert body["live_search"] is True
    assert body["summary"] == (
        "Arsenal host Chelsea with both sides in good form, per ESPN.")
    if sources is None:
        sources = [
            {"title": "ESPN",
             "url": "https://www.espn.com/soccer/team/_/id/359/arsenal"},
            {"title": "Chelsea - ESPN",
             "url": "https://www.espn.com/soccer/team/_/id/363/chelsea"},
        ]
    assert body["sources"] == sources
    assert body["searched_at"]
    return body


def test_ai_brief_success_agentrouter(monkeypatch):
    _set_key(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(200, FAKE_AGENTROUTER_OK)) as mock_post:
        r = _post_brief(client)
        _assert_briefing_body(r)
        # The key may never appear in the response.
        assert "agentrouter-test-key" not in r.text
        # One AgentRouter call, to /v1/responses, with the fingerprint.
        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert args[0] == api.AGENTROUTER_RESPONSES_URL
        assert kwargs["headers"]["Authorization"] == "Bearer agentrouter-test-key"
        assert kwargs["headers"]["User-Agent"] == (
            "claude-cli/2.1.220 (external, claude-desktop)")
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
        assert kwargs["timeout"] == 90
        sent = kwargs["json"]
        assert sent["model"] == api.AGENTROUTER_MODEL
        assert "Arsenal" in sent["input"] and "Chelsea" in sent["input"]
        # The call searches the web: the web_search tool is attached and
        # the prompt tells the model to use only the search results.
        assert sent["tools"] == [{"type": "web_search"}]
        assert "web search" in sent["input"].lower()
        assert "Use ONLY what you find in the web search results" in sent["input"]
        # A repeat request for the same pair is served from cache.
        r2 = _post_brief(client)
        assert r2.status_code == 200
        assert mock_post.call_count == 1


def test_ai_brief_sources_from_web_search_call(monkeypatch):
    """Sources are also taken from web_search_call output items."""
    _set_key(monkeypatch)
    payload = {
        "output_text": BRIEFING_JSON,
        "output": [
            {"type": "web_search_call",
             "action": {"type": "search",
                        "query": "Arsenal Chelsea injuries"}},
            {"type": "web_search_call",
             "results": [
                 {"title": "BBC Sport",
                  "url": "https://www.bbc.com/sport/football/arsenal"},
                 {"title": "ESPN",
                  "url": "https://www.espn.com/soccer/team/_/id/359/arsenal"},
             ]},
        ],
    }
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(200, payload)):
        r = _post_brief(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["live_search"] is True
    assert body["sources"] == [
        {"title": "ESPN",
         "url": "https://www.espn.com/soccer/team/_/id/359/arsenal"},
        {"title": "BBC Sport",
         "url": "https://www.bbc.com/sport/football/arsenal"},
    ]


def test_ai_brief_success_output_shape_fallback(monkeypatch):
    _set_key(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(
                200, FAKE_AGENTROUTER_OUTPUT_SHAPE)):
        r = _post_brief(client)
        # This fixture carries only the markdown link, so only the
        # Arsenal source is extracted.
        _assert_briefing_body(r, sources=[
            {"title": "ESPN",
             "url": "https://www.espn.com/soccer/team/_/id/359/arsenal"},
        ])


def test_ai_brief_plain_text_falls_back_to_summary(monkeypatch):
    _set_key(monkeypatch)
    payload = {"output_text": "Arsenal are in good form lately."}
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(200, payload)):
        r = _post_brief(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"] == "Arsenal are in good form lately."
    assert body["form"] == [] and body["injuries"] == [] and body["news"] == []
    assert body["live_search"] is True
    assert body["sources"] == []


def test_ai_brief_missing_key_503(monkeypatch):
    monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    with _client() as client, patch(
            "football_agent.api.requests.post") as mock_post:
        r = _post_brief(client)
    assert r.status_code == 503, r.text
    assert "not configured" in r.json()["detail"]["error"]
    assert "AGENTROUTER_API_KEY" in r.json()["detail"]["hint"]
    mock_post.assert_not_called()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_ai_brief_http_error_502(monkeypatch, status):
    _set_key(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(
                status, {"error": {"message": "gateway error"}})):
        r = _post_brief(client)
    assert r.status_code == 502, r.text
    assert "unavailable" in r.json()["detail"]["error"]


@pytest.mark.parametrize("exc", [
    requests.exceptions.ConnectTimeout("timed out"),
    requests.exceptions.Timeout("timed out"),
    requests.exceptions.ConnectionError("no route"),
])
def test_ai_brief_transport_error_502(monkeypatch, exc):
    _set_key(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post", side_effect=exc):
        r = _post_brief(client)
    assert r.status_code == 502, r.text


def test_ai_brief_empty_answer_502(monkeypatch):
    _set_key(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(200, {"output_text": ""})):
        r = _post_brief(client)
    assert r.status_code == 502, r.text


def test_ai_brief_same_team_400(monkeypatch):
    _set_key(monkeypatch)
    with _client() as client:
        r = client.post("/ai-brief",
                        json={"home_team": "Arsenal", "away_team": "Arsenal"})
    assert r.status_code == 400


def test_root_page_has_briefing_section():
    with _client() as client:
        html = client.get("/").text
    assert "/ai-brief" in html
    assert "AI match briefing" in html
    assert "ai-brief" in html.lower()
    # The page says the briefing is a live web search.
    assert "live web search" in html.lower()
    assert "Searched the web" in html
    assert "not a live web search" not in html
    assert "own knowledge" not in html
    assert "safe bet" not in html.lower()
    assert "guaranteed win" not in html.lower()
