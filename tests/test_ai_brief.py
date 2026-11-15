"""AI match briefing tests: POST /ai-brief, AgentRouter primary with a
Xingchi fallback.

Live web search: the /v1/responses call carries the web_search tool and
sources are extracted from the answer's markdown links and from the
web_search_call output items. If AgentRouter fails and a Xingchi key is
set, the briefing falls back to Xingchi's /v1/chat/completions
(gpt-6.1-sol, knowledge-only: live_search false, no sources). No real
keys and no network: football_agent.api.requests.post (both gateways)
is patched.
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


FAKE_XINGCHI_OK = {
    "choices": [{"message": {"role": "assistant", "content": BRIEFING_JSON}}],
}

FAKE_XINGCHI_OK_NO_JSON = {
    "choices": [{"message": {"role": "assistant",
                             "content": "Arsenal are in good form lately."}}],
}


def _dispatch(mapping):
    """requests.post side_effect dispatching on the URL called."""
    def _call(url, **kwargs):
        for prefix, resp in mapping.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unexpected URL: {url}")
    return _call


def _set_key(monkeypatch):
    monkeypatch.setenv("AGENTROUTER_API_KEY", "agentrouter-test-key")


def _set_keys(monkeypatch, agentrouter=True, xingchi=True):
    if agentrouter:
        monkeypatch.setenv("AGENTROUTER_API_KEY", "agentrouter-test-key")
    else:
        monkeypatch.delenv("AGENTROUTER_API_KEY", raising=False)
    if xingchi:
        monkeypatch.setenv("XINGCHI_API_KEY", "xingchi-test-key")
    else:
        monkeypatch.delenv("XINGCHI_API_KEY", raising=False)


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
        # the prompt tells the model to run one search only and use
        # only that search result.
        assert sent["tools"] == [{"type": "web_search"}]
        assert "web search" in sent["input"].lower()
        assert "one search only" in sent["input"].lower()
        assert "do not run more searches" in sent["input"].lower()
        assert "Use ONLY what you find in the web search result" in sent["input"]
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


def test_ai_brief_agentrouter_failure_falls_back_to_xingchi(monkeypatch):
    """AgentRouter 500 + Xingchi success => knowledge-only briefing."""
    _set_keys(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=_dispatch({
                api.AGENTROUTER_RESPONSES_URL: _FakeResponse(
                    500, {"error": {"message": "gateway error"}}),
                api.XINGCHI_CHAT_URL: _FakeResponse(200, FAKE_XINGCHI_OK),
            })) as mock_post:
        r = _post_brief(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Arsenal"
    assert "good form" in body["summary"]
    assert body["form"] == ["Arsenal: W W D W W", "Chelsea: W D W W L"]
    # The fallback is honest: no live search, no live sources.
    assert body["live_search"] is False
    assert body["sources"] == []
    assert "xingchi-test-key" not in r.text
    assert "agentrouter-test-key" not in r.text
    # Three calls: AgentRouter first, its one retry, then Xingchi.
    assert mock_post.call_count == 3
    ar_args, ar_kwargs = mock_post.call_args_list[0]
    assert ar_args[0] == api.AGENTROUTER_RESPONSES_URL
    assert ar_kwargs["json"]["tools"] == [{"type": "web_search"}]
    xc_args, xc_kwargs = mock_post.call_args_list[2]
    assert xc_args[0] == api.XINGCHI_CHAT_URL
    assert xc_kwargs["headers"]["Authorization"] == "Bearer xingchi-test-key"
    assert xc_kwargs["headers"]["User-Agent"] == (
        "claude-cli/2.1.220 (external, claude-desktop)")
    assert xc_kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert xc_kwargs["timeout"] == 30
    sent = xc_kwargs["json"]
    assert sent["model"] == "gpt-6.1-sol"
    assert "tools" not in sent
    content = sent["messages"][0]["content"]
    assert sent["messages"][0]["role"] == "user"
    assert "no web access" in content
    assert "Arsenal" in content and "Chelsea" in content
    # A repeat request for the same pair is served from cache.
    r2 = _post_brief(client)
    assert r2.status_code == 200
    assert mock_post.call_count == 3


def test_brief_prompt_is_single_search():
    """The briefing prompt asks for exactly one search and keeps the
    same JSON keys and rules as the verified-live prompt."""
    prompt = api._brief_prompt("Arsenal", "Chelsea", "PL", "2026-11-16")
    assert "one search only" in prompt.lower()
    assert "do not run more searches" in prompt.lower()
    assert "'Arsenal vs Chelsea latest news injuries form'" in prompt
    assert "Premier League" in prompt
    for key in ('"summary"', '"form"', '"injuries"', '"news"'):
        assert key in prompt
    assert "Cite every factual claim" in prompt
    assert "Do NOT invent players, results, or injuries" in prompt
    assert "Do NOT predict a winner, scoreline, or probabilities" in prompt
    assert "Do NOT mention odds" in prompt


def test_ai_brief_fast_failure_retried_once_then_succeeds(monkeypatch):
    """A fast AgentRouter failure is retried once; a retry success
    returns a live briefing."""
    _set_key(monkeypatch)
    calls = {"n": 0}

    def flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(
                500, {"error": {"message": "gateway error"}})
        return _FakeResponse(200, FAKE_AGENTROUTER_OK)

    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=flaky) as mock_post:
        r = _post_brief(client)
    _assert_briefing_body(r)
    assert mock_post.call_count == 2
    assert all(c[0][0] == api.AGENTROUTER_RESPONSES_URL
               for c in mock_post.call_args_list)


def test_build_brief_slow_failure_not_retried():
    """A slow AgentRouter failure is not retried (one call only).

    The fake clock advances 31s inside the failing call, so the
    elapsed time is past the 30s fast-retry window; _build_brief is
    called directly to keep the fake clock out of the HTTP stack.
    """
    calls = {"n": 0}
    clock = {"t": 1000.0}

    def fake_monotonic():
        return clock["t"]

    def failing(*args, **kwargs):
        calls["n"] += 1
        clock["t"] += 31.0  # the call took 31s
        raise api._BriefError("AgentRouter returned HTTP 500")

    with patch("football_agent.api.time.monotonic",
               side_effect=fake_monotonic), patch(
            "football_agent.api._summarise_with_agentrouter",
            side_effect=failing):
        with pytest.raises(api._BriefError):
            api._build_brief("Arsenal", "Chelsea", "PL",
                             "agentrouter-test-key", None)
    assert calls["n"] == 1


def test_ai_brief_retry_failure_falls_back_to_xingchi(monkeypatch):
    """AgentRouter fails twice (first try + one retry), then Xingchi
    answers with a knowledge-only briefing."""
    _set_keys(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=_dispatch({
                api.AGENTROUTER_RESPONSES_URL: _FakeResponse(
                    500, {"error": {"message": "gateway error"}}),
                api.XINGCHI_CHAT_URL: _FakeResponse(200, FAKE_XINGCHI_OK),
            })) as mock_post:
        r = _post_brief(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["live_search"] is False
    assert body["sources"] == []
    assert "good form" in body["summary"]
    # Three calls: AgentRouter, its one retry, then Xingchi.
    assert mock_post.call_count == 3
    assert mock_post.call_args_list[0][0][0] == api.AGENTROUTER_RESPONSES_URL
    assert mock_post.call_args_list[1][0][0] == api.AGENTROUTER_RESPONSES_URL
    assert mock_post.call_args_list[2][0][0] == api.XINGCHI_CHAT_URL


def test_ai_brief_both_fail_502(monkeypatch):
    _set_keys(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=_dispatch({
                api.AGENTROUTER_RESPONSES_URL: _FakeResponse(
                    500, {"error": {"message": "gateway error"}}),
                api.XINGCHI_CHAT_URL: _FakeResponse(
                    503, {"error": {"message": "Service temporarily unavailable"}}),
            })):
        r = _post_brief(client)
    assert r.status_code == 502, r.text
    assert "unavailable" in r.json()["detail"]["error"]


def test_ai_brief_xingchi_403_falls_through_to_502(monkeypatch):
    """A Xingchi 403 (no fingerprint / bad key) is a 502, not a 503."""
    _set_keys(monkeypatch)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=_dispatch({
                api.AGENTROUTER_RESPONSES_URL: _FakeResponse(
                    500, {"error": {"message": "gateway error"}}),
                api.XINGCHI_CHAT_URL: _FakeResponse(
                    403, {"error": {"code": "client_not_allowed",
                                    "message": "forbidden"}}),
            })):
        r = _post_brief(client)
    assert r.status_code == 502, r.text


def test_ai_brief_xingchi_only_key_uses_fallback(monkeypatch):
    """Only a Xingchi key set: the fallback answers directly."""
    _set_keys(monkeypatch, agentrouter=False, xingchi=True)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            side_effect=_dispatch({
                api.XINGCHI_CHAT_URL: _FakeResponse(200, FAKE_XINGCHI_OK),
            })) as mock_post:
        r = _post_brief(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["live_search"] is False
    assert body["sources"] == []
    assert "good form" in body["summary"]
    assert mock_post.call_count == 1
    assert mock_post.call_args[0][0] == api.XINGCHI_CHAT_URL


def test_ai_brief_no_keys_503(monkeypatch):
    _set_keys(monkeypatch, agentrouter=False, xingchi=False)
    with _client() as client, patch(
            "football_agent.api.requests.post") as mock_post:
        r = _post_brief(client)
    assert r.status_code == 503, r.text
    assert "not configured" in r.json()["detail"]["error"]
    assert "AGENTROUTER_API_KEY" in r.json()["detail"]["hint"]
    assert "XINGCHI_API_KEY" in r.json()["detail"]["hint"]
    mock_post.assert_not_called()


def test_ai_brief_agentrouter_fails_no_xingchi_key_502(monkeypatch):
    """AgentRouter fails (first try + one retry) and no Xingchi key: 502."""
    _set_keys(monkeypatch, agentrouter=True, xingchi=False)
    with _client() as client, patch(
            "football_agent.api.requests.post",
            return_value=_FakeResponse(
                500, {"error": {"message": "gateway error"}})) as mock_post:
        r = _post_brief(client)
    assert r.status_code == 502, r.text
    assert mock_post.call_count == 2
    assert mock_post.call_args[0][0] == api.AGENTROUTER_RESPONSES_URL


@pytest.mark.parametrize("status", [401, 429, 500])
def test_ai_brief_http_error_502(monkeypatch, status):
    _set_key(monkeypatch)
    monkeypatch.delenv("XINGCHI_API_KEY", raising=False)
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
    # The page says the briefing is a live web search when AgentRouter
    # answers, and says it is knowledge-only when the fallback does.
    assert "live web search" in html.lower()
    assert "Searched the web" in html
    assert "From the AI’s own knowledge, not a live web search" in html
    assert "own knowledge" in html
    assert "safe bet" not in html.lower()
    assert "guaranteed win" not in html.lower()
