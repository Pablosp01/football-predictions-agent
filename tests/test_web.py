"""Web chat page tests: GET / serves the page, API endpoints still work.

Runs against the bundled offline sample data (FOOTBALL_DATA_API_KEY empty).
"""
import os
import re

from fastapi.testclient import TestClient

os.environ["FOOTBALL_DATA_API_KEY"] = ""
os.environ.pop("API_KEY", None)

from football_agent import api  # noqa: E402


def _client():
    api.clear_cache()
    return TestClient(api.app)


def test_root_serves_chat_page():
    with _client() as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    html = r.text
    assert "Football Predictions Agent" in html
    assert 'id="predict-form"' in html
    assert 'id="home-input"' in html
    assert 'id="away-input"' in html
    assert 'id="league-select"' in html
    assert 'id="chat-log"' in html
    assert 'id="predict-button"' in html


def test_root_page_calls_health_then_predict():
    """The page's JS must wake the service with /health before /predict,
    and POST to /predict with the team names."""
    with _client() as client:
        html = client.get("/").text
    assert '"/health"' in html
    assert '"/predict"' in html
    assert "fetchWithTimeout" in html
    # Health is called before predict in the flow (wake, then predict).
    assert html.index('"/health"') < html.index('"/predict"')


def test_root_page_has_no_betting_language():
    with _client() as client:
        html = client.get("/").text.lower()
    assert "safe bet" not in html
    assert "guaranteed win" not in html


def test_chatbot_panel_has_wider_chat_examples():
    """The Chatbot tab offers today's matches, form and league-winner
    example chips, and its help text names them."""
    with _client() as client:
        html = client.get("/").text
    assert "Today's matches" in html
    assert "Arsenal's form" in html
    assert "Who will win the Premier League?" in html
    assert "today’s matches" in html
    assert "head to head" in html
    # The new question endpoints are wired into the page JS.
    assert '"/fixtures?league="' in html
    assert '"/today?date="' in html
    assert '"/team?team="' in html
    assert '"/ratings?league="' in html
    assert '"/head-to-head?home_team="' in html
    assert '"/predict-batch"' in html


def test_page_is_not_cached():
    """The page must never be served from a browser cache: it is the
    whole app, and a deploy changes it. A cached copy after a deploy
    shows the old dropdown and the old chatbot (false failure on
    2026-11-16)."""
    with _client() as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")


def test_page_has_no_handler_name_collisions():
    """A page function must not share its name with a DOM-element
    variable: the element assignment overwrites the hoisted function,
    so calling it throws "not a function" and the question dies
    silently. That was the 2026-11-16 form bug (chatForm was both the
    form element and the form-answer function)."""
    with _client() as client:
        html = client.get("/").text
    element_vars = set(re.findall(
        r"var\s+(\w+)\s*=\s*document\.getElementById", html))
    functions = set(re.findall(r"function\s+(\w+)\s*\(", html))
    assert not (element_vars & functions), element_vars & functions


def test_health_and_docs_still_work():
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


def test_predict_still_works_via_test_client():
    with _client() as client:
        r = client.post("/predict",
                        json={"home_team": "Arsenal", "away_team": "Chelsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["home_team"] == "Arsenal"
    assert body["most_likely"] in ("H", "D", "A")
