"""Chatbot parser tests: today's matches, multi-match, form, ratings,
head-to-head and next-match intents.

Runs the exact parser code from index.html under node, the same way
test_verdict.py runs the rounding helper: the block between the
__CHAT_PARSER_START__ / __CHAT_PARSER_END__ markers is extracted and
fed to node with the cases.
"""
import json
import re
import subprocess
from pathlib import Path

PARSER_PATH = (Path(__file__).resolve().parents[1]
               / "src" / "football_agent" / "static" / "index.html")


def _run_parser(cases):
    """Run parseChatMessage under node; return the list of results."""
    html = PARSER_PATH.read_text()
    match = re.search(r"// __CHAT_PARSER_START__(.*?)// __CHAT_PARSER_END__",
                      html, re.S)
    assert match, "chat parser block missing from the page"
    driver = match.group(1) + (
        "console.log(JSON.stringify(CASES.map(function (c) {"
        " return parseChatMessage(c[0], c[1], c[2]); })));"
    ).replace("CASES", json.dumps(cases))
    out = subprocess.run(["node", "-e", driver],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


# (text, pending team, default league)
CASES = [
    ["Arsenal vs Chelsea", None, "PL"],
    ["Who wins, Man Utd or Liverpool?", None, "PL"],
    ["predict today's matches", None, "PL"],
    ["What matches are on today?", None, "PL"],
    ["tomorrow's matches", None, "PL"],
    ["Today's La Liga matches", None, "PL"],
    ["Arsenal vs Chelsea and Man Utd vs Liverpool", None, "PL"],
    ["Arsenal vs Chelsea, Liverpool vs Man City", None, "PL"],
    ["Arsenal's form", None, "PL"],
    ["How is Liverpool doing?", None, "PL"],
    ["Who will win the Premier League?", None, "PL"],
    ["Who is the best team in La Liga?", None, "PL"],
    ["Arsenal vs Chelsea head to head", None, "PL"],
    ["Arsenal's next match", None, "PL"],
    ["When do Arsenal play next?", None, "PL"],
    ["Brighton and Hove Albion vs Chelsea", None, "PL"],
    ["hi", None, "PL"],
    ["help", None, "PL"],
    ["Arsenal", None, "PL"],
    ["Chelsea", "Arsenal", "PL"],
    ["this weekend", None, "PL"],
    ["Real Madrid vs Barcelona", None, "PD"],
    ["Arsenal vs Chelsea today", None, "PL"],
    ["Argentina vs Brazil", None, "WC"],
]


def test_chat_parser_kinds():
    results = _run_parser(CASES)
    assert len(results) == len(CASES)
    kinds = [r["kind"] for r in results]
    assert kinds == [
        "predict", "predict",
        "today", "today", "today", "today",
        "multi", "multi",
        "form", "form",
        "ratings", "ratings",
        "h2h",
        "next", "next",
        "predict",
        "greeting", "help",
        "pending", "predict",
        "weekend",
        "predict", "predict", "predict",
    ]


def test_chat_parser_fields():
    results = _run_parser(CASES)
    by_text = dict(zip([c[0] for c in CASES], results))

    today = by_text["predict today's matches"]
    assert today["when"] == "today" and today["league"] == "PL"
    assert by_text["tomorrow's matches"]["when"] == "tomorrow"
    assert by_text["Today's La Liga matches"]["league"] == "PD"

    multi = by_text["Arsenal vs Chelsea and Man Utd vs Liverpool"]
    assert multi["pairs"] == [["Arsenal", "Chelsea"],
                              ["Man Utd", "Liverpool"]]
    assert multi["league"] == "PL"

    assert by_text["Arsenal's form"] == {
        "kind": "form", "team": "Arsenal", "league": "PL"}
    assert by_text["How is Liverpool doing?"]["team"] == "Liverpool"

    assert by_text["Who will win the Premier League?"] == {
        "kind": "ratings", "league": "PL"}
    assert by_text["Who is the best team in La Liga?"]["league"] == "PD"

    assert by_text["Arsenal vs Chelsea head to head"] == {
        "kind": "h2h", "home": "Arsenal", "away": "Chelsea",
        "league": "PL"}

    assert by_text["Arsenal's next match"] == {
        "kind": "next", "team": "Arsenal", "league": "PL"}

    # "Brighton and Hove Albion" is one team, not a split on "and".
    brighton = by_text["Brighton and Hove Albion vs Chelsea"]
    assert brighton == {"kind": "predict", "home": "Brighton",
                        "away": "Chelsea", "league": "PL"}

    # Pending flow: one team, then the second team completes the match.
    assert by_text["Arsenal"] == {"kind": "pending", "team": "Arsenal"}
    assert by_text["Chelsea"] == {"kind": "predict", "home": "Arsenal",
                                  "away": "Chelsea", "league": "PL"}

    # Teams outside the embedded list still parse via split sides.
    madrid = by_text["Real Madrid vs Barcelona"]
    assert madrid["home"] == "real madrid" and madrid["away"] == "barcelona"
    assert madrid["league"] == "PD"

    # "today" after a named match is a tail word, not a today request.
    assert by_text["Arsenal vs Chelsea today"]["kind"] == "predict"

    # The league follows the dropdown when the message names none.
    assert by_text["Argentina vs Brazil"]["league"] == "WC"
