# Football Predictions Agent

An accurate, testable football predictions agent for Python 3.12. It combines
**Elo ratings** (K=20, home advantage 60 Elo) with a **Dixon–Coles Poisson
goal model**, blended with recent form (last 5), head-to-head, rest days, and a
logistic calibration layer. Every prediction is validated by walk-forward
backtesting — no look-ahead.

> **No guarantee of profit. This is a statistical model, not betting advice.**
> Probabilities are estimates; football is noisy and bookmaker odds include a
> margin. Never bet money you cannot afford to lose.

## Quickstart (no API key needed)

```bash
pip install -r requirements.txt
python -m pytest -q
python -m football_agent.cli predict --home Arsenal --away Chelsea
python -m football_agent.cli backtest --league PL
```

Or install the entry point: `pip install -e .` then use `football-predict`.

## Commands

| Command | What it does |
|---|---|
| `football-predict predict --home Arsenal --away Chelsea` | 1X2 probs, xG, top-5 scorelines, BTTS, over/under 1.5 / 2.5 / 3.5, model's top pick + confidence + top-3 factors |
| `football-predict fixtures --league PL --date 2026-11-15` | Predict that day's fixtures (live API) |
| `football-predict backtest --league PL` | Walk-forward backtest on history |
| `football-predict evaluate --league PL` | Backtest + plain-language verdict |
| `football-predict fetch --league PL` | Fetch live data (needs API key) |
| `football-predict agent --home Arsenal Liverpool --away Chelsea "Man City"` | Rank fixtures, write `output/predictions.csv` + `output/predictions.md` |

## Connect to ChatGPT

Give ChatGPT your team names and get a prediction **from this model**,
not from ChatGPT guessing: deploy the FastAPI wrapper, create a Custom
GPT, and add it as an Action. Full non-technical steps:
[`chatgpt/README.md`](chatgpt/README.md).

```bash
pip install -r requirements.txt
uvicorn football_agent.api:app --reload   # or: python -m football_agent.api
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

Endpoints: `GET /health`, `POST /predict`, `POST /predict-batch`,
`GET /fixtures?league=PL&date=2026-11-15`, `GET /backtest?league=PL`,
`POST /ai-brief` (AI match briefing: live web search via AgentRouter —
needs `AGENTROUTER_API_KEY`).
Team names are fuzzy-matched ("Man Utd", "Spurs" work); an unknown name
returns 400 with "did you mean" suggestions. The fitted model is cached
per league. If `API_KEY` is set, every endpoint except `/health` requires
an `X-API-Key` header. Deploy free on Render with the included
`render.yaml`, or with the included `Dockerfile`.

## Web chat page (free, no ChatGPT plan needed)

If your ChatGPT account can't create Custom GPTs, use the web chat page
served by the same API — type two team names and get the model's
prediction straight away:

- Live: **https://football-predictions-agent-agk8.onrender.com/**
- Locally: `uvicorn football_agent.api:app` then open
  http://localhost:8000/

The page is a chat: it shows a "waking the model up…" state, calls
`GET /health` first to wake the free host, then `POST /predict`, and
renders the reply as a chat bubble with win/draw/away % bars, expected
goals, most likely outcome + confidence, top-5 scorelines, BTTS and
over/under 2.5, and the model's factors. It also shows a "Model's picks"
section that states the pick in plain words, one sentence per line —
"Pick: Arsenal win (47.6%)", "Pick: Both teams to score — Yes (52.5%)",
"Pick: Over 1.5 goals (74.5%)" — followed by the supporting detail:
the match pick, BTTS Yes/No with both probabilities, and the over/under
probabilities for total-goals lines 1.5, 2.5 and 3.5, with the model's
single most confident totals call badged as the pick. All of it is model
probabilities, never betting recommendations. Unknown team names show the
"did you mean" suggestions as clickable chips; timeouts offer a Retry
button. On Render's free plan the first request after idle can take
50–90s while the service wakes; once warm, predictions take a few
seconds.

### AI match briefing (optional, one free key)

After each prediction the page also fetches `POST /ai-brief`, which
asks an AI model for match context — recent form, injuries/suspensions
and team news. The briefing never changes the model's numbers.

The briefing is a live web search: the AgentRouter call carries the
`web_search` tool, so the model searches the web for the fixture and
cites the pages it found. The response carries `live_search: true`
and the cited sources. The search itself is slow — a live call can
take up to about a minute — so the endpoint allows 95s and the page
shows a loading state for that long; briefings are cached in memory
for 15 minutes per team pair, so repeat requests are instant.

To enable it, create one free key and set it as an environment
variable on the server (Render dashboard → Environment, or
`AGENTROUTER_API_KEY=... uvicorn football_agent.api:app` locally):

- AgentRouter key — free credits, no card: register at
  https://agentrouter.org and copy the key from
  https://agentrouter.org/console/token. It calls
  `POST https://agentrouter.org/v1/responses` (its models do not
  support `/v1/chat/completions`) and sends the Claude Code client
  fingerprint headers the gateway requires.

Until the key is set, `/ai-brief` answers 503 (naming the missing
key) and the page hides the section silently. The
model is `gpt-5.6-sol` by default (override with the
`AGENTROUTER_MODEL` environment variable). If AgentRouter errors or
times out (95s total budget), the endpoint answers 502 and the page
says so without touching the prediction.

## How accuracy is measured

`backtest` uses **walk-forward, expanding-window** evaluation: for each match
after a 100-match warm-up, the model is fit only on matches played before it,
predicts it, then moves on. Reported metrics:

- **Accuracy (1X2)** — fraction of matches where the most likely outcome won.
  A naive "always home" baseline scores ~47% on the sample data, so accuracy
  alone is a weak signal.
- **Brier score** — mean squared error of the probability vector (0 = perfect;
  0.667 = always predicting 1/3 each). The honest headline metric.
- **Log loss** — penalises confident wrong predictions heavily.
- **ROI vs odds** — flat 1-unit stakes on the most-likely outcome using the
  odds in the data (only when an odds CSV is supplied / present in sample data).

Run it yourself: `python -m football_agent.cli backtest` prints the real table
on the bundled sample data.

## Data

- **Primary:** [football-data.org](https://www.football-data.org/) v4, free tier
  (10 requests/minute). The free tier (TIER_ONE) covers exactly 12
  competitions — every competition the app supports:

  | Code | Competition | Works without a key? |
  |------|-------------|----------------------|
  | PL | Premier League | Yes (bundled sample) |
  | PD | La Liga | No — needs the key |
  | SA | Serie A | No — needs the key |
  | BL1 | Bundesliga | No — needs the key |
  | FL1 | Ligue 1 | No — needs the key |
  | ELC | Championship | No — needs the key |
  | DED | Eredivisie | No — needs the key |
  | PPL | Primeira Liga | No — needs the key |
  | BSA | Campeonato Brasileiro Série A | No — needs the key |
  | CL | Champions League | No — needs the key |
  | WC | FIFA World Cup | Yes (bundled international sample) |
  | EC | European Championship | Yes (bundled international sample) |

  Get a free key at https://www.football-data.org/client/register, then:
  ```bash
  cp .env.example .env   # then set FOOTBALL_DATA_API_KEY
  export FOOTBALL_DATA_API_KEY=your_key_here
  football-predict fetch --league PL
  ```
  API responses are cached for 6h in `data/cache/`. The free tier only
  returns the current season, so with a key the model trains on this
  season's finished matches; scores are delayed, not live.
- **Fallback (offline):** bundled sample datasets
  - `data/sample/pl_matches.csv` — 480 realistic Premier League matches
    (2024-25 full season, 380 + 2025-26 first 10 rounds, 100), generated with a
    seeded round-robin schedule and team strengths so home-win rate (~47%),
    goals (~1.59 home / 1.27 away) and scorelines look like real PL data.
  - `data/sample/international_matches.csv` — 480 real senior men's
    international results, 2024-01-12 to 2026-10-06, 168 national teams.
    Source: [martj42/international_results](https://github.com/martj42/international_results)
    (`results.csv`, public domain), filtered to competitive matches:
    the finals of the 2026 World Cup, Euro 2024, Copa América 2024,
    AFCON 2023/2025, Gold Cup 2025 and AFC Asian Cup 2023, plus the
    most recent qualifiers and Nations League matches.
  Any CSV with columns `date,season,league,home_team,away_team,home_goals,
  away_goals,home_odds,draw_odds,away_odds` works via `CsvProvider`.

## International tournaments

World Cup (WC) and Euros (EC) run on their **own rating pool** of national
teams — club Elo is never mixed with international Elo. With no data key,
both use the bundled international sample above; with
`FOOTBALL_DATA_API_KEY` set, they use the live football-data.org
competition instead (current season only, so a World Cup/Euros season
may hold few matches outside tournament years).

Two things are different from club football, and the model accounts for
both:

- **Neutral venues:** 347 of the 480 bundled matches were played at a
  neutral venue and the listed "home" team is nominal, so the
  international pool is fitted with **no Elo home advantage**
  (`Predictor(home_advantage=0.0)`); club leagues keep the usual +60.
- **Only national teams are predictable offline:** a team needs at least
  one match in the bundled sample to be known (168 teams). Anything
  outside it returns a 400 with "did you mean" suggestions, same as an
  unknown club team.

## Model

1. **Elo** — ratings start at 1500, K=20, home advantage +60 Elo.
2. **Dixon–Coles Poisson** — per-team attack/defence strengths (log scale,
   mean-centred), fitted by maximum likelihood; rho correction for low scores
   (0-0, 1-0, 0-1, 1-1). Outputs a full 11×11 scoreline matrix.
3. **Adjustments** — expected goals are nudged by form difference (last 5),
   head-to-head points, and rest days (<=3 days vs >3).
4. **Calibration** — multinomial logistic regression over raw probs + Elo +
   form difference, fit on out-of-sample predictions; falls back to raw
   probabilities when data is scarce.

Output per match: P(home/draw/away), xG both teams, top-5 scorelines, P(BTTS),
P(over 2.5), confidence (max outcome prob), and the top 3 driving factors.
Goals-total markets are also priced from the scoreline distribution:
`totals` gives over/under probabilities for lines 1.5, 2.5 and 3.5
(`{"1.5": {"over": p, "under": 1-p}, ...}`), `top_pick` is the model's
single most confident totals call (`{"market": "Over 1.5",
"probability": p}`), `match_pick` is the most likely 1X2 outcome
(`{"outcome": "home"|"draw"|"away", "probability": p}`), and
`btts_pick` is the more likely BTTS side (`{"side": "yes"|"no",
"probability": p}`, ties go to "yes"). The page and CLI turn the three
picks into plain sentences: "Pick: Arsenal win (47.6%)", "Pick: Both
teams to score — Yes (52.5%)", "Pick: Over 1.5 goals (74.5%)". The legacy
`p_btts` and `p_over_25` fields are unchanged.

## Limitations

- Sample data is synthetic (seeded, realistic but not real results); live
  accuracy should be measured on real data via the free API.
- The model uses only match results — no line-ups, injuries, xG from
  Opta-style event data, or market odds as a feature.
- Low-scoring football means even a good model is ~50-55% accurate on 1X2;
  the edge, if any, is small and variance is large.
- No betting advice is offered or implied.

## Project layout

```
src/football_agent/
  data/      providers (football-data.org, CSV, sample) + Match type
  model/     elo.py, poisson.py (Dixon-Coles), features.py, calibration.py, predict.py
  backtest.py  walk-forward evaluation
  agent.py     rank fixtures -> predictions.csv / predictions.md
  cli.py       football-predict entry point
  api.py       FastAPI wrapper for ChatGPT (see "Connect to ChatGPT")
  static/index.html  free web chat page served at GET / (no ChatGPT needed)
chatgpt/     openapi.yaml, gpt-instructions.md, README.md (Custom GPT setup)
render.yaml  free Render deploy for the API
Dockerfile   container deploy for the API
data/sample/pl_matches.csv   bundled offline dataset (480 matches)
data/sample/international_matches.csv  bundled international results (480 matches)
tests/       pytest: Elo, Poisson, Dixon-Coles, no-leakage backtest, CLI smoke, API, web page
```

## License

MIT — see `LICENSE`.
