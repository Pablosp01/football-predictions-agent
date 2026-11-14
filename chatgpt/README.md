# Connect the model to ChatGPT

Goal: you type team names in ChatGPT, ChatGPT asks **your model**
(Elo + Dixon-Coles Poisson, in this repo) for the numbers, adds the latest
news from a web search as context, and shows you the model's prediction.
ChatGPT never guesses the numbers itself.

You need a **paid ChatGPT plan** (Plus / Pro / Team) to create a Custom
GPT with Actions. The API hosting below is free.

## Step 1 — Put the API on the internet (Render, free)

The ChatGPT Action needs a public URL, so the API has to live somewhere
public. The repo already has everything needed (`render.yaml`,
`Dockerfile`).

1. The code is already on GitHub: https://github.com/Pablosp01/football-predictions-agent
2. Go to https://dashboard.render.com → **New +** → **Blueprint**.
3. Connect your GitHub account, pick the `football-predictions-agent`
   repository, and click **Apply**. Render reads `render.yaml` and
   creates a free web service.
4. Wait for the deploy to finish (a few minutes), then open the service
   and copy its URL — it looks like
   `https://football-predictions-agent.onrender.com`.
   Test it in your browser: add `/health` to the end of the URL. You
   should see `{"status":"ok","model":"Elo + Dixon-Coles Poisson",...}`.
5. Optional, for real upcoming fixtures: in Render, go to the service →
   **Environment** → add `FOOTBALL_DATA_API_KEY` with a free key from
   https://www.football-data.org/client/register. Without it the API
   still predicts any two teams from the bundled sample data, but
   `/fixtures` has no future fixtures. Redeploy after adding the key.
6. Optional, to keep the API private: add an `API_KEY` environment
   variable in Render (any password-like string you choose), and when
   you set up the Action in Step 3, add it as the `X-API-Key` header
   (see "Authentication" below). If you skip this, the API is public —
   anyone with the URL can call it.

> **Free Render sleeps.** On the free plan Render spins the service down
> after ~15 minutes idle, so the first ChatGPT call after a pause can
> take ~30 seconds to wake up. That is normal — just wait, or ask
> ChatGPT to try again. A paid plan ($7/month) keeps it always on.

Alternatives if you prefer them: Railway or Fly.io — same idea, deploy
this repo with `uvicorn football_agent.api:app --host 0.0.0.0 --port $PORT`.

## Step 2 — Create the Custom GPT

1. Go to https://chatgpt.com/gpts/editor and click **Create a GPT**.
2. Give it a name, e.g. **Football Predictions Agent**, and a short
   description ("Predictions from my Elo + Dixon-Coles model").
3. Click the **Configure** tab.
4. In **Instructions**, paste the whole contents of
   [`gpt-instructions.md`](gpt-instructions.md) in this folder.
5. Turn on **Web Search** in Capabilities (so it can check injuries,
   line-ups and form). Turn off anything you do not need.

## Step 3 — Add the Action (this connects it to your model)

1. Still in **Configure**, scroll to **Actions** → **Create new action**.
2. Click **Import from URL**? No — choose the **Schema** box and paste
   the whole contents of [`openapi.yaml`](openapi.yaml) in this folder.
3. **Set the server URL.** In the schema, replace `{BASE_URL}` with the
   Render URL from Step 1 (no trailing slash), e.g.
   `https://football-predictions-agent.onrender.com`.
   Then click **Save** — if the schema is accepted you will see the
   operations `predictMatch`, `predictBatch`, `getFixtures`,
   `getBacktest` and `health` listed.
4. **Authentication (only if you set API_KEY in Step 1):** in the Action
   settings choose **API Key**, Header, name `X-API-Key`, and enter the
   same value you put in Render. If you did not set `API_KEY`, leave
   Authentication as **None**.
5. Click **Update** / **Save** (top right), then **Test** / **Preview**.

## Step 4 — Test it

In the preview pane on the right, type:

> **Arsenal vs Chelsea**

You should see the model's probabilities, xG, top scorelines and
factors — the same numbers as running
`python -m football_agent.cli predict --home Arsenal --away Chelsea`
locally. Try nicknames too: **"Man Utd vs Spurs"**, and a batch:
**"Predict Arsenal-Chelsea, Liverpool-Man City and Tottenham-Newcastle"**.

## Troubleshooting

| Problem | Fix |
|---|---|
| ChatGPT says the API is unreachable / times out | Open `<your-URL>/health` in a browser. If it loads after ~30 s, it was just waking up from sleep — try again in ChatGPT. If it never loads, check the Render deploy logs. |
| "Unknown team" | The API replies with "did you mean" suggestions. Ask ChatGPT to use the suggested name, or type the full name (e.g. "Manchester United"). |
| Predictions look the same every day | That is the sample data: without `FOOTBALL_DATA_API_KEY` the model is fitted on the bundled 2024-25/2025-26 sample. Add the key in Render (Step 1.5) and redeploy for live data. |
| 401 error from the Action | Your `API_KEY` in the Action does not match the one in Render. Re-enter it, or remove `API_KEY` from Render and set Authentication to None. |
| Schema import fails | Paste `openapi.yaml` exactly as-is (it is OpenAPI 3.1). Do not paste the README. After pasting, only change the `{BASE_URL}` value. |

## How it works, in one paragraph

ChatGPT receives your team names, calls `POST /predict` on your
deployed API, and gets back the model's probabilities, xG and
scorelines. It then runs a web search for injuries/line-ups/form and
presents the model numbers first, with the search findings as caveats.
The instructions forbid it from changing the model's numbers or
predicting without calling the API — so what you see is your model,
not ChatGPT guessing.
