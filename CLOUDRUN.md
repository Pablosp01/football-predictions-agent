# Football predictions on Google Cloud Run

Why: the AI briefing's live search works from this computer but not from
Render — AgentRouter's firewall blocks requests from Render's servers.
Cloud Run is the test for whether a different host's servers get through.
The predictions site stays on Render; only the briefing is tried here
first. If `/ai-brief` on Cloud Run returns live search results, the Render
site's page can call the Cloud Run URL for briefings instead.

## What it costs

Cloud Run's free tier is 2 million requests, 180,000 vCPU-seconds and
360,000 GiB-seconds a month. This service is tiny (a few briefing calls
a day), so it should stay inside the free tier — but Google requires a
billing account with a card on file to use Cloud Run at all, and anything
over the free tier is billed, not paused. Set a budget alert in the
Cloud console after creating the project. New Google Cloud accounts also
get $300 of trial credit.

## Setup (Vincent, ~10 minutes)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and
   sign in with your Google account. Create a new project (any name,
   e.g. `football-predictions`).
2. In that project, enable billing: Billing -> Link a billing account.
   A card is required even for free-tier use.
3. Open Cloud Shell (the `>_` icon, top right in the console) and run:
   ```
   git clone https://github.com/Pablosp01/football-predictions-agent.git
   cd football-predictions-agent
   PROJECT_ID=<your project id> ./deploy-cloudrun.sh
   ```
   The script enables Cloud Run + Cloud Build, asks for your
   AgentRouter key (typed hidden, never shown or saved in a file),
   and deploys. It prints the service URL at the end.
4. Send Vesper the Cloud Run URL. Vesper tests `/health` and
   POST `/ai-brief` (Arsenal vs Chelsea) on it and reports whether
   live search works from Cloud Run.

## Deploy settings

- Same `Dockerfile` as Render (Python 3.12, uvicorn on `$PORT`).
- `--timeout 300`: the briefing can take up to 90 seconds.
- `--cpu 1 --memory 512Mi --max-instances 1`: keeps usage inside the
  free tier and limits any surprise bill.
- `--allow-unauthenticated`: the API is public, as on Render.
- Env: `AGENTROUTER_API_KEY` (set by the script), `XINGCHI_API_KEY`
  left empty — Xingchi is only the knowledge-only fallback.
