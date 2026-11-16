#!/bin/sh
# Deploy the football predictions API to Google Cloud Run.
# Run in Google Cloud Shell (console.cloud.google.com -> Cloud Shell),
# from a clone of this repo, after creating a GCP project with billing.
# Usage:
#   PROJECT_ID=my-project REGION=europe-west1 ./deploy-cloudrun.sh
# The script prompts for AGENTROUTER_API_KEY (never echo it, never commit it).
# It deploys the same Dockerfile Render uses, with a 300s request timeout
# (the AI briefing can take 90s), 512Mi / 1 vCPU, max 1 instance so the
# free tier (2M requests, 180k vCPU-seconds/month) is hard to exceed.
set -eu
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-football-predictions-agent}"
if [ -z "$PROJECT_ID" ]; then
  echo "Set PROJECT_ID first: PROJECT_ID=my-project $0" >&2
  exit 1
fi
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
printf 'Paste your AGENTROUTER_API_KEY (input hidden): '
stty -echo
read -r AGENTROUTER_API_KEY
stty echo
echo
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 1 --memory 512Mi \
  --concurrency 20 \
  --timeout 300 \
  --max-instances 1 \
  --set-env-vars "AGENTROUTER_API_KEY=$AGENTROUTER_API_KEY,XINGCHI_API_KEY=" \
  --format='value(status.url)'
echo "Test it with:"
echo "  curl <url>/health"
echo "  curl -X POST <url>/ai-brief -H 'Content-Type: application/json' -d '{\"home_team\":\"Arsenal\",\"away_team\":\"Chelsea\",\"league\":\"PL\"}'"
