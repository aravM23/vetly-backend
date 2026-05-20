# Vetly Backend

FastAPI service that powers Club Stanley creator sourcing.

## Endpoints
- `POST /api/users/{id}/discover/run` — kick off LLM-driven sourcing pipeline
- `GET  /api/users/{id}/discover/candidates` — list sourced creators (filterable by status / shortlist)
- `POST /api/users/{id}/discover/candidates/{id}/shortlist` — pick for Club Stanley cohort
- `DELETE /api/users/{id}/discover/candidates/{id}/shortlist` — remove
- `POST /api/users/{id}/discover/candidates/{id}/approve|reject`
- `GET  /health`

## Deploy to Render (free, always on with uptime pings)
This repo includes `render.yaml`. In Render: **New → Blueprint → connect this repo**.
Everything (env vars, build command, start command) is pre-wired.

## Local
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Env vars (already in render.yaml)
- `OPENROUTER_API_KEY` — drives LLM sourcing + scoring
- `LLM_MODEL` — default `openai/gpt-4o-mini`
