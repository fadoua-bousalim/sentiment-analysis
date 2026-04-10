# Reddit Sentiment API

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/fadoua-bousalim/sentiment-backend/actions/workflows/ci.yml/badge.svg)

A lightweight REST API that searches Reddit for any keyword and returns **VADER sentiment analysis** across recent posts — with a built-in browser UI, no separate frontend needed.

> *How does Reddit feel about Python? About GPT-4? About your favourite band?*

## Features

- **Keyword search** across all of Reddit or a specific subreddit
- **VADER NLP** — fast, lexicon-based sentiment scoring (positive / neutral / negative)
- **Built-in UI** — visit `/` for a zero-dependency browser interface
- **Interactive API docs** at `/docs` (Swagger) and `/redoc` (ReDoc)
- One-command deploy to **Render** or **Koyeb** (both have free tiers)

## Quick start

### 1. Get Reddit API credentials

Go to **https://www.reddit.com/prefs/apps** → *create app* → choose **script** type.  
Copy the **client ID** (short string under the app name) and the **secret**.

### 2. Configure environment

```bash
cp .env.example .env
# fill in REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
```

### 3. Install and run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://localhost:8000** for the UI or **http://localhost:8000/docs** for the interactive API explorer.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser UI |
| `GET` | `/analyze` | Sentiment analysis (JSON) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

### `GET /analyze`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Keyword or phrase to search |
| `limit` | integer | `25` | Posts to analyse (1–100) |
| `subreddit` | string | `all` | Restrict to a subreddit, e.g. `python` |

**Example request**

```
GET /analyze?query=rust+programming&limit=10&subreddit=programming
```

**Example response**

```json
{
  "query": "rust programming",
  "total": 10,
  "positive": 6,
  "neutral": 3,
  "negative": 1,
  "avg_compound": 0.3124,
  "posts": [
    {
      "title": "Rust is the most loved language for the 8th year running",
      "url": "https://reddit.com/r/programming/comments/...",
      "subreddit": "programming",
      "score": 4821,
      "num_comments": 203,
      "sentiment": "positive",
      "compound": 0.7351
    }
  ]
}
```

## How it works

```
Browser / API client
        │
        ▼
   FastAPI app
        ├── praw ──► Reddit Search API ──► raw posts
        └── vaderSentiment ──► compound score [-1, 1] per post
                                    │
                                    └── classify: positive / neutral / negative
```

[VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based model tuned for social-media text — it handles slang, punctuation emphasis, and emoji without any training data.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDDIT_CLIENT_ID` | Yes | — | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Yes | — | Reddit app client secret |
| `REDDIT_USER_AGENT` | No | `sentiment-app/0.1` | Identifies your app to Reddit's API |

## Deploy

### Render (free)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo.
3. Add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` as environment secrets in the Render dashboard.
4. Deploy — you'll get a public URL like `https://reddit-sentiment.onrender.com`.

> **Note:** Render's free tier sleeps after 15 min of inactivity; the first request after sleep takes ~30 s to wake.

### Koyeb

A `koyeb.yaml` is included for one-command deployment. Koyeb injects environment variables directly — no `.env` file is used.

In the Koyeb dashboard, create two **Secrets** with these exact names before deploying:

| Secret name | Value |
|-------------|-------|
| `reddit-client-id` | your Reddit client ID |
| `reddit-client-secret` | your Reddit client secret |

Then push the repo and deploy — `koyeb.yaml` wires those secrets to the right env vars automatically.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v    # run tests
ruff check .        # lint
```

## License

MIT — see [LICENSE](LICENSE).