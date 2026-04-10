# Reddit Sentiment API

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/fadoua-bousalim/sentiment-backend/actions/workflows/ci.yml/badge.svg)
![Deployed on Fly.io](https://img.shields.io/badge/deployed-fly.io-purple?logo=flydotio&logoColor=white)

A lightweight REST API that searches Reddit for any keyword and returns **VADER sentiment analysis** across recent posts — with a built-in browser UI, no separate frontend needed.

> *How does Reddit feel about Python? About GPT-4? About your favourite band?*

## Features

- **No API key required** — uses Reddit's public JSON endpoints
- **Keyword search** across all of Reddit or a specific subreddit
- **VADER NLP** — fast, lexicon-based sentiment scoring (positive / neutral / negative)
- **Built-in UI** — visit `/` for a zero-dependency browser interface
- **Interactive API docs** at `/docs` (Swagger) and `/redoc` (ReDoc)
- Deployed on **[Fly.io](https://sentiment-analysis-flyio.fly.dev/)** (free tier)

## Quick start

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
        ├── httpx ──► Reddit public JSON API ──► raw posts
        └── vaderSentiment ──► compound score [-1, 1] per post
                                    │
                                    └── classify: positive / neutral / negative
```

[VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based model tuned for social-media text — it handles slang, punctuation emphasis, and emoji without any training data.

## Deploy

Deployed on **Fly.io**: https://sentiment-analysis-flyio.fly.dev/

A `Dockerfile` and `fly.toml` are included. Connect the repo in the [Fly.io dashboard](https://fly.io/dashboard) — no secrets needed.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v    # run tests
ruff check .        # lint
```

## License

MIT — see [LICENSE](LICENSE).
