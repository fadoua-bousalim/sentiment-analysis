import logging
import os
import time
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

REDDIT_HEADERS = {"User-Agent": "sentiment-app/0.1"}
SCRAPERAPI_KEY = os.environ.get("SCRAPERAPI_KEY")
CACHE_TTL = 300  # seconds

analyzer = SentimentIntensityAnalyzer()
_cache: dict = {}

# ---------- Models ----------
class PostSentiment(BaseModel):
    title: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    sentiment: str  # "positive" | "neutral" | "negative"
    compound: float

class AnalyzeResponse(BaseModel):
    query: str
    total: int
    positive: int
    neutral: int
    negative: int
    avg_compound: float
    posts: List[PostSentiment]

# ---------- App ----------
app = FastAPI(
    title="Reddit Sentiment API",
    version="0.1.0",
    description="Search Reddit for any keyword and get VADER sentiment scores across recent posts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def classify(compound: float) -> str:
    """Map a VADER compound score [-1, 1] to a sentiment label.

    Thresholds follow the VADER paper recommendation:
    positive >= 0.05, negative <= -0.05, neutral in between.
    """
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


async def _fetch_posts(query: str, limit: int, subreddit: Optional[str]) -> list:
    """Fetch posts from Reddit via ScraperAPI, with in-memory TTL cache."""
    cache_key = (query, limit, subreddit)
    cached = _cache.get(cache_key)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        logger.info("cache hit query=%r", query)
        return cached["data"]

    if subreddit:
        reddit_url = f"https://www.reddit.com/r/{subreddit}/search.json"
        reddit_params = {"q": query, "limit": limit, "sort": "relevance", "restrict_sr": "on"}
    else:
        reddit_url = "https://www.reddit.com/search.json"
        reddit_params = {"q": query, "limit": limit, "sort": "relevance"}

    target_url = f"{reddit_url}?{urlencode(reddit_params)}"

    if SCRAPERAPI_KEY:
        fetch_url = "http://api.scraperapi.com"
        params = {"api_key": SCRAPERAPI_KEY, "url": target_url}
    else:
        fetch_url = target_url
        params = {}

    async with httpx.AsyncClient(headers=REDDIT_HEADERS, timeout=30) as client:
        response = await client.get(fetch_url, params=params)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Subreddit r/{subreddit!r} not found.")
        response.raise_for_status()
        data = response.json()["data"]["children"]
        _cache[cache_key] = {"data": data, "ts": time.time()}
        return data


@app.get("/health", summary="Health check")
def health() -> dict:
    """Return service liveness."""
    return {"status": "ok"}


@app.get("/analyze", response_model=AnalyzeResponse, summary="Analyze Reddit sentiment")
async def analyze(
    query: str = Query(..., min_length=1, description="Keyword or phrase to search on Reddit"),
    limit: int = Query(25, ge=1, le=100, description="Number of posts to fetch (1–100)"),
    subreddit: Optional[str] = Query(None, description="Restrict to a specific subreddit, e.g. 'python'"),
) -> AnalyzeResponse:
    """Fetch Reddit posts matching *query* and return per-post and aggregate VADER sentiment."""
    try:
        children = await _fetch_posts(query, limit, subreddit)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.error("Reddit API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Reddit returned {exc.response.status_code}.")
    except httpx.RequestError as exc:
        logger.error("Reddit network error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach Reddit. Try again later.")

    posts: List[PostSentiment] = []
    pos = neu = neg = 0
    compound_sum = 0.0

    for child in children:
        s = child["data"]
        text = f"{s['title']}. {s.get('selftext') or ''}"
        scores = analyzer.polarity_scores(text)
        label = classify(scores["compound"])
        compound_sum += scores["compound"]
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1
        posts.append(
            PostSentiment(
                title=s["title"],
                url=f"https://reddit.com{s['permalink']}",
                subreddit=s["subreddit"],
                score=s["score"],
                num_comments=s["num_comments"],
                sentiment=label,
                compound=round(scores["compound"], 4),
            )
        )

    total = len(posts)
    avg = round(compound_sum / total, 4) if total else 0.0
    logger.info("analyze query=%r total=%d pos=%d neu=%d neg=%d", query, total, pos, neu, neg)

    return AnalyzeResponse(
        query=query,
        total=total,
        positive=pos,
        neutral=neu,
        negative=neg,
        avg_compound=avg,
        posts=posts,
    )

# ---------- Tiny frontend ----------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Reddit Sentiment</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 780px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: .25rem; }
  form { display: flex; gap: .5rem; margin: 1rem 0; }
  input, button { padding: .6rem .8rem; font-size: 1rem; border-radius: 8px; border: 1px solid #888; }
  input { flex: 1; }
  button { cursor: pointer; }
  .summary { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .card { padding: .8rem 1rem; border-radius: 10px; border: 1px solid #8884; flex: 1; min-width: 120px; text-align: center; }
  .pos { background: #22c55e22; } .neu { background: #a3a3a322; } .neg { background: #ef444422; }
  .post { border-bottom: 1px solid #8884; padding: .7rem 0; }
  .meta { font-size: .85rem; opacity: .75; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .75rem; }
  .badge.pos { background: #22c55e44; } .badge.neu { background: #a3a3a344; } .badge.neg { background: #ef444444; }
  a { color: inherit; }
</style>
</head>
<body>
  <h1>Reddit Sentiment</h1>
  <p class="meta">Type a keyword and see how Reddit feels about it.</p>
  <form id="f">
    <input id="q" placeholder="e.g. python, taylor swift, bitcoin" required/>
    <button>Analyze</button>
  </form>
  <div id="out"></div>

<script>
const f = document.getElementById('f');
const out = document.getElementById('out');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  out.innerHTML = 'Loading…';
  try {
    const r = await fetch(`/analyze?query=${encodeURIComponent(q)}&limit=25`);
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    out.innerHTML = `
      <div class="summary">
        <div class="card pos"><b>${d.positive}</b><div class="meta">positive</div></div>
        <div class="card neu"><b>${d.neutral}</b><div class="meta">neutral</div></div>
        <div class="card neg"><b>${d.negative}</b><div class="meta">negative</div></div>
        <div class="card"><b>${d.avg_compound}</b><div class="meta">avg score</div></div>
      </div>
      ${d.posts.map(p => `
        <div class="post">
          <span class="badge ${p.sentiment === 'positive' ? 'pos' : p.sentiment === 'negative' ? 'neg' : 'neu'}">${p.sentiment}</span>
          <a href="${p.url}" target="_blank" rel="noopener">${p.title}</a>
          <div class="meta">r/${p.subreddit} • ${p.score} upvotes • ${p.num_comments} comments • compound ${p.compound}</div>
        </div>`).join('')}
    `;
  } catch (err) {
    out.innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
});
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return INDEX_HTML
