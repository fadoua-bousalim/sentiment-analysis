import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import praw
import prawcore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Config ----------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sentiment-app/0.1")

# ---------- Clients ----------
# Instantiate only when credentials are present; the /analyze endpoint guards
# against a None client and returns 503 if they are missing.
if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    reddit.read_only = True
else:
    reddit = None

analyzer = SentimentIntensityAnalyzer()


# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning(
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — /analyze will return 503."
        )
    else:
        logger.info("Reddit credentials loaded. App ready.")
    yield


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
    lifespan=lifespan,
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


@app.get("/health", summary="Health check")
def health() -> dict:
    """Return service liveness and whether Reddit credentials are configured."""
    return {
        "status": "ok",
        "reddit_configured": bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET),
    }


@app.get("/analyze", response_model=AnalyzeResponse, summary="Analyze Reddit sentiment")
def analyze(
    query: str = Query(..., min_length=1, description="Keyword or phrase to search on Reddit"),
    limit: int = Query(25, ge=1, le=100, description="Number of posts to fetch (1–100)"),
    subreddit: Optional[str] = Query(None, description="Restrict to a specific subreddit, e.g. 'python'"),
) -> AnalyzeResponse:
    """Fetch Reddit posts matching *query* and return per-post and aggregate VADER sentiment."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Reddit credentials not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.",
        )

    try:
        source = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")
        submissions = list(source.search(query, limit=limit, sort="relevance"))
    except prawcore.exceptions.Redirect:
        raise HTTPException(status_code=404, detail=f"Subreddit r/{subreddit!r} not found.")
    except prawcore.exceptions.ResponseException as exc:
        logger.error("Reddit API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Reddit API error: {exc.response.status_code}")
    except prawcore.exceptions.RequestException as exc:
        logger.error("Reddit network error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach Reddit. Try again later.")

    posts: List[PostSentiment] = []
    pos = neu = neg = 0
    compound_sum = 0.0

    for s in submissions:
        text = f"{s.title}. {s.selftext or ''}"
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
                title=s.title,
                url=f"https://reddit.com{s.permalink}",
                subreddit=str(s.subreddit),
                score=s.score,
                num_comments=s.num_comments,
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
