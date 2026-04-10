from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from main import app, classify

client = TestClient(app)

MOCK_CHILDREN = [
    {
        "data": {
            "title": "Python is amazing",
            "selftext": "I love Python for data science.",
            "permalink": "/r/python/comments/abc123/python_is_amazing/",
            "subreddit": "python",
            "score": 100,
            "num_comments": 42,
        }
    }
]


# ---------- classify() ----------


class TestClassify:
    def test_positive_boundary(self):
        assert classify(0.05) == "positive"

    def test_positive_high(self):
        assert classify(0.9) == "positive"

    def test_negative_boundary(self):
        assert classify(-0.05) == "negative"

    def test_negative_high(self):
        assert classify(-0.9) == "negative"

    def test_neutral_zero(self):
        assert classify(0.0) == "neutral"

    def test_neutral_just_below_positive(self):
        assert classify(0.04) == "neutral"

    def test_neutral_just_above_negative(self):
        assert classify(-0.04) == "neutral"


# ---------- /health ----------


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- / ----------


def test_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Reddit Sentiment" in response.text


# ---------- /analyze ----------


def test_analyze_missing_query_param():
    response = client.get("/analyze")
    assert response.status_code == 422


def test_analyze_limit_too_low():
    response = client.get("/analyze?query=python&limit=0")
    assert response.status_code == 422


def test_analyze_limit_too_high():
    response = client.get("/analyze?query=python&limit=101")
    assert response.status_code == 422


def test_analyze_returns_results():
    with patch("main._fetch_posts", return_value=MOCK_CHILDREN):
        response = client.get("/analyze?query=python&limit=1")

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "python"
    assert data["total"] == 1
    assert data["posts"][0]["title"] == "Python is amazing"
    assert data["posts"][0]["sentiment"] in {"positive", "neutral", "negative"}
    assert data["posts"][0]["subreddit"] == "python"
    assert data["posts"][0]["score"] == 100


def test_analyze_empty_results():
    with patch("main._fetch_posts", return_value=[]):
        response = client.get("/analyze?query=xyzzy_nonexistent")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["avg_compound"] == 0.0
    assert data["posts"] == []


def test_analyze_network_error_returns_502():
    with patch("main._fetch_posts", side_effect=httpx.RequestError("timeout")):
        response = client.get("/analyze?query=python")
    assert response.status_code == 502


def test_analyze_invalid_subreddit_returns_404():
    from fastapi import HTTPException

    with patch("main._fetch_posts", side_effect=HTTPException(status_code=404, detail="not found")):
        response = client.get("/analyze?query=python&subreddit=thisdoesnotexist99999")
    assert response.status_code == 404
