from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app, classify

client = TestClient(app)


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
    data = response.json()
    assert data["status"] == "ok"
    assert "reddit_configured" in data


# ---------- / ----------


def test_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Reddit Sentiment" in response.text


# ---------- /analyze ----------


def test_analyze_missing_credentials():
    with patch("main.REDDIT_CLIENT_ID", None):
        response = client.get("/analyze?query=python")
    assert response.status_code == 503


def test_analyze_missing_query_param():
    response = client.get("/analyze")
    assert response.status_code == 422  # FastAPI validation error


def test_analyze_limit_too_low():
    response = client.get("/analyze?query=python&limit=0")
    assert response.status_code == 422


def test_analyze_limit_too_high():
    response = client.get("/analyze?query=python&limit=101")
    assert response.status_code == 422


def test_analyze_returns_results():
    mock_post = MagicMock()
    mock_post.title = "Python is amazing"
    mock_post.selftext = "I love Python for data science."
    mock_post.permalink = "/r/python/comments/abc123/python_is_amazing/"
    mock_post.subreddit = MagicMock(__str__=lambda _: "python")
    mock_post.score = 100
    mock_post.num_comments = 42

    with (
        patch("main.REDDIT_CLIENT_ID", "fake_id"),
        patch("main.REDDIT_CLIENT_SECRET", "fake_secret"),
        patch("main.reddit") as mock_reddit,
    ):
        mock_reddit.subreddit.return_value.search.return_value = [mock_post]
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
    with (
        patch("main.REDDIT_CLIENT_ID", "fake_id"),
        patch("main.REDDIT_CLIENT_SECRET", "fake_secret"),
        patch("main.reddit") as mock_reddit,
    ):
        mock_reddit.subreddit.return_value.search.return_value = []
        response = client.get("/analyze?query=xyzzy_nonexistent")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["avg_compound"] == 0.0
    assert data["posts"] == []