from concurrent.futures import ThreadPoolExecutor

import pytest

from helpers import strict_loads
from sentiment import MAX_CHARS, MAX_TEXTS


def body(response):
    return strict_loads(response.get_data(as_text=True))


def test_health(client):
    out = client.get("/api/health")
    assert out.status_code == 200
    assert body(out) == {"status": "ok", "model": "finbert-int8", "sessions": 60}


def test_sentiment_single(client):
    out = client.post("/api/sentiment", json={"text": "profit rose"})
    data = body(out)
    assert out.status_code == 200
    assert data["model"] == "finbert-int8" and len(data["predictions"]) == 1 and data["latency_ms"] >= 0
    assert set(data["predictions"][0]["probabilities"]) == {"negative", "neutral", "positive"}


def test_sentiment_batch_matches_singles(client):
    texts = ["profit rose", "loss fell", "the company will publish the report"]
    batch = body(client.post("/api/sentiment", json={"texts": texts}))["predictions"]
    singles = [body(client.post("/api/sentiment", json={"text": t}))["predictions"][0] for t in texts]
    assert batch == singles


@pytest.mark.parametrize("payload", [
    '{"text": NaN}',
    '{"text": Infinity}',
    "not json",
    "",
    "[]",
    '{"texts": ["a", true]}',
    '{"text": "a", "texts": ["b"]}',
    '{"text": "' + "x" * (MAX_CHARS + 1) + '"}',
    '{"texts": [' + ", ".join(['"a"'] * (MAX_TEXTS + 1)) + "]}",
])
def test_sentiment_bad_bodies(client, payload):
    out = client.post("/api/sentiment", data=payload, content_type="application/json")
    assert out.status_code == 400
    assert "error" in body(out)


def test_sentiment_form_body_is_rejected(client):
    out = client.post("/api/sentiment", data={"text": "profit"})
    assert out.status_code == 400 and "error" in body(out)


def test_oversized_body_is_413_json(client):
    out = client.post("/api/sentiment", data='{"text": "' + "x" * 300_000 + '"}', content_type="application/json")
    assert out.status_code == 413 and "error" in body(out)


@pytest.mark.parametrize("path,method,code", [("/api/nothing", "get", 404), ("/api/sentiment", "get", 405), ("/api/health", "post", 405), ("/api/market/sessions", "post", 405), ("/api/metrics", "delete", 405)])
def test_api_errors_are_json(client, path, method, code):
    out = getattr(client, method)(path)
    assert out.status_code == code and out.is_json and "error" in body(out)


def test_market_sessions_endpoint(client, market):
    dates = [r["date"] for r in market.query()]
    assert len(body(client.get("/api/market/sessions"))["sessions"]) == 60
    some = body(client.get(f"/api/market/sessions?start={dates[5]}&end={dates[9]}"))["sessions"]
    assert [r["date"] for r in some] == dates[5:10]


@pytest.mark.parametrize("query", ["start=2020/01/02", "end=nope", "start=2020-02-30", "start=2020-03-01&end=2020-02-01"])
def test_market_bad_dates(client, query):
    out = client.get(f"/api/market/sessions?{query}")
    assert out.status_code == 400 and "error" in body(out)


def test_responses_are_strict_json(client):
    for path in ("/api/health", "/api/market/sessions", "/api/metrics"):
        text = client.get(path).get_data(as_text=True)
        assert "NaN" not in text and "Infinity" not in text
        strict_loads(text)
    assert any(r["net_tone"] is None for r in body(client.get("/api/market/sessions"))["sessions"])


def test_metrics_endpoint(client):
    data = body(client.get("/api/metrics"))
    assert set(data) == {"benchmarks", "sentiment", "market"}
    assert {b["kind"] for b in data["benchmarks"]} == {"sentiment", "market"}


def test_concurrent_requests_match_sequential(app):
    texts = [f"profit rose {'and sales rose ' * (i % 4)}".strip() for i in range(24)]
    sequential = [app.test_client().post("/api/sentiment", json={"text": t}).get_json()["predictions"] for t in texts]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda t: app.test_client().post("/api/sentiment", json={"text": t}).get_json()["predictions"], texts))
    assert concurrent == sequential
