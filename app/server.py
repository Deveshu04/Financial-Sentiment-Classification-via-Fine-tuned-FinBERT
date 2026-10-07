import json
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import HTTPException

from artifacts import ensure_artifacts, hub_download
from errors import ValidationError
from market import Market
from sentiment import MAX_CHARS, MAX_TEXTS, MODEL_NAME, Classifier, parse_texts

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "Quarterly revenue rose 12 percent, beating analyst expectations.",
    "The company cut its full-year profit forecast after weak holiday sales.",
    "The board will publish the interim report on 14 August.",
    "The lender set aside more money for bad loans as defaults climbed.",
]
SENTIMENT_MODELS = ("finbert", "baseline", "prosus")
MARKET_ORDER = ("sentiment", "sentiment_classifier", "sentiment_returns", "returns_only", "previous_direction", "always_up")


class StrictJSON(DefaultJSONProvider):
    sort_keys = False

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


class App(Flask):
    json_provider_class = StrictJSON


def fmt(value, kind="decimal"):
    if value is None:
        return "n/a"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "points":
        return f"{value * 100:.1f} points"
    if kind == "pvalue":
        return "< 0.0001" if value < 1e-4 else f"{value:.4f}"
    return f"{value:.3f}"


def thousands(value):
    return f"{value:,}"


def create_app(artifact_dir=None):
    artifact_dir = Path(artifact_dir or os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
    ensure_artifacts(artifact_dir, os.environ.get("ARTIFACT_REPO"), os.environ.get("HF_TOKEN"), hub_download)
    classifier = Classifier(artifact_dir)
    market = Market(artifact_dir)
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    charts = json.loads((artifact_dir / "charts.json").read_text(encoding="utf-8"))
    sentiment, market_metrics, names = metrics["sentiment"], metrics["market"], charts["model_names"]
    benchmarks = {kind: [b for b in metrics["benchmarks"] if b["kind"] == kind] for kind in ("sentiment", "market")}
    hero = {"text": EXAMPLES[0], **classifier.predict([EXAMPLES[0]])[0]}
    app = App(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    app.add_template_filter(fmt, "fmt")
    app.add_template_filter(thousands, "thousands")

    @app.get("/")
    def console():
        limits = {"texts": MAX_TEXTS, "chars": MAX_CHARS}
        return render_template("console.html", active="console", hero=hero, examples=EXAMPLES, limits=limits, base_url=request.url_root.rstrip("/"), int8=sentiment["int8"], data={"limits": limits})

    @app.get("/model")
    def model_page():
        target = next(b["target"] for b in benchmarks["sentiment"] if b["format"] == "percent")
        data = {"slices": sentiment["slices"], "full": {key: sentiment["models"][key]["accuracy"] for key in SENTIMENT_MODELS}, "names": names, "test_n": sentiment["counts"]["test"], "target": target}
        return render_template("model.html", active="model", s=sentiment, benchmarks=benchmarks["sentiment"], names=names, data=data)

    @app.get("/market")
    def market_page():
        target = next(b["target"] for b in benchmarks["market"] if b["format"] == "percent")
        data = {"models": market_metrics["models"], "rolling": charts["rolling"], "tone": charts["tone"], "names": names, "benchmark": target}
        return render_template("market.html", active="market", m=market_metrics, benchmarks=benchmarks["market"], names=names, order=MARKET_ORDER, data=data)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=MODEL_NAME, sessions=len(market.sessions))

    @app.post("/api/sentiment")
    def score():
        texts = parse_texts(request.get_json(silent=True))
        started = time.perf_counter()
        predictions = classifier.predict(texts)
        return jsonify(model=MODEL_NAME, predictions=predictions, latency_ms=round((time.perf_counter() - started) * 1000, 2))

    @app.get("/api/market/sessions")
    def sessions():
        return jsonify(sessions=market.query(request.args.get("start"), request.args.get("end")))

    @app.get("/api/metrics")
    def metrics_api():
        return jsonify(benchmarks=metrics["benchmarks"], sentiment=sentiment, market=market_metrics)

    @app.errorhandler(ValidationError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            return jsonify(error=error.description), error.code
        return error

    return app
