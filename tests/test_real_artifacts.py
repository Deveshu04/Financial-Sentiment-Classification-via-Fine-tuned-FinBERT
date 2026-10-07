import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sentiment import Classifier

APP = Path(__file__).resolve().parents[1] / "app"
REAL = APP / "artifacts"

pytestmark = pytest.mark.skipif(not (REAL / "finbert-int8.onnx").exists(), reason="real artifacts not present")


def load(name):
    return json.loads((REAL / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real():
    return Classifier(REAL)


def test_real_parity(real):
    rows = load("parity.json")
    worst = 0.0
    for row in rows:
        probs = real.probabilities(row["text"])
        worst = max(worst, float(np.abs(probs - np.array(row["probabilities"])).max()))
        assert real.labels[int(np.argmax(probs))] == row["label"], row["text"]
    print(f"largest probability difference from notebook 07 on {len(rows)} sentences: {worst:.2e}")
    assert worst < 1e-4


def test_real_market_metrics_match_sessions():
    tested = [row for row in load("sessions.json") if row["in_test"]]
    headline = load("metrics.json")["market"]["models"]["sentiment"]
    correct = sum(1 for row in tested if row["correct"])
    assert len(tested) == headline["n"] and correct == headline["correct"]
    tail = sum(math.comb(len(tested), k) for k in range(correct, len(tested) + 1)) / 2 ** len(tested)
    assert math.isclose(tail, headline["binomial_p"], rel_tol=1e-6)


def test_real_bundle_has_no_dataset_text():
    allowed = {row["text"] for row in load("parity.json")}
    long_strings = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and len(node) > 60 and node not in allowed:
            long_strings.append(node)

    for name in ("metrics.json", "charts.json", "sessions.json", "labels.json", "versions.json"):
        walk(load(name))
    assert long_strings == []


def test_memory_fits_render():
    code = "import psutil, server; server.create_app(); print(psutil.Process().memory_info().rss)"
    out = subprocess.run([sys.executable, "-c", code], cwd=APP, capture_output=True, text=True, check=True)
    rss = int(out.stdout.strip().splitlines()[-1])
    print(f"resident memory after loading the app: {rss / 1e6:.0f} MB")
    assert rss < 400e6


def test_latency(real):
    texts = [row["text"] for row in load("parity.json")]
    single = []
    for text in texts * 3:
        started = time.perf_counter()
        real.predict([text])
        single.append((time.perf_counter() - started) * 1000)
    batch = []
    for _ in range(5):
        started = time.perf_counter()
        real.predict(texts[:16])
        batch.append((time.perf_counter() - started) * 1000)
    print(f"one text: p50 {np.median(single):.0f} ms, p95 {np.percentile(single, 95):.0f} ms; 16 texts: p50 {np.median(batch):.0f} ms")
    assert np.percentile(single, 95) < 2000
