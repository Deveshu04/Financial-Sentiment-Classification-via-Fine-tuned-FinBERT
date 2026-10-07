import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "profit", "rose", "fell", "loss", "the", "company", "sales", "shares", "##s", "record", "weak", "strong", "report", "will", "publish", "."]
LABELS = ["negative", "neutral", "positive"]
DIM = 4
CONTRASTS = ["sentiment_classifier", "sentiment_returns", "returns_only", "previous_direction", "always_up"]
MARKET_KEYS = ["sentiment"] + CONTRASTS
MODEL_NAMES = {
    "finbert": "Fine-tuned FinBERT",
    "baseline": "TF-IDF + Logistic Regression",
    "prosus": "ProsusAI/finbert (trained on PhraseBank)",
    "sentiment": "XGBoost regressor, sentiment features (headline)",
    "sentiment_classifier": "XGBoost classifier, sentiment features",
    "sentiment_returns": "XGBoost regressor, sentiment and returns",
    "returns_only": "XGBoost regressor, lagged returns only",
    "previous_direction": "Previous session direction",
    "always_up": "Always up",
}


def strict_loads(text):
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


def weekdays(count, start=date(2020, 1, 2)):
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def weights():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(VOCAB), DIM)).astype(np.float32)
    types = rng.normal(scale=0.1, size=(2, DIM)).astype(np.float32)
    weight = rng.normal(size=(DIM, 3)).astype(np.float32)
    bias = np.array([0.1, 0.0, -0.1], dtype=np.float32)
    return emb, types, weight, bias


def reference_probabilities(ids):
    emb, types, weight, bias = weights()
    pooled = (emb[np.asarray(ids)] + types[0]).mean(axis=0)
    logits = (pooled @ weight + bias).astype(np.float64)
    z = np.exp(logits - logits.max())
    return z / z.sum()


def build_tokenizer(path):
    tokenizer = Tokenizer(WordPiece({token: i for i, token in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(path))


def build_model(path):
    emb, types, weight, bias = weights()
    nodes = [
        helper.make_node("Gather", ["emb", "input_ids"], ["tok"]),
        helper.make_node("Gather", ["types", "token_type_ids"], ["typ"]),
        helper.make_node("Add", ["tok", "typ"], ["hidden"]),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_f", "axis2"], ["mask3"]),
        helper.make_node("Mul", ["hidden", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["mask3", "axis1"], ["count"], keepdims=0),
        helper.make_node("Div", ["summed", "count"], ["pooled"]),
        helper.make_node("MatMul", ["pooled", "weight"], ["scores"]),
        helper.make_node("Add", ["scores", "bias"], ["logits"]),
    ]
    initializers = [
        numpy_helper.from_array(emb, "emb"),
        numpy_helper.from_array(types, "types"),
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
    ]
    inputs = [helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "sequence"]) for name in ("input_ids", "attention_mask", "token_type_ids")]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])]
    model = helper.make_model(helper.make_graph(nodes, "tiny", inputs, outputs, initializers), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def interval(value, spread=0.02):
    return {"value": value, "low": round(value - spread, 4), "high": round(value + spread, 4)}


def sentiment_metrics():
    def model(acc, f1):
        return {"accuracy": interval(acc), "macro_f1": interval(f1), "per_class": {label: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 30} for label in LABELS}, "confusion": [[25, 4, 1], [3, 50, 7], [2, 6, 22]]}

    slices = []
    for name, n, acc in (("agreement >= 66%", 420, 0.9), ("agreement >= 75%", 345, 0.93), ("agreement = 100%", 226, 0.97)):
        slices.append({"name": name, "n": n, "models": {m: {"accuracy": interval(a), "macro_f1": interval(a - 0.02)} for m, a in (("finbert", acc), ("baseline", acc - 0.1), ("prosus", 0.97))}})
    return {
        "checkpoint": "finbert-tone",
        "chosen_seed": 43,
        "counts": {"train": 3872, "val": 484, "test": 484},
        "models": {"finbert": model(0.88, 0.86), "baseline": model(0.76, 0.71), "prosus": model(0.95, 0.94)},
        "improvement": {"relative_accuracy": interval(0.158), "absolute_accuracy": interval(0.12), "relative_macro_f1": interval(0.211), "absolute_macro_f1": interval(0.15)},
        "mcnemar": {"finbert_only": 70, "baseline_only": 12, "p": 1.2e-10},
        "slices": slices,
        "seeds": [{"seed": s, "val_accuracy": 0.87 + i / 100, "test_accuracy": 0.875 + i / 200, "test_macro_f1": 0.85 + i / 200} for i, s in enumerate((42, 43, 44))],
        "seed_accuracy": {"mean": 0.88, "sd": 0.005},
        "int8": {split: {"accuracy": 0.878, "macro_f1": 0.858, "full_accuracy": 0.88, "full_macro_f1": 0.86, "flipped": 3, "n": 484} for split in ("val", "test")},
    }


def market_metrics():
    models = {}
    for i, key in enumerate(MARKET_KEYS):
        acc = 0.55 - i * 0.01
        constant = key == "always_up"
        models[key] = {"label": MODEL_NAMES[key], "accuracy": acc, "correct": round(acc * 40), "n": 40, "low": acc - 0.05, "high": acc + 0.05, "binomial_p": 0.2, "binomial_p_majority": 0.5, "pt_stat": None if constant else 1.1, "pt_p": None if constant else 0.13}
    return {
        "models": models,
        "always_up_rate": 0.52,
        "counts": {"headlines_scored": 32574, "headlines_used": 32400, "sessions": 80, "eligible": 60, "initial": 20, "test": 40},
        "test_start": "2020-01-30",
        "test_end": "2020-03-25",
        "diagnostics": {"half_year": [{"period": "2020H1", "n": 40, "accuracy": 0.55}], "covid": {"feb_apr_2020": {"n": 35, "accuracy": 0.54}, "rest": {"n": 5, "accuracy": 0.6}}},
        "feature_gain": [{"feature": "net_tone_r5", "gain": 1.5}, {"feature": "count", "gain": 0.7}],
    }


BENCHMARKS = [
    {"name": "Test accuracy", "scope": "full test split", "kind": "sentiment", "direction": ">=", "format": "percent", "target": 0.934, "achieved": 0.88, "low": 0.85, "high": 0.91, "met": False},
    {"name": "Test macro F1", "scope": "full test split", "kind": "sentiment", "direction": ">=", "format": "decimal", "target": 0.91, "achieved": 0.86, "low": 0.83, "high": 0.89, "met": False},
    {"name": "Relative accuracy gain over TF-IDF + LR", "scope": "full test split", "kind": "sentiment", "direction": ">=", "format": "percent", "target": 0.19, "achieved": 0.158, "low": 0.11, "high": 0.2, "met": False},
    {"name": "Test accuracy", "scope": "100% agreement slice", "kind": "sentiment", "direction": ">=", "format": "percent", "target": 0.934, "achieved": 0.97, "low": 0.95, "high": 0.99, "met": True},
    {"name": "Test macro F1", "scope": "100% agreement slice", "kind": "sentiment", "direction": ">=", "format": "decimal", "target": 0.91, "achieved": 0.95, "low": 0.92, "high": 0.98, "met": True},
    {"name": "Directional accuracy", "scope": "walk-forward, 40 sessions", "kind": "market", "direction": ">=", "format": "percent", "target": 0.563, "achieved": 0.55, "low": 0.5, "high": 0.6, "met": False},
    {"name": "One-sided binomial p against 0.5", "scope": "walk-forward, 40 sessions", "kind": "market", "direction": "<", "format": "pvalue", "target": 0.01, "achieved": 0.2, "low": None, "high": None, "met": False},
]


def sessions():
    rng = np.random.default_rng(1)
    rows = []
    for i, day in enumerate(weekdays(60)):
        in_test = i >= 20
        target = round(float(rng.normal(0, 1)), 4)
        count = 0 if i == 30 else int(rng.integers(20, 80))
        pred = round(float(rng.normal(0, 0.3)), 4) if in_test else None
        up = None if pred is None else pred >= 0
        contrasts = {name: (bool(rng.random() < 0.5) if in_test else None) for name in CONTRASTS}
        if in_test:
            contrasts["always_up"] = True
        rows.append({"date": day, "count": count, "net_tone": None if count == 0 else round(float(rng.normal(0, 0.2)), 4), "share_pos": None if count == 0 else 0.3, "share_neg": None if count == 0 else 0.2,
                     "target": target, "in_test": in_test, "pred_return": pred, "pred_up": up, "actual_up": target > 0, "correct": None if up is None else up == (target > 0), "contrasts": contrasts})
    return rows


def charts(rows):
    tested = [r for r in rows if r["in_test"]]
    accuracy = [None if i < 9 else round(sum(r["correct"] for r in tested[i - 9:i + 1]) / 10, 4) for i in range(len(tested))]
    always = [None if i < 9 else round(sum(r["actual_up"] for r in tested[i - 9:i + 1]) / 10, 4) for i in range(len(tested))]
    return {
        "model_names": MODEL_NAMES,
        "rolling": {"window": 10, "dates": [r["date"] for r in tested], "accuracy": accuracy, "always_up": always},
        "tone": {"dates": [r["date"] for r in rows], "net_tone_r5": [r["net_tone"] for r in rows], "target_r5": [r["target"] for r in rows]},
    }


def build_bundle(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    build_model(path / "finbert-int8.onnx")
    build_tokenizer(path / "tokenizer.json")
    rows = sessions()
    files = {
        "labels.json": LABELS,
        "metrics.json": {"sentiment": sentiment_metrics(), "market": market_metrics(), "benchmarks": BENCHMARKS},
        "charts.json": charts(rows),
        "sessions.json": rows,
    }
    for name, value in files.items():
        (path / name).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path
