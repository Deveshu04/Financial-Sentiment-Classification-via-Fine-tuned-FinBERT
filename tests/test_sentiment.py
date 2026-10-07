from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from errors import ValidationError
from helpers import LABELS, VOCAB, reference_probabilities
from sentiment import MAX_CHARS, MAX_TEXTS, MAX_TOKENS, parse_texts


def test_encoding_wraps_and_lowercases(classifier):
    feed = classifier.encode("Profits ROSE")
    assert feed["input_ids"].tolist() == [[VOCAB.index("[CLS]"), VOCAB.index("profit"), VOCAB.index("##s"), VOCAB.index("rose"), VOCAB.index("[SEP]")]]
    assert feed["attention_mask"].tolist() == [[1] * 5]
    assert feed["token_type_ids"].tolist() == [[0] * 5]
    assert feed["input_ids"].dtype == np.int64


def test_long_text_is_truncated(classifier):
    assert classifier.encode("profit " * 400)["input_ids"].shape == (1, MAX_TOKENS)


def test_unicode_and_long_words_are_scored(classifier):
    for text in ("Gewinn stieg um 5 Prozent \U0001F4C8", "\u5229\u6da6\u589e\u957f", "x" * MAX_CHARS, "profit\u2013loss"):
        assert classifier.encode(text)["input_ids"].shape[1] <= MAX_TOKENS
        probs = classifier.probabilities(text)
        assert probs.shape == (3,) and np.isclose(probs.sum(), 1.0)


def test_probabilities_match_reference(classifier):
    text = "the company will publish record sales"
    ids = classifier.encode(text)["input_ids"][0]
    assert np.allclose(classifier.probabilities(text), reference_probabilities(ids), atol=1e-5)


def test_predict_shape_and_label_order(classifier):
    assert classifier.labels == LABELS
    for item in classifier.predict(["profit rose", "loss fell"]):
        assert list(item["probabilities"]) == LABELS
        assert item["label"] == max(item["probabilities"], key=item["probabilities"].get)
        assert abs(sum(item["probabilities"].values()) - 1) < 1e-5


def test_batch_equals_single(classifier):
    texts = ["profit rose", "the company report", "weak sales fell"]
    assert classifier.predict(texts) == [classifier.predict([t])[0] for t in texts]


def test_concurrent_equals_sequential(classifier):
    texts = [" ".join(VOCAB[5 + (i + j) % 15] for j in range(i % 5 + 1)) for i in range(40)]
    sequential = [classifier.predict([t]) for t in texts]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda t: classifier.predict([t]), texts))
    assert concurrent == sequential


@pytest.mark.parametrize("body", [
    None, [], "text", 5, {}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": True}, {"text": None},
    {"texts": []}, {"texts": "profit"}, {"texts": ["ok", 3]}, {"texts": ["ok", ""]}, {"texts": ["ok"] * (MAX_TEXTS + 1)},
    {"text": "x" * (MAX_CHARS + 1)}, {"text": "a", "texts": ["b"]}, {"text": "a", "mode": "fast"}, {"texts": [float("nan")]},
])
def test_parse_texts_rejects(body):
    with pytest.raises(ValidationError):
        parse_texts(body)


def test_parse_texts_accepts():
    assert parse_texts({"text": "profit rose"}) == ["profit rose"]
    assert parse_texts({"texts": ["a", "b"]}) == ["a", "b"]
    assert parse_texts({"texts": ["x" * MAX_CHARS] * MAX_TEXTS}) == ["x" * MAX_CHARS] * MAX_TEXTS
