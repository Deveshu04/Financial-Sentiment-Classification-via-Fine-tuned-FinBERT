import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from errors import ValidationError

MAX_TOKENS = 128
MAX_TEXTS = 16
MAX_CHARS = 2000
MODEL_NAME = "finbert-int8"


def softmax(logits):
    z = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def parse_texts(body):
    if not isinstance(body, dict) or len(body) != 1 or not ({"text", "texts"} & set(body)):
        raise ValidationError("send a JSON object with exactly one of 'text' (a string) or 'texts' (a list of strings)")
    texts = [body["text"]] if "text" in body else body["texts"]
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS:
        raise ValidationError(f"'texts' must be a list of 1 to {MAX_TEXTS} strings")
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("every text must be a non-empty string")
        if len(text) > MAX_CHARS:
            raise ValidationError(f"each text is limited to {MAX_CHARS} characters")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValidationError("texts must be valid Unicode; one contains an unpaired surrogate") from None
    return texts


class Classifier:
    def __init__(self, artifact_dir, threads=1):
        artifact_dir = Path(artifact_dir)
        self.labels = json.loads((artifact_dir / "labels.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(artifact_dir / "finbert-int8.onnx"), options, providers=["CPUExecutionProvider"])
        self.inputs = [item.name for item in self.session.get_inputs()]

    def encode(self, text):
        ids = np.array([self.tokenizer.encode(text).ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        return {name: feed[name] for name in self.inputs}

    def probabilities(self, text):
        return softmax(self.session.run(["logits"], self.encode(text))[0])[0]

    def predict(self, texts):
        out = []
        for text in texts:
            probs = self.probabilities(text)
            out.append({"label": self.labels[int(np.argmax(probs))], "probabilities": {label: round(float(p), 6) for label, p in zip(self.labels, probs)}})
        return out
