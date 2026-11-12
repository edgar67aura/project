from __future__ import annotations

import re
from typing import Any

from gliner import GLiNER

MODEL_NAME = "gliner-community/gliner_small-v2.5"


def load_gliner_model(model_name: str = MODEL_NAME, device: str | None = None) -> GLiNER:
    """Load a GLiNER model (downloads it the first time)."""
    model = GLiNER.from_pretrained(model_name)
    if device:
        model = model.to(device)
    return model


def _chunks(text: str, words_per_chunk: int = 200):
    """Yield (offset, chunk_text). GLiNER truncates long inputs, so we split."""
    words = list(re.finditer(r"\S+", text))
    for i in range(0, len(words), words_per_chunk):
        start = words[i].start()
        end = words[min(i + words_per_chunk, len(words)) - 1].end()
        yield start, text[start:end]


def predict_entities(
    model: GLiNER, text: str, labels: list[str], threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Predict entities for one text, with spans relative to the full text."""
    entities = []
    for offset, chunk in _chunks(text):
        for e in model.predict_entities(chunk, labels, threshold=threshold):
            e["start"] += offset
            e["end"] += offset
            entities.append(e)
    return entities


def predict_dataset(
    model: GLiNER, samples: list[dict[str, Any]], labels: list[str], threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Run predict_entities on every sample. Output format matches evaluation.py."""
    responses = []
    for sample in samples:
        responses.append(
            {
                "id": sample.get("id"),
                "text": sample["text"],
                "entities": predict_entities(model, sample["text"], labels, threshold),
                "language": sample.get("language"),
            }
        )
    return responses
