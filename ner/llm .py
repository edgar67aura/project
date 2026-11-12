from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

INSTRUCTIONS = """Extract named entities from each document below.
Allowed labels: {labels}.
Copy each entity text exactly as it appears in the document (same characters, same spacing).
Answer ONLY with one JSON object per document, one per line, no explanations, no code block, in this format:
{{"id": "<document id>", "entities": [{{"text": "<entity text>", "label": "<LABEL>"}}]}}
List each different entity once, even if it appears several times.
"""


def build_prompt(samples: list[dict[str, Any]], labels: list[str]) -> str:
    """Build one prompt to paste into an LLM chat for a batch of documents."""
    parts = [INSTRUCTIONS.format(labels=", ".join(labels))]
    for sample in samples:
        parts.append(f"### Document id: {sample['id']}\n{sample['text']}\n")
    return "\n".join(parts)


def answers_to_predictions(
    answers_path: str | Path, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Turn the LLM answers (id, text, label) into predictions with start/end spans.

    The LLM gives no character positions, so every occurrence of each answered
    entity text is located in the document. Stops at the first document without
    an answer, so the result is a prefix of the dataset.
    """
    answers = {}
    raw = Path(answers_path).read_text(encoding="utf-8")
    raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```"))
    decoder = json.JSONDecoder()
    position = 0
    while True:
        # skip spaces/newlines; objects may be on one line or stuck together
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position >= len(raw):
            break
        record, position = decoder.raw_decode(raw, position)
        answers[record["id"]] = record.get("entities", [])

    predictions = []
    for sample in samples:
        if sample["id"] not in answers:
            break
        text = sample["text"]
        found = set()
        for entity in answers[sample["id"]]:
            if not entity.get("text"):
                continue
            for match in re.finditer(re.escape(entity["text"]), text):
                found.add((match.start(), match.end(), entity["text"], entity["label"]))
        predictions.append(
            {
                "id": sample["id"],
                "text": text,
                "entities": [
                    {"start": s, "end": e, "text": t, "label": l}
                    for s, e, t, l in sorted(found)
                ],
                "language": sample.get("language"),
            }
        )
    return predictions
