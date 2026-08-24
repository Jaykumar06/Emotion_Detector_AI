"""
emotion_detector.py

Loads a HuggingFace emotion classification model and exposes a single
function, detect_emotion(text), that returns the top emotion label
and its confidence score.

Model used: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from functools import lru_cache
from transformers import pipeline


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def _get_classifier():
    """
    Loads the model only once (cached) since loading it on every
    call would be slow. lru_cache handles this automatically.
    """
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,  # return scores for all labels, not just the top one
    )


def detect_emotion(text: str) -> dict:
    """
    Runs the emotion classifier on a piece of text.

    Returns a dict like:
        {
            "label": "joy",
            "score": 0.87,
            "all_scores": {"joy": 0.87, "neutral": 0.06, ...}
        }
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 1.0, "all_scores": {}}

    classifier = _get_classifier()
    results = classifier(text)[0]  # list of {"label": ..., "score": ...}

    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]

    all_scores = {r["label"]: round(r["score"], 4) for r in results}

    return {
        "label": top["label"],
        "score": round(top["score"], 4),
        "all_scores": all_scores,
    }


if __name__ == "__main__":
    # Quick manual test: python emotion_detector.py
    samples = [
        "I just got promoted, I can't believe it!",
        "I'm so tired of everything going wrong today.",
        "Why didn't you tell me about this earlier?!",
        "I'm a bit nervous about tomorrow's exam.",
    ]
    for s in samples:
        print(s, "->", detect_emotion(s))