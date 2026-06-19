"""Text preprocessing for sentiment classification."""

import re


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
