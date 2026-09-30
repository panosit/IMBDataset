"""Data loading, validation, and deterministic preprocessing."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import pandas as pd

HTML_TAG_RE = re.compile(r"<.*?>")
REQUIRED_COLUMNS = {"review", "sentiment"}
VALID_SENTIMENTS = {"positive", "negative"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: str | Path) -> pd.DataFrame:
    """Read, validate, and deduplicate the source dataset."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[["review", "sentiment"]].isna().any().any():
        raise ValueError("Dataset contains missing review or sentiment values.")
    if not df["review"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All reviews must be strings.")
    observed = set(df["sentiment"].unique())
    if not observed <= VALID_SENTIMENTS:
        raise ValueError(f"Unexpected sentiment labels: {sorted(observed - VALID_SENTIMENTS)}")
    return df.drop_duplicates().reset_index(drop=True)


def clean_text(text: str) -> str:
    """Apply the preregistered sparse-model normalization."""
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Create immutable analysis columns without changing source labels."""
    result = df.copy()
    result["clean_review"] = result["review"].map(clean_text)
    result["label"] = (result["sentiment"] == "positive").astype(int)
    return result
