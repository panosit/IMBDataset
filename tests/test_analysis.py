import pandas as pd
import pytest

from imdb_sentiment.artifacts import BaselineArtifactError
from run_analysis import load_saved_split


def test_load_saved_split_uses_persisted_row_ids(tmp_path):
    df = pd.DataFrame({"review": ["a", "b", "c", "d"]}, index=[0, 1, 2, 3])
    pd.DataFrame({"source_index": [0, 2]}).to_csv(tmp_path / "train_split.csv", index=False)
    pd.DataFrame({"source_index": [1, 3]}).to_csv(tmp_path / "test_split.csv", index=False)

    train, test = load_saved_split(df, tmp_path)

    assert list(train.index) == [0, 2]
    assert list(test.index) == [1, 3]


def test_load_saved_split_rejects_overlapping_ids(tmp_path):
    df = pd.DataFrame({"review": ["a", "b"]})
    pd.DataFrame({"source_index": [0]}).to_csv(tmp_path / "train_split.csv", index=False)
    pd.DataFrame({"source_index": [0]}).to_csv(tmp_path / "test_split.csv", index=False)

    with pytest.raises(BaselineArtifactError, match="not disjoint"):
        load_saved_split(df, tmp_path)
