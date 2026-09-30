#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_sentiment.data import file_sha256, load_data
path = Path("data/raw/imdb_dataset.csv")
df = load_data(path)
print(json.dumps({"rows_after_deduplication": len(df), "class_counts": df.sentiment.value_counts().to_dict(), "sha256": file_sha256(path)}, indent=2))
