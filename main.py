"""Compatibility entry point for the formal experiment framework."""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
from imdb_sentiment.data import HTML_TAG_RE, clean_text, load_data, preprocess
from imdb_sentiment.experiment import run_experiment
DATA_PATH = BASE_DIR / "data/raw/imdb_dataset.csv"
RANDOM_STATE = 42

def main():
    import json
    config = json.loads((BASE_DIR / "configs/baseline.json").read_text(encoding="utf-8"))
    print(f"Artifacts written to {run_experiment(config)}")
if __name__ == "__main__": main()
