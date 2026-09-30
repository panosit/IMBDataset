#!/usr/bin/env python3
"""Run the preregistered baseline experiment from a JSON configuration."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imdb_sentiment.experiment import run_experiment
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/baseline.json")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text(encoding="utf-8"))
print(f"Artifacts written to {run_experiment(config)}")
