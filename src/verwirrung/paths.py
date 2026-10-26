"""Централизованные пути к данным и артефактам."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS_IMAGES = PROJECT_ROOT / "docs" / "images"

RAW_FB2 = DATA_RAW / "Kiz_Flowers.fb2"
ENTRIES_CSV = DATA_PROCESSED / "entries.csv"
ENTRIES_NLP_CSV = DATA_PROCESSED / "entries_nlp.csv"
NON_DIARY_CSV = DATA_PROCESSED / "non_diary.csv"
ENTRIES_SENTIMENT_CSV = DATA_PROCESSED / "entries_sentiment.csv"

for p in (DATA_PROCESSED, DOCS_IMAGES):
    p.mkdir(parents=True, exist_ok=True)
