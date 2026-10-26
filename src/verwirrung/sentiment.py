"""Анализ эмоциональной тональности дневника Чарли."""

import pandas as pd
from transformers import pipeline

from verwirrung.paths import ENTRIES_NLP_CSV, ENTRIES_SENTIMENT_CSV


def load_sentiment_pipeline():
    """Загружает лёгкую модель для русского сентимента."""
    return pipeline(
        "text-classification",
        model="sergeyzh/rubert-tiny-sentiment",
        truncation=True,
        max_length=512,
    )


def analyze_sentiment(text: str, classifier) -> dict:
    """Возвращает вероятности трёх классов тональности."""
    if not isinstance(text, str) or not text.strip():
        return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}

    # Модель возвращает список с одним словарём (label, score)
    result = classifier(text[:2000])[0]  # обрезаем очень длинные записи
    label = result["label"].lower()
    score = result["score"]

    # Преобразуем в три вероятности (упрощённо: остальные два класса получают остаток)
    probs = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    probs[label] = score
    remaining = (1 - score) / 2
    for k in probs:
        if k != label:
            probs[k] = remaining
    return probs


def main():
    print("Загрузка модели тональности...")
    classifier = load_sentiment_pipeline()

    df = pd.read_csv(ENTRIES_NLP_CSV)
    print(f"Записей для анализа: {len(df)}")

    records = []
    for _, row in df.iterrows():
        probs = analyze_sentiment(row["text"], classifier)
        records.append(
            {
                "entry_id": row["entry_id"],
                "date_raw": row["date_raw"],
                "report_num": row["report_num"],
                "neg": round(probs["negative"], 4),
                "neu": round(probs["neutral"], 4),
                "pos": round(probs["positive"], 4),
                # Полярность: от -1 (негатив) до +1 (позитив)
                "polarity": round(probs["positive"] - probs["negative"], 4),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(ENTRIES_SENTIMENT_CSV, index=False, encoding="utf-8")
    print(f"Сохранено: {ENTRIES_SENTIMENT_CSV}")

    # Краткая статистика по фазам
    merged = df.merge(out, on="entry_id")
    for phase_name, mask in [
        ("Март (начало)", merged["month"] == 3),
        ("Май-Июнь (пик)", merged["month"].isin([5, 6])),
        ("Октябрь-Ноябрь (регресс)", merged["month"].isin([10, 11])),
    ]:
        subset = merged[mask]
        if len(subset):
            avg_pol = subset["polarity"].mean()
            print(f"  {phase_name}: средняя полярность = {avg_pol:+.3f}")


if __name__ == "__main__":
    main()
