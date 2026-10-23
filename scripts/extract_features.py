"""Обогащение датасета NLP-метриками с помощью pymorphy3.

Добавляет в entries.csv:
  - unknown_ratio  (доля слов вне словаря = прокси ошибок)
  - ttr            (лексическое разнообразие, Type-Token Ratio)
  - noun_ratio, verb_ratio, adj_ratio (доли частей речи)
"""

import re
from pathlib import Path

import pandas as pd
import pymorphy3
from razdel import tokenize

INPUT = Path("data/processed/entries.csv")
OUTPUT = Path("data/processed/entries_nlp.csv")

# Инициализируем морфологический анализатор один раз (это долго, ~2 сек)
print("Загрузка словарей pymorphy3...")
morph = pymorphy3.MorphAnalyzer()

# Регулярка для отсеивания чистой пунктуации и цифр
WORD_RE = re.compile(r"[а-яА-ЯёЁa-zA-Z]+")


def analyze_text(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "unknown_ratio": None, "ttr": None,
            "noun_ratio": None, "verb_ratio": None, "adj_ratio": None
        }

    # Токенизация
    tokens = [t.text.lower() for t in tokenize(text) if WORD_RE.match(t.text)]

    if not tokens:
        return {
            "unknown_ratio": None, "ttr": None,
            "noun_ratio": None, "verb_ratio": None, "adj_ratio": None
        }

    unknown_count = 0
    lemmas = set()
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJF": 0, "ADJS": 0}  # ADJF - полные прил., ADJS - краткие

    for token in tokens:
        # .parse() возвращает список возможных разборов, берем самый вероятный [0]
        pv = morph.parse(token)[0]
        
        # Проверка на наличие в словаре (ошибки/опечатки)
        if not pv.is_known:
            unknown_count += 1
            
        # Лемма для подсчета TTR (уникальные слова)
        lemmas.add(pv.normal_form)
        
        # Часть речи
        pos = pv.tag.POS
        if pos == "NOUN":
            pos_counts["NOUN"] += 1
        elif pos in ("VERB", "INFN", "PRTF", "GRND"):  # глаголы, инфинитивы, причастия, деепричастия
            pos_counts["VERB"] += 1
        elif pos in ("ADJF", "ADJS"):  # прилагательные
            pos_counts["ADJF"] += 1

    total = len(tokens)

    return {
        "unknown_ratio": round(unknown_count / total, 4),
        "ttr": round(len(lemmas) / total, 4),
        "noun_ratio": round(pos_counts["NOUN"] / total, 4),
        "verb_ratio": round(pos_counts["VERB"] / total, 4),
        "adj_ratio": round(pos_counts["ADJF"] / total, 4),
    }


def main():
    if not INPUT.exists():
        raise SystemExit(f"Файл не найден: {INPUT}. Сначала запустите parse_fb2.py")

    df = pd.read_csv(INPUT)
    print(f"Загружено {len(df)} записей. Анализ...")

    # Применяем функцию к каждой записи
    features = df["text"].apply(analyze_text)
    
    # Превращаем список словарей в DataFrame и склеиваем с исходным
    df_features = pd.DataFrame(features.tolist())
    df = pd.concat([df, df_features], axis=1)

    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Сохранено: {OUTPUT}\n")

    # Выведем статистику по самым важным метрикам
    cols_to_show = ["entry_id", "date_raw", "report_num", 
                    "unknown_ratio", "ttr", "noun_ratio", "adj_ratio"]
    
    print("Первые 5 записей (ранний Чарли):")
    print(df[cols_to_show].head(5).to_string(index=False))
    
    print("\nЗаписи в районе пика (отчёт 11-13):")
    peak_df = df[(df["report_num"] >= 11) & (df["report_num"] <= 13)]
    print(peak_df[cols_to_show].head(5).to_string(index=False))

    print("\nПоследние 5 записей (регресс):")
    print(df[cols_to_show].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
