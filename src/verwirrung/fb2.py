"""Финальный парсер FB2 'Цветы для Элджернона'.

Логика:
  top-level <section>  = отчёт
  <subtitle> с датой   = начало новой дневниковой под-записи
  текст до первого subtitle (если есть) = отдельная под-запись с датой из title

Сохраняет data/processed/entries.csv.
"""

import re
from pathlib import Path

import pandas as pd
from lxml import etree
from razdel import sentenize, tokenize
from verwirrung.paths import RAW_FB2, ENTRIES_CSV, NON_DIARY_CSV

FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
NS = {"fb": FB2_NS}

MONTH_ROOTS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "май": 5,
    "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
    "ноябр": 11, "декабр": 12,
}
MONTH_ALT = "|".join(MONTH_ROOTS.keys())
DATE_DM_RE = re.compile(rf"(\d{{1,2}})\s+({MONTH_ALT})[а-я]*", re.IGNORECASE)
DATE_MD_RE = re.compile(rf"({MONTH_ALT})[а-я]*\s+(\d{{1,2}})", re.IGNORECASE)
REPORT_NUM_RE = re.compile(r"(\d+)")


def get_text(el):
    return " ".join(el.itertext()).strip()


def parse_date(text):
    if not text:
        return None, None, None
    m = DATE_DM_RE.search(text)
    if m:
        return int(m.group(1)), MONTH_ROOTS[m.group(2).lower()], m.group(0).strip()
    m = DATE_MD_RE.search(text)
    if m:
        return int(m.group(2)), MONTH_ROOTS[m.group(1).lower()], m.group(0).strip()
    return None, None, None


def parse_report_num(title):
    m = REPORT_NUM_RE.search(title or "")
    return int(m.group(1)) if m else None

def parse_date_from_title(title, report_num):
    if not title:
        return None, None, None
    title_clean = title
    if report_num is not None:
        title_clean = re.sub(rf"\b{report_num}\b", "", title, count=1)
    return parse_date(title_clean)


def extract_subentries(section):
    title_el = section.find("fb:title", NS)
    title = get_text(title_el) if title_el is not None else ""
    report_num = parse_report_num(title)
    day0, month0, raw0 = parse_date_from_title(title, report_num)

    entries = []
    current_date = (day0, month0, raw0)
    current_source = "title"
    current_parts = []

    for child in section:
        tag = etree.QName(child).localname
        if tag == "title":
            continue
        if tag == "subtitle":
            sub_txt = get_text(child)
            day, month, raw = parse_date(sub_txt)
            if day is not None:
                if current_parts:
                    entries.append((*current_date, current_source,
                                    "\n".join(current_parts).strip()))
                    current_parts = []
                current_date = (day, month, raw)
                current_source = "subtitle"
            else:
                current_parts.append(sub_txt)
        elif tag == "p":
            txt = get_text(child)
            if txt:
                current_parts.append(txt)
        elif tag == "empty-line":
            current_parts.append("")

    if current_parts:
        entries.append((*current_date, current_source,
                        "\n".join(current_parts).strip()))
    return title, report_num, entries


def main():
    raw_path = RAW_FB2
    out_diary = ENTRIES_CSV
    out_other = NON_DIARY_CSV

    tree = etree.parse(str(raw_path))
    body = tree.getroot().find("fb:body", NS)
    sections = body.findall("fb:section", NS)

    rows = []
    for sec_idx, sec in enumerate(sections):
        title, report_num, entries = extract_subentries(sec)
        is_diary = bool(re.search(r"атчет|Отчет", title, re.IGNORECASE))

        for sub_idx, (day, month, raw, source, text) in enumerate(entries):
            if not text:
                continue

            # интерполяция для отчёта 4 (в книге дата отсутствует)
            if day is None and report_num == 4 and is_diary:
                day, month = 5.5, 3
                raw = "≈5-6 марта"
                source = "inferred"

            sentences = [s.text for s in sentenize(text)]
            words = [t.text for t in tokenize(text) if t.text.strip()]
            rows.append({
                "entry_id": None,  # заполним ниже
                "section_idx": sec_idx,
                "report_num": report_num,
                "sub_idx": sub_idx,
                "title": title,
                "is_diary": is_diary,
                "date_raw": raw,
                "date_source": source,
                "year": 1965 if day is not None else None,
                "month": month,
                "day": day,
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_len": round(len(words) / len(sentences), 2) if sentences else None,
                "text": text,
            })

    df = pd.DataFrame(rows)
    df_diary = df[df["is_diary"]].reset_index(drop=True)
    df_diary["entry_id"] = df_diary.index + 1

    df_other = df[~df["is_diary"]].copy()
    if len(df_other):
        df_other["entry_id"] = None

    df_diary.to_csv(out_diary, index=False, encoding="utf-8")
    if len(df_other):
        df_other.to_csv(out_other, index=False, encoding="utf-8")

    print("=== Готово ===")
    print(f"Секций:                   {len(sections)}")
    print(f"Дневниковых записей:      {len(df_diary)}  ->  {out_diary}")
    print(f"Не-дневник (Послесловие): {len(df_other)}  ->  {out_other if len(df_other) else '(нет)'}")
    print(f"С датой:                  {df_diary['day'].notna().sum()}")
    print(f"date_source breakdown:    {df_diary['date_source'].value_counts().to_dict()}\n")

    cols = ["entry_id", "report_num", "sub_idx", "date_raw", "date_source",
            "day", "month", "char_count", "sentence_count", "avg_sentence_len"]
    print("Первые 20:")
    print(df_diary[cols].head(20).to_string(index=False))
    print("\nПоследние 10:")
    print(df_diary[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()