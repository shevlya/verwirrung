"""Смотрим, где в FB2 спрятаны даты для отчётов 4, 6, 7, 8, 9."""

import re
from lxml import etree

FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
NS = {"fb": FB2_NS}

MONTH_RE = re.compile(
    r"(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|"
    r"июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|"
    r"ноябр[ья]|декабр[ья])",
    re.IGNORECASE,
)


def txt(el):
    return " ".join(el.itertext()).strip()


def title_text(sec):
    t = sec.find("fb:title", NS)
    return txt(t) if t is not None else ""


tree = etree.parse("data/raw/Kiz_Flowers.fb2")
root = tree.getroot()
body = root.find("fb:body", NS)
sections = body.findall("fb:section", NS)

for idx in [4, 6, 7, 8, 9]:
    sec = sections[idx]
    print(f"\n{'=' * 70}")
    print(f"SECTION [{idx}]  title={title_text(sec)!r}")

    subs = sec.findall("fb:subtitle", NS)
    print(f"subtitles ({len(subs)}):")
    for s in subs[:6]:
        print("   ", txt(s)[:100])

    dates = sec.findall("fb:date", NS)
    print(f"<date> elements: {len(dates)}")
    for d in dates:
        print("   value=", d.get("value"), "text=", repr(txt(d)))

    ps = sec.findall("fb:p", NS)
    if ps:
        print(f"first p: {txt(ps[0])[:180]!r}")

    full = " ".join(sec.itertext())
    m = MONTH_RE.search(full)
    if m:
        start = max(0, m.start() - 70)
        end = min(len(full), m.end() + 40)
        print(f"month context: ...{full[start:end]!r}...")
    else:
        print("no month mentions found")
