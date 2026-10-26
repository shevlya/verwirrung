"""Визуализация таймлайна дневника Чарли Гордона

Читает data/processed/entries_nlp.csv
Добавлены метрики: unknown_ratio (ошибки) и ttr (разнообразие)
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from verwirrung.paths import ENTRIES_NLP_CSV, DOCS_IMAGES

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

YEAR = 1965
OUT_DIR = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS_RU = {
    3: "мар", 4: "апр", 5: "май", 6: "июн", 7: "июл",
    8: "авг", 9: "сен", 10: "окт", 11: "ноя",
}

EVENTS = [
    (datetime(YEAR, 3, 9),   "операция (≈)",          8),
    (datetime(YEAR, 10, 20), "смерть Элджернона (≈)",  8),
    (datetime(YEAR, 11, 2),  "последняя запись",      25),
]


def to_datetime(row):
    if pd.isna(row["day"]) or pd.isna(row["month"]):
        return pd.NaT
    day = float(row["day"])
    base = datetime(YEAR, int(row["month"]), int(day))
    frac = day - int(day)
    return base + timedelta(hours=frac * 24)


def load():
    # Читаем обогащённый файл!
    df = pd.read_csv(ENTRIES_NLP_CSV)
    df["dt"] = df.apply(to_datetime, axis=1)
    return df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)


def draw_events(ax):
    for ev_dt, ev_label, dy in EVENTS:
        ax.axvline(ev_dt, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1)
        ax.annotate(
            ev_label,
            xy=(ev_dt, 0.05), xycoords=("data", "axes fraction"),
            xytext=(0, dy), textcoords="offset points",
            rotation=0, va="bottom", ha="right",
            fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )


def plot_metric(ax, df, column, ylabel, title, color, ylim=None):
    ax.scatter(
        df["dt"], df[column],
        s=22, alpha=0.5, color=color, edgecolor="none", zorder=2,
        label="сырые значения",
    )

    inferred = df[df["date_source"] == "inferred"]
    if len(inferred):
        ax.scatter(
            inferred["dt"], inferred[column],
            s=90, marker="X", color="crimson", zorder=4,
            label="дата восстановлена (≈)",
        )

    window = 7
    rolled = df[column].rolling(window=window, min_periods=1).mean()
    ax.plot(
        df["dt"], rolled, color=color, linewidth=2.0, zorder=3,
        label=f"скользящее среднее (окно {window})",
    )

    draw_events(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: MONTHS_RU.get(mdates.num2date(x).month, ""))
    )


def main():
    df = load()
    if not len(df):
        raise SystemExit("Нет данных с датами")

    # Теперь 5 панелей, фигура выше
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

    # 1. Объём
    plot_metric(axes[0], df, "char_count",
                "символов в записи", "Объём дневниковой записи", "#1f77b4")

    peak = df.loc[df["char_count"].idxmax()]
    axes[0].annotate(
        f"пик: {peak['date_raw']}\n({int(peak['char_count'])} симв.)",
        xy=(peak["dt"], peak["char_count"]),
        xytext=(-15, 15), textcoords="offset points",
        fontsize=8, color="#1f77b4", ha="right",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
    )

    # 2. Средняя длина предложения
    plot_metric(axes[1], df, "avg_sentence_len",
                "слов на предложение", "Средняя длина предложения", "#2ca02c",
                ylim=(0, 30))

    # 3. Количество предложений
    plot_metric(axes[2], df, "sentence_count",
                "предложений", "Количество предложений", "#d62728")

    # 4. Ошибки (unknown_ratio)
    plot_metric(axes[3], df, "unknown_ratio",
                "доля неизвестных слов", "Доля орфографических ошибок (прокси)", "#ff7f0e")
    
    # Подсветим последнюю точку (возврат ошибок)
    last_entry = df.iloc[-1]
    axes[3].annotate(
        f"регресс:\n{last_entry['unknown_ratio']*100:.1f}% ошибок",
        xy=(last_entry["dt"], last_entry["unknown_ratio"]),
        xytext=(-20, 15), textcoords="offset points",
        fontsize=8, color="#ff7f0e", ha="right",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8),
    )

    # 5. Лексическое разнообразие (TTR)
    plot_metric(axes[4], df, "ttr",
                "Type-Token Ratio", "Лексическое разнообразие (TTR)", "#9467bd")

    axes[-1].set_xlabel("дата записи (год условно 1965)")
    fig.suptitle(
        "«Цветы для Элджернона»: когнитивная траектория Чарли Гордона",
        fontsize=16, y=0.995,
    )
    fig.tight_layout()

    out = DOCS_IMAGES / "timeline_nlp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {out}")

    # Выведем пиковые значения для контроля
    print(f"\nМаксимум ошибок (unknown_ratio): {df['unknown_ratio'].max():.3f}")
    print(f"Минимум ошибок (unknown_ratio): {df['unknown_ratio'].min():.3f}")
    print(f"Финальная запись: {last_entry['unknown_ratio']:.3f}")


if __name__ == "__main__":
    main()
