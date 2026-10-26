import matplotlib.pyplot as plt
import pandas as pd

from verwirrung.paths import ENTRIES_SENTIMENT_CSV, DOCS_IMAGES

df = pd.read_csv(ENTRIES_SENTIMENT_CSV)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["entry_id"], df["pos"], color="#2ca02c", label="Позитив")
ax.plot(df["entry_id"], df["neu"], color="#7f7f7f", label="Нейтрально")
ax.plot(df["entry_id"], df["neg"], color="#d62728", label="Негатив")

ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.set_xlabel("Номер записи")
ax.set_ylabel("Вероятность")
ax.set_title("Эмоциональная дуга дневника Чарли")
ax.legend()
ax.grid(True, alpha=0.3)

out = DOCS_IMAGES / "sentiment_timeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Сохранено: {out}")
