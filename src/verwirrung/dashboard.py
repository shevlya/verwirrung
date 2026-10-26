"""Интерактивный дашборд (Streamlit) с тепловой картой мыслей."""

import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from verwirrung.paths import ENTRIES_NLP_CSV
from verwirrung.maze import generate_maze, walk_dfs, bfs_path

MAZE_SEED = 42
MAZE_SIZE = 35

# Кастомная тепловая карта: прозрачный -> жёлтый -> оранжевый -> тёмно-красный
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat",
    [
        (1, 1, 1, 0.0),     # 0.0: прозрачный
        (1, 1, 0.6, 0.4),   # 0.2: бледно-жёлтый
        (1, 0.7, 0, 0.7),   # 0.5: оранжевый
        (0.9, 0.3, 0, 0.9), # 0.8: красно-оранжевый
        (0.6, 0, 0, 1.0),   # 1.0: тёмно-красный
    ]
)

MAZE_CMAP = ListedColormap(["#ffffff", "#1a1a1a"])

st.set_page_config(page_title="Цветы для Элджернона", layout="wide", page_icon="🌼")


@st.cache_resource
def get_static_maze(size=MAZE_SIZE, seed=MAZE_SEED):
    """Один и тот же лабиринт для всех записей."""
    state = random.getstate()
    random.seed(seed)
    maze = generate_maze(size, size)
    random.setstate(state)
    return maze


@st.cache_data
def load_data():
    df = pd.read_csv(ENTRIES_NLP_CSV)

    def phase(row):
        if row["unknown_ratio"] > 0.1:
            return "Хаос (много ошибок)"
        if row["ttr"] > 0.55:
            return "Пик (гений)"
        return "Спад / Регресс"

    df["phase"] = df.apply(phase, axis=1)
    return df


def get_path_for_entry(maze, entry):
    """Детерминированный путь для конкретной записи."""
    size = maze.shape[0]
    is_lost = entry["unknown_ratio"] > 0.1

    state = random.getstate()
    random.seed(int(entry["entry_id"]))
    try:
        if is_lost:
            path = walk_dfs(maze, max_steps=800)
        else:
            path = bfs_path(maze, (1, 0), (size - 2, size - 1))
    finally:
        random.setstate(state)
    return path, is_lost


@st.cache_data
def compute_heatmap(df, maze, current_id):
    """Тепловая карта посещений ТОЛЬКО для предыдущих записей."""
    size = maze.shape[0]
    heatmap = np.zeros((size, size), dtype=float)

    # Берем строго предыдущие записи!
    subset = df[df["entry_id"] < current_id]
    
    if subset.empty:
        return heatmap  # Пустая карта для первой записи

    for _, entry in subset.iterrows():
        path, _ = get_path_for_entry(maze, entry)
        for y, x in path:
            heatmap[y, x] += 1

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def plot_maze_and_heatmap(entry, maze, heatmap, show_path):
    """Лабиринт + тепловая карта + текущий путь."""
    size = maze.shape[0]
    path, is_lost = get_path_for_entry(maze, entry)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # 1. Базовый лабиринт
    ax.imshow(maze, cmap=MAZE_CMAP, interpolation="nearest",
              vmin=0, vmax=1, zorder=0)

    # 2. Тепловая карта (только если есть история)
    if heatmap.max() > 0:
        im_heat = ax.imshow(heatmap, cmap=HEAT_CMAP, vmin=0, vmax=1,
                            interpolation="nearest", zorder=1)
        plt.colorbar(im_heat, ax=ax, fraction=0.04, pad=0.02,
                     label="Частота блужданий (прошлое)")

    # 3. Текущий путь
    if show_path:
        xs = [px for py, px in path]
        ys = [py for py, px in path]
        ax.plot(xs, ys, color="#0044ff", linewidth=2.5, alpha=0.7, zorder=3)

    # 4. Вход / выход
    ax.scatter(0, 1, color="#2ca02c", s=200, marker=">", zorder=5,
               edgecolors="black", linewidths=1)
    ax.scatter(size - 1, size - 2, color="#1f77b4", s=250, marker="*", zorder=5,
               edgecolors="black", linewidths=1)

    status = "Тупики" if is_lost else "Ясность"
    color_status = "#8b0000" if is_lost else "#1f77b4"
    ax.set_title(f"Текущее состояние: {status}\nШагов в этом пути: {len(path)}",
                 fontsize=13, color=color_status, pad=10)

    plt.tight_layout()
    return fig


def main():
    df = load_data()
    maze = get_static_maze()

    # --- Sidebar ---
    st.sidebar.header("Навигация по дневнику")

    min_id = int(df["entry_id"].min())
    max_id = int(df["entry_id"].max())

    # ДЕФОЛТНОЕ ЗНАЧЕНИЕ 50: сразу показываем пик интеллекта для красивого старта
    selected_id = st.sidebar.slider(
        "Выберите запись:",
        min_value=min_id,
        max_value=max_id,
        value=50,
    )

    entry = df[df["entry_id"] == selected_id].iloc[0]
    heatmap = compute_heatmap(df, maze, selected_id)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Настройки отображения")
    show_path = st.sidebar.checkbox("Показывать синий путь текущей записи", value=True)
    
    # Защита авторских прав: по умолчанию текст скрыт
    show_text = st.sidebar.checkbox(
        "Показывать текст записи (только для локального просмотра)", 
        value=False
    )

    # Sidebar debug info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Информация")
    st.sidebar.markdown(f"**Фаза:** {entry['phase']}")
    st.sidebar.markdown(f"**Записей накоплено:** {selected_id - 1}")
    if heatmap.max() > 0:
        nonzero = int((heatmap > 0.01).sum())
        st.sidebar.markdown(f"**Клеток с историей:** {nonzero}")

    # --- Main ---
    st.title("🌼 Лабиринт сознания Чарли Гордона")
    
    # Приветственный текст для первого запуска
    if selected_id == 50:
        st.info("**Добро пожаловать!** Это середина книги — пик интеллекта Чарли. "
                "Подвиньте слайдер влево (к марту) или вправо (к ноябрю), "
                "чтобы увидеть, как меняется его сознание.")

    st.markdown(
        f"**Дата:** {entry['date_raw']} • "
        f"**Отчёт:** {entry['report_num']} • "
        f"**Ошибки:** {entry['unknown_ratio']*100:.1f}%"
    )

    # Мини-таймлайн
    st.markdown("---")
    fig_t, ax_t = plt.subplots(figsize=(12, 2))
    df_sorted = df.sort_values("entry_id")
    rolled = df_sorted["unknown_ratio"].rolling(window=7, min_periods=1).mean()

    ax_t.plot(df_sorted["entry_id"], rolled, color="#ff7f0e", linewidth=2)
    ax_t.scatter(df_sorted["entry_id"], df_sorted["unknown_ratio"],
                 s=10, alpha=0.3, color="#ff7f0e")
    ax_t.axvline(selected_id, color="red", linestyle="--", linewidth=2)
    ax_t.set_ylim(0, 0.3)
    ax_t.set_xlim(min_id, max_id)
    ax_t.set_ylabel("Доля ошибок")
    ax_t.set_xlabel("Номер записи")
    ax_t.grid(True, alpha=0.3, linestyle=":")
    st.pyplot(fig_t)
    plt.close(fig_t)

    # Лабиринт + метрики
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Тепловая карта мыслей")
        st.caption(
            "Тёмно-красные зоны — где Чарли зацикливался чаще всего "
            "в предыдущих записях (до этой)."
        )

        if heatmap.max() == 0:
            st.info("Это первая запись, истории прошлых блужданий ещё нет. "
                    "Сдвиньте слайдер вправо.")

        fig_maze = plot_maze_and_heatmap(entry, maze, heatmap, show_path)
        st.pyplot(fig_maze)
        plt.close(fig_maze)

    with col2:
        st.subheader("Метрики")

        m1, m2 = st.columns(2)
        m1.metric("Ошибки", f"{entry['unknown_ratio']*100:.1f}%")
        m2.metric("TTR", f"{entry['ttr']:.2f}")

        m3, m4 = st.columns(2)
        m3.metric("Слов", f"{int(entry['word_count'])}")
        m4.metric("Слов/предл.", f"{entry['avg_sentence_len']:.1f}")

        st.info(f"**Фаза:** {entry['phase']}")

        st.markdown("---")
        st.subheader("Текст записи")
        
        if show_text:
            with st.container(height=450):
                st.write(entry["text"].replace("\n", "\n\n"))
        else:
            st.warning("🔒 Текст скрыт для соблюдения авторских прав. "
                       "Установите галочку в меню слева, если вы читаете это локально и купили книгу.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Проект по книге Дэниела Киза")


if __name__ == "__main__":
    main()
