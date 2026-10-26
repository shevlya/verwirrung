"""Интерактивный дашборд (Streamlit) с тепловой картой и эмоциями."""

import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from verwirrung.paths import ENTRIES_NLP_CSV, ENTRIES_SENTIMENT_CSV
from verwirrung.maze import generate_maze, walk_dfs, bfs_path
from verwirrung.simulator import simulate_entry

MAZE_SEED = 42
MAZE_SIZE = 35

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat",
    [
        (1, 1, 1, 0.0),
        (1, 1, 0.6, 0.4),
        (1, 0.7, 0, 0.7),
        (0.9, 0.3, 0, 0.9),
        (0.6, 0, 0, 1.0),
    ]
)
MAZE_CMAP = ListedColormap(["#ffffff", "#1a1a1a"])

st.set_page_config(page_title="Цветы для Элджернона", layout="wide", page_icon="🌼")


@st.cache_resource
def get_static_maze(size=MAZE_SIZE, seed=MAZE_SEED):
    state = random.getstate()
    random.seed(seed)
    maze = generate_maze(size, size)
    random.setstate(state)
    return maze


@st.cache_data
def load_data():
    df = pd.read_csv(ENTRIES_NLP_CSV)
    
    # Пытаемся подгрузить эмоции, если скрипт уже отработал
    try:
        df_sent = pd.read_csv(ENTRIES_SENTIMENT_CSV)
        # Оставляем только нужные колонки, чтобы не было дублей
        df = df.merge(df_sent[['entry_id', 'neg', 'neu', 'pos', 'polarity']], on='entry_id', how='left')
    except FileNotFoundError:
        pass # Если файла еще нет, дашборд просто не покажет эмоции

    def phase(row):
        if row["unknown_ratio"] > 0.1:
            return "Хаос (много ошибок)"
        if row["ttr"] > 0.55:
            return "Пик (гений)"
        return "Спад / Регресс"

    df["phase"] = df.apply(phase, axis=1)
    return df


def get_path_for_entry(maze, entry):
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
    size = maze.shape[0]
    heatmap = np.zeros((size, size), dtype=float)
    subset = df[df["entry_id"] < current_id]
    if subset.empty:
        return heatmap

    for _, entry in subset.iterrows():
        path, _ = get_path_for_entry(maze, entry)
        for y, x in path:
            heatmap[y, x] += 1

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def plot_maze_and_heatmap(entry, maze, heatmap, show_path):
    size = maze.shape[0]
    path, is_lost = get_path_for_entry(maze, entry)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.imshow(maze, cmap=MAZE_CMAP, interpolation="nearest", vmin=0, vmax=1, zorder=0)

    if heatmap.max() > 0:
        im_heat = ax.imshow(heatmap, cmap=HEAT_CMAP, vmin=0, vmax=1,
                            interpolation="nearest", zorder=1)
        plt.colorbar(im_heat, ax=ax, fraction=0.04, pad=0.02,
                     label="Частота блужданий (прошлое)")

    if show_path:
        xs = [px for py, px in path]
        ys = [py for py, px in path]
        ax.plot(xs, ys, color="#0044ff", linewidth=2.5, alpha=0.7, zorder=3)

    ax.scatter(0, 1, color="#2ca02c", s=200, marker=">", zorder=5, edgecolors="black", linewidths=1)
    ax.scatter(size - 1, size - 2, color="#1f77b4", s=250, marker="*", zorder=5, edgecolors="black", linewidths=1)

    status = "Тупики" if is_lost else "Ясность"
    color_status = "#8b0000" if is_lost else "#1f77b4"
    ax.set_title(f"Текущее состояние: {status}\nШагов в этом пути: {len(path)}",
                 fontsize=13, color=color_status, pad=10)
    plt.tight_layout()
    return fig


def plot_emotions(df, selected_id):
    """Рисует сглаженную эмоциональную дугу и отмечает текущую запись."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    
    df_sorted = df.sort_values("entry_id").copy()
    
    # Сглаживаем полярность (окно 5), чтобы убрать шум
    df_sorted["polarity_smooth"] = df_sorted["polarity"].rolling(window=5, min_periods=1).mean()
    
    # Рисуем линию полярности
    ax.plot(df_sorted["entry_id"], df_sorted["polarity_smooth"], color="black", linewidth=2, label="Эмоциональный фон (сглаженный)")
    
    # Заливка цветом: зеленый выше нуля, красный ниже
    ax.fill_between(df_sorted["entry_id"], df_sorted["polarity_smooth"], 0, 
                    where=(df_sorted["polarity_smooth"] >= 0), color="#2ca02c", alpha=0.3, interpolate=True)
    ax.fill_between(df_sorted["entry_id"], df_sorted["polarity_smooth"], 0, 
                    where=(df_sorted["polarity_smooth"] < 0), color="#d62728", alpha=0.3, interpolate=True)
    
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    
    # Отмечаем текущую точку
    curr_pol = df_sorted[df_sorted["entry_id"] == selected_id]["polarity_smooth"].values[0]
    ax.scatter(selected_id, curr_pol, color="red", s=150, zorder=5, label="Текущая запись")
    
    ax.set_xlim(df["entry_id"].min(), df["entry_id"].max())
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Полярность (Негатив <--> Позитив)")
    ax.set_xlabel("Номер записи")
    ax.set_title("Эмоциональная дуга дневника", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    return fig


def main():
    df = load_data()
    maze = get_static_maze()

    # --- Sidebar ---
    st.sidebar.header("Навигация по дневнику")

    min_id = int(df["entry_id"].min())
    max_id = int(df["entry_id"].max())

    selected_id = st.sidebar.slider("Выберите запись:", min_value=min_id, max_value=max_id, value=50)
    entry = df[df["entry_id"] == selected_id].iloc[0]
    
    heatmap = compute_heatmap(df, maze, selected_id)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Режим просмотра")
    
    # ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ
    view_mode = st.sidebar.radio(
        "Что показываем?",
        ["Лабиринт мыслей", "Эмоциональная дуга"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Настройки отображения")
    show_path = st.sidebar.checkbox("Показывать синий путь текущей записи", value=True)
    show_text = st.sidebar.checkbox("Показывать оригинальный текст", value=False)
    use_simulator = st.sidebar.checkbox("🧠 Включить симулятор дневника", value=True)

    # Sidebar debug info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Информация")
    st.sidebar.markdown(f"**Фаза:** {entry['phase']}")

    # --- Main ---
    st.title("🌼 Дневник Чарли Гордона")
    
    if selected_id == 50:
        st.info("**Добро пожаловать!** Это середина книги — пик интеллекта Чарли. "
                "Подвиньте слайдер влево (к марту) или вправо (к ноябрю).")

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
    ax_t.scatter(df_sorted["entry_id"], df_sorted["unknown_ratio"], s=10, alpha=0.3, color="#ff7f0e")
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
        if view_mode == "Лабиринт мыслей":
            st.subheader("Тепловая карта мыслей")
            st.caption("Тёмно-красные зоны — где Чарли зацикливался чаще всего в предыдущих записях.")
            if heatmap.max() == 0:
                st.info("Это первая запись, истории прошлых блужданий ещё нет. Сдвиньте слайдер вправо.")
            fig_maze = plot_maze_and_heatmap(entry, maze, heatmap, show_path)
            st.pyplot(fig_maze)
            plt.close(fig_maze)
            
        else:
            st.subheader("Эмоциональная дуга")
            st.caption("Чёрная линия — сглаженный фон. Зелёный — позитив, красный — негатив. Красная точка — выбранная запись.")
            
            if "polarity" in df.columns and pd.notna(entry["polarity"]):
                fig_emo = plot_emotions(df, selected_id)
                st.pyplot(fig_emo)
                plt.close(fig_emo)
            else:
                st.warning("Данные об эмоциях не найдены. Запустите `uv run verwirrung-sentiment`.")

    with col2:
        st.subheader("Метрики")

        m1, m2 = st.columns(2)
        m1.metric("Ошибки", f"{entry['unknown_ratio']*100:.1f}%")
        m2.metric("TTR", f"{entry['ttr']:.2f}")

        m3, m4 = st.columns(2)
        m3.metric("Слов", f"{int(entry['word_count'])}")
        m4.metric("Слов/предл.", f"{entry['avg_sentence_len']:.1f}")

        st.info(f"**Фаза:** {entry['phase']}")

        # БЛОК ЭМОЦИЙ ДЛЯ ТЕКУЩЕЙ ЗАПИСИ
        if "polarity" in df.columns and pd.notna(entry["polarity"]):
            st.markdown("---")
            st.subheader("Эмоции в этой записи")
            st.progress(float(entry["pos"]), text=f"Позитив: {entry['pos']*100:.0f}%")
            st.progress(float(entry["neu"]), text=f"Нейтрально: {entry['neu']*100:.0f}%")
            st.progress(float(entry["neg"]), text=f"Негатив: {entry['neg']*100:.0f}%")

        st.markdown("---")
        st.subheader("Текст записи")
        
        if use_simulator:
            st.caption("*Симуляция на основе текущих метрик*")
            sim_text = simulate_entry(entry["text"], entry["unknown_ratio"], entry["ttr"], int(entry["entry_id"]))
            with st.container(height=350):
                st.write(sim_text.replace("\n", "\n\n"))
        elif show_text:
            with st.container(height=350):
                st.write(entry["text"].replace("\n", "\n\n"))
        else:
            st.warning("🔒 Текст скрыт.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Проект по книге Дэниела Киза")


if __name__ == "__main__":
    main()
