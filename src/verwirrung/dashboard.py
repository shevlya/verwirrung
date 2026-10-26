"""
Интерактивный дашборд (Streamlit) для исследования дневника Чарли.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from verwirrung.paths import ENTRIES_NLP_CSV
from verwirrung.maze import generate_maze, walk_dfs, bfs_path

# Настройка страницы
st.set_page_config(page_title="Цветы для Элджернона", layout="wide", page_icon="🌼")

CMAP = ListedColormap(["#ffffff", "#1a1a1a", "#ffdede", "#ffcc00"])

@st.cache_data
def load_data():
    """Загружаем данные один раз и кэшируем."""
    df = pd.read_csv(ENTRIES_NLP_CSV)
    return df


def plot_maze_for_entry(entry):
    """Генерирует лабиринт и путь под конкретную запись."""
    # Размер лабиринта зависит от TTR (лексического разнообразия)
    size = int(11 + entry["ttr"] * 20)
    size = max(11, size if size % 2 == 1 else size + 1)
    
    maze = generate_maze(size, size)
    
    # Если ошибок много — используем DFS (тупики). Если мало — BFS (кратчайший путь)
    is_lost = entry["unknown_ratio"] > 0.1
    if is_lost:
        path = walk_dfs(maze, max_steps=500)
    else:
        path = bfs_path(maze, (1, 0), (size - 2, size - 1))
        
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#f9f9f9")
    ax.axis("off")
    
    im = ax.imshow(maze, cmap=CMAP, interpolation="nearest", vmin=0, vmax=3)
    
    # Рисуем след пути
    xs = [px for py, px in path]
    ys = [py for py, px in path]
    ax.plot(xs, ys, color="#8b0000", linewidth=2.5, alpha=0.8, zorder=4)
    
    # Старт и финиш
    ax.scatter(0, 1, color="#2ca02c", s=150, marker=">", zorder=6)
    ax.scatter(size - 1, size - 2, color="#1f77b4", s=200, marker="*", zorder=6)
    
    if is_lost:
        ax.set_title(f"Состояние: Поиск (тупики)\nПуть: {len(path)} шагов", fontsize=12, color="#8b0000")
    else:
        ax.set_title(f"Состояние: Ясность (BFS)\nПуть: {len(path)} шагов", fontsize=12, color="#1f77b4")
        
    plt.tight_layout()
    return fig


def main():
    df = load_data()
    
    # --- Сайдбар ---
    st.sidebar.header("Навигация по дневнику")
    st.sidebar.markdown("Выберите запись, чтобы увидеть состояние сознания Чарли.")
    
    # Слайдер по датам (используем entry_id, но показываем даты)
    min_id = int(df["entry_id"].min())
    max_id = int(df["entry_id"].max())
    
    selected_id = st.sidebar.slider(
        "Выберите запись (entry_id):", 
        min_value=min_id, 
        max_value=max_id, 
        value=min_id
    )
    
    # Получаем данные выбранной записи
    entry = df[df["entry_id"] == selected_id].iloc[0]
    
    # --- Главный экран ---
    st.title("🌼 Лабиринт сознания Чарли Гордона")
    st.markdown(f"**Дата записи:** {entry['date_raw']} (Отчёт {entry['report_num']})")
    
    # Верхняя панель: мини-таймлайн
    st.markdown("---")
    st.subheader("Положение записи на таймлайне")
    
    # Рисуем упрощенный таймлайн с отметкой
    fig_timeline, ax_t = plt.subplots(figsize=(12, 2))
    
    # Строим скользящее среднее для фона
    df_sorted = df.sort_values("entry_id")
    rolled = df_sorted["unknown_ratio"].rolling(window=7, min_periods=1).mean()
    
    ax_t.plot(df_sorted["entry_id"], rolled, color="#ff7f0e", linewidth=2, label="Сглаженные ошибки")
    ax_t.scatter(df_sorted["entry_id"], df_sorted["unknown_ratio"], s=10, alpha=0.3, color="#ff7f0e")
    
    # Отмечаем текущую точку
    ax_t.axvline(selected_id, color="red", linestyle="--", linewidth=2)
    ax_t.annotate(f"Вы здесь: {entry['date_raw']}", xy=(selected_id, 0.25), 
                  xytext=(10, 0), textcoords="offset points", color="red", fontweight="bold")
    
    ax_t.set_ylim(0, 0.3)
    ax_t.set_xlim(min_id, max_id)
    ax_t.set_ylabel("Доля ошибок")
    ax_t.set_xlabel("Номер записи (от марта к ноябрю)")
    ax_t.grid(True, alpha=0.3, linestyle=":")
    ax_t.legend(loc="upper right")
    
    st.pyplot(fig_timeline)
    plt.close(fig_timeline)
    
    # Нижняя панель: Лабиринт и Текст
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Визуализация мыслей")
        fig_maze = plot_maze_for_entry(entry)
        st.pyplot(fig_maze)
        plt.close(fig_maze)
        
    with col2:
        st.subheader("Метрики")
        
        # Красивые карточки с метриками
        m1, m2 = st.columns(2)
        m1.metric("Доля ошибок", f"{entry['unknown_ratio'] * 100:.1f}%", 
                  delta="Плохо" if entry['unknown_ratio'] > 0.1 else "Хорошо", 
                  delta_color="inverse")
        m2.metric("Лексическое разнообразие (TTR)", f"{entry['ttr']:.2f}")
        
        m3, m4 = st.columns(2)
        m3.metric("Слов в записи", f"{int(entry['word_count'])}")
        m4.metric("Слов на предложение", f"{entry['avg_sentence_len']:.1f}")
        
        st.markdown("---")
        st.subheader("Текст записи")
        
        # Текст в прокручиваемом контейнере
        with st.container(height=400):
            st.write(entry["text"].replace("\n", "\n\n"))
            
    st.sidebar.markdown("---")
    st.sidebar.caption("Проект по книге Дэниела Киза «Цветы для Элджернона»")

if __name__ == "__main__":
    main()
