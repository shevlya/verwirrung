"""Анимация прохождения лабиринта сознания Чарли (GIF).

Логика:
  - Ранний Чарли / Регресс (много ошибок): DFS (блуждание с возвратами в тупики).
  - Гений (мало ошибок): BFS (кратчайший путь).
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

OUT_DIR = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Цвета: 0 - путь (белый), 1 - стена (чёрный), 2 - след (красный), 3 - агент (жёлтый)
CMAP = ListedColormap(["#ffffff", "#1a1a1a", "#ff4d4d", "#ffcc00"])


def generate_maze(width, height):
    """Генерация лабиринта (алгоритм Прима)."""
    maze = np.ones((height, width), dtype=int)  # 1 = стена, 0 = путь
    
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 0

    walls = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 < nx < width - 1 and 0 < ny < height - 1:
            walls.append((nx, ny, start_x, start_y))

    while walls:
        idx = random.randrange(len(walls))
        wx, wy, px, py = walls.pop(idx)

        if maze[wy, wx] == 1:
            maze[wy, wx] = 0
            maze[py, px] = 0

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                    walls.append((nx, ny, wx, wy))
                    
    # Прорубаем вход (слева сверху) и выход (справа снизу)
    maze[1, 0] = 0
    maze[height - 2, width - 1] = 0
    
    return maze


def walk_bfs(maze):
    """Поиск кратчайшего пути (BFS)."""
    h, w = maze.shape
    start = (1, 0)
    end = (h - 2, w - 1)
    queue = [start]
    visited = {start: None}

    while queue:
        y, x = queue.pop(0)
        if (y, x) == end:
            path = []
            curr = end
            while curr is not None:
                path.append(curr)
                curr = visited[curr]
            return path[::-1]

        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] == 0 and (ny, nx) not in visited:
                queue.append((ny, nx))
                visited[(ny, nx)] = (y, x)
    return [start]


def walk_dfs(maze):
    """Поиск в глубину с возвратами (DFS)."""
    h, w = maze.shape
    start = (1, 0)
    path = [start]
    visited = {start}
    
    stack = [(start, [(start[0] + dy, start[1] + dx) 
             for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)] 
             if 0 <= start[0] + dy < h and 0 <= start[1] + dx < w and maze[start[0] + dy, start[1] + dx] == 0])]
    
    max_steps = 500  # Ограничим, чтобы GIF не был слишком тяжёлым
    
    while stack and len(path) < max_steps:
        curr, neighbors = stack[-1]
        unvisited = [n for n in neighbors if n not in visited]
        
        if unvisited:
            nxt = random.choice(unvisited)
            visited.add(nxt)
            path.append(nxt)
            nx, ny = nxt
            next_neighbors = [(nx + dy, ny + dx) 
                              for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)] 
                              if 0 <= nx + dy < h and 0 <= ny + dx < w and maze[nx + dy, ny + dx] == 0]
            stack.append((nxt, next_neighbors))
        else:
            stack.pop()
            if stack:
                path.append(stack[-1][0])  # Шаг назад (backtrack)
                
    return path


def animate_maze(maze, path, title, subtitle, out_path, fps=15):
    """Создание GIF-анимации."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    ax.axis("off")
    
    h, w = maze.shape
    
    # Маркеры входа и выхода
    ax.scatter(0, 1, color="#2ca02c", s=200, marker=">", zorder=5, label="Вход")
    ax.scatter(w - 1, h - 2, color="#1f77b4", s=200, marker="*", zorder=5, label="Выход")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    im = ax.imshow(maze, cmap=CMAP, interpolation="nearest", vmin=0, vmax=3)

    def update(frame):
        display = maze.copy()
        # Рисуем след
        for py, px in path[:frame]:
            if display[py, px] == 0:
                display[py, px] = 2
        # Рисуем агента
        if frame > 0:
            ay, ax_ = path[frame - 1]
            display[ay, ax_] = 3
            
        im.set_array(display)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(path) + 1, interval=50, blit=True)
    
    print(f"Сохранение {out_path.name} ({len(path)} кадров)...")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    df = pd.read_csv("data/processed/entries_nlp.csv")
    early_entry = df.iloc[0]    # Март
    peak_entry = df.iloc[40]    # Май
    
    # Увеличили размер для наглядности стенок
    size = 41  
    
    random.seed(42)
    maze = generate_maze(size, size)
    
    # 1. Ранний Чарли (DFS)
    path_dfs = walk_dfs(maze)
    animate_maze(
        maze, path_dfs, 
        "Ранний Чарли / Регресс (Март)", 
        f"Ошибки: {early_entry['unknown_ratio']*100:.1f}% | Поиск в глубину (тупики)",
        OUT_DIR / "maze_early.gif"
    )
    
    # 2. Гений (BFS)
    path_bfs = walk_bfs(maze)
    animate_maze(
        maze, path_bfs, 
        "Гений (Май)", 
        f"Ошибки: {peak_entry['unknown_ratio']*100:.1f}% | Кратчайший путь (BFS)",
        OUT_DIR / "maze_smart.gif"
    )

    print("Готово! Файлы лежат в docs/images/")


if __name__ == "__main__":
    main()
