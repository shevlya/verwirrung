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

# 0 - стена (темная), 1 - свободно (светлая), 2 - след (красный), 3 - агент (желтый)
CMAP = ListedColormap(["#1a1a1a", "#e0e0e0", "#ff8080", "#ffcc00"])


def generate_maze(width, height):
    """Генерация идеального лабиринта (алгоритм Прима)."""
    maze = np.zeros((height, width), dtype=int)
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 1

    walls = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 < nx < width - 1 and 0 < ny < height - 1:
            walls.append((nx, ny, start_x, start_y))

    while walls:
        idx = random.randrange(len(walls))
        wx, wy, px, py = walls.pop(idx)

        if maze[wy, wx] == 0:
            maze[wy, wx] = 1
            maze[py, px] = 1

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 0:
                    walls.append((nx, ny, wx, wy))
    return maze


def walk_bfs(maze):
    """Поиск кратчайшего пути (для гения)."""
    h, w = maze.shape
    start = (1, 1)
    end = (h - 2, w - 2)
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
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] == 1 and (ny, nx) not in visited:
                queue.append((ny, nx))
                visited[(ny, nx)] = (y, x)
    return [start]


def walk_dfs(maze):
    """Поиск в глубину с возвратами (для раннего Чарли — тупики сознания)."""
    h, w = maze.shape
    start = (1, 1)
    path = [start]
    visited = {start}
    
    # Стек для DFS: (текущая_точка, список_соседей)
    # Чтобы анимация не шла вечно, ограничим количество шагов
    stack = [(start, [(start[0]+dy, start[1]+dx) for dy, dx in [(0,1),(1,0),(0,-1),(-1,0)] 
             if 0 <= start[0]+dy < h and 0 <= start[1]+dx < w and maze[start[0]+dy, start[1]+dx] == 1])]
    
    max_steps = 300  # Ограничение, чтобы GIF не был гигантским
    
    while stack and len(path) < max_steps:
        curr, neighbors = stack[-1]
        
        # Ищем непосещенного соседа
        unvisited = [n for n in neighbors if n not in visited]
        
        if unvisited:
            nxt = random.choice(unvisited)
            visited.add(nxt)
            path.append(nxt)
            # Добавляем в стек нового соседа со списком его соседей
            nx, ny = nxt
            next_neighbors = [(nx+dy, ny+dx) for dy, dx in [(0,1),(1,0),(0,-1),(-1,0)] 
                              if 0 <= nx+dy < h and 0 <= ny+dx < w and maze[nx+dy, ny+dx] == 1]
            stack.append((nxt, next_neighbors))
        else:
            # Тупик! Возвращаемся назад (backtracking)
            stack.pop()
            if stack:
                path.append(stack[-1][0])
                
    return path


def animate_maze(maze, path, title, subtitle, out_path, fps=10):
    """Создание GIF-анимации."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    ax.axis("off")
    
    # Базовый слой лабиринта
    display_base = maze.copy()
    
    im = ax.imshow(display_base, cmap=CMAP, interpolation="nearest", vmin=0, vmax=3)

    def update(frame):
        display = display_base.copy()
        
        # Рисуем след (все шаги до текущего)
        for py, px in path[:frame]:
            if display[py, px] == 1:
                display[py, px] = 2
                
        # Рисуем агента (текущая позиция)
        if frame > 0:
            ay, ax_ = path[frame - 1]
            display[ay, ax_] = 3
            
        im.set_array(display)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(path) + 1, interval=100, blit=True)
    
    print(f"Сохранение {out_path.name} ({len(path)} кадров)...")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    df = pd.read_csv("data/processed/entries_nlp.csv")
    early_entry = df.iloc[0]    # Март
    peak_entry = df.iloc[40]    # Май
    
    # Размер лабиринта (сделаем чуть больше для наглядности)
    size = 21
    
    # Генерируем ОДИН лабиринт для обоих, чтобы сравнение было честным
    random.seed(42)
    maze = generate_maze(size, size)
    
    # 1. Ранний Чарли (DFS)
    path_dfs = walk_dfs(maze)
    animate_maze(
        maze, path_dfs, 
        "Ранний Чарли / Регресс (Март)", 
        f"Ошибки: {early_entry['unknown_ratio']*100:.1f}% | Поиск в глубину (тыкается в тупики)",
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
