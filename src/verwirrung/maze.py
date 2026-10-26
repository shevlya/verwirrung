import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from verwirrung.paths import ENTRIES_NLP_CSV, DOCS_IMAGES

# 0=коридор (белый), 1=стена (чёрный), 2=след (бледно-розовый), 3=агент (жёлтый)
CMAP = ListedColormap(["#ffffff", "#1a1a1a", "#ffdede", "#ffcc00"])
LINE_COLOR = "#8b0000"  # тёмно-красный для траектории


def generate_maze(width, height):
    """Лабиринт (алгоритм Прима)."""
    maze = np.ones((height, width), dtype=int)
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 0

    walls = []

    def add_walls(fx, fy):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            wx, wy = fx + dx, fy + dy
            if 0 < wx < width - 1 and 0 < wy < height - 1 and maze[wy, wx] == 1:
                walls.append((wx, wy, fx, fy))

    add_walls(start_x, start_y)

    while walls:
        idx = random.randrange(len(walls))
        wx, wy, px, py = walls.pop(idx)
        if maze[wy, wx] != 1:
            continue
        nx, ny = wx + (wx - px), wy + (wy - py)
        if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
            maze[wy, wx] = 0
            maze[ny, nx] = 0
            add_walls(nx, ny)

    maze[1, 0] = 0
    maze[height - 2, width - 1] = 0
    return maze


def _neighbors(maze, pos):
    h, w = maze.shape
    y, x = pos
    return [(y + dy, x + dx) for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if 0 <= y + dy < h and 0 <= x + dx < w and maze[y + dy, x + dx] == 0]


def bfs_path(maze, start, end):
    """Кратчайший путь BFS от start до end."""
    queue = deque([start])
    visited = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur)
                cur = visited[cur]
            return path[::-1]
        for nxt in _neighbors(maze, cur):
            if nxt not in visited:
                visited[nxt] = cur
                queue.append(nxt)
    return [start]


def walk_dfs(maze, max_steps=4000):
    """DFS с возвратами. Останавливается при достижении выхода.

    Если за max_steps не дошёл — добирает остаток через BFS,
    чтобы агент гарантированно вышел из лабиринта.
    """
    h, w = maze.shape
    start = (1, 0)
    end = (h - 2, w - 1)

    path = [start]
    visited = {start}
    stack = [(start, _neighbors(maze, start))]

    while stack and len(path) < max_steps:
        cur, nb = stack[-1]
        if cur == end:
            return path  # дошли!
        unvisited = [n for n in nb if n not in visited]
        if unvisited:
            nxt = random.choice(unvisited)
            visited.add(nxt)
            path.append(nxt)
            stack.append((nxt, _neighbors(maze, nxt)))
        else:
            stack.pop()
            if stack:
                path.append(stack[-1][0])  # шаг назад

    # Fallback: если не дошли за max_steps, добираем путь до выхода
    if path[-1] != end:
        tail = bfs_path(maze, path[-1], end)
        path.extend(tail[1:])
    return path


def animate_maze(maze, path, title, subtitle, out_path, fps=20, target_frames=300):
    """GIF с траекторией."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.02, right=0.78)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    ax.axis("off")

    h, w = maze.shape

    ax.scatter(0, 1, color="#2ca02c", s=220, marker=">", zorder=6, label="Вход")
    ax.scatter(w - 1, h - 2, color="#1f77b4", s=260, marker="*", zorder=6, label="Выход")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    im = ax.imshow(maze, cmap=CMAP, interpolation="nearest", vmin=0, vmax=3)
    line, = ax.plot([], [], color=LINE_COLOR, linewidth=1.8, alpha=0.85, zorder=4)

    # Прореживаем кадры, чтобы GIF не был гигантским
    frame_skip = max(1, len(path) // target_frames)
    sampled = path[::frame_skip]
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])

    def update(frame_idx):
        idx = min(frame_idx * frame_skip, len(path))
        display = maze.copy()
        for py, px in path[:idx]:
            if display[py, px] == 0:
                display[py, px] = 2
        if idx > 0:
            ay, ax_ = path[idx - 1]
            display[ay, ax_] = 3
        im.set_array(display)
        
        xs = [px for py, px in path[:idx]]
        ys = [py for py, px in path[:idx]]
        line.set_data(xs, ys)
        return [im, line]

    anim = FuncAnimation(fig, update, frames=len(sampled) + 1,
                         interval=1000 // fps, blit=True)

    print(f"Сохранение {out_path.name} "
          f"(шагов: {len(path)}, кадров: {len(sampled) + 1}, skip={frame_skip})...")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    df = pd.read_csv(ENTRIES_NLP_CSV)
    early_entry = df.iloc[0]
    peak_entry = df.iloc[40]

    size = 41
    random.seed(42)
    maze = generate_maze(size, size)

    # Ранний Чарли — DFS с возвратами (длинный, извилистый путь)
    path_dfs = walk_dfs(maze)
    animate_maze(
        maze, path_dfs,
        "Ранний Чарли / Регресс (Март)",
        (f"Ошибки: {early_entry['unknown_ratio']*100:.1f}% | "
         f"DFS с возвратами: {len(path_dfs)} шагов"),
        DOCS_IMAGES / "maze_early.gif",
    )

    # Гений — BFS (короткий, прямой путь)
    h, w = maze.shape
    path_bfs = bfs_path(maze, (1, 0), (h - 2, w - 1))
    animate_maze(
        maze, path_bfs,
        "Гений (Май)",
        (f"Ошибки: {peak_entry['unknown_ratio']*100:.1f}% | "
         f"Кратчайший путь: {len(path_bfs)} шагов"),
        DOCS_IMAGES / "maze_smart.gif",
    )

    print("\nГотово.")
    print(f"  dumb: {len(path_dfs)} шагов,  smart: {len(path_bfs)} шагов")
    print(f"  Отношение длин путей: ×{len(path_dfs) / max(len(path_bfs), 1):.1f}")


if __name__ == "__main__":
    main()
