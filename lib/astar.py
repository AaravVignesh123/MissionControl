"""A* pathfinding for the Mission Control rescue grid.

Pure Python standard library only -- this module has to run inside the CPython
runtime bundled with the `jac` binary, so no numpy / no third-party imports.

Grid convention (frozen in CONTRACT.md):
    8x8, origin (0,0) is TOP-LEFT, x = column (-> east), y = row (v south)
    NORTH = y-1, SOUTH = y+1, EAST = x+1, WEST = x-1
    `cells` is a 64-entry row-major list of strings; index = y*8 + x
    Only kind == "hazard" is impassable.

Determinism: the open set is a heap keyed by (f, h, insertion_counter).  Ties on
f are broken toward the lower heuristic (depth-first along optimal paths) and
then by insertion order, which is fixed by NEIGHBOR_ORDER below.  Given the same
inputs this module always returns byte-identical output -- the demo path never
wobbles between runs.
"""

import heapq

W = 8
H = 8

# (dx, dy, name).  Order matters: it is the final tie-breaker inside A*.
# SOUTH-then-EAST first biases the baseline route down through the wall gap at
# (3,4) and along the (4,5) corridor, which is the cell the demo disrupts.
NEIGHBOR_ORDER = (
    (0, 1, "SOUTH"),
    (1, 0, "EAST"),
    (0, -1, "NORTH"),
    (-1, 0, "WEST"),
)

MOVE_DELTA = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


def in_bounds(x, y):
    """True if (x, y) is inside the 8x8 grid."""
    return 0 <= x < W and 0 <= y < H


def is_hazard(cells, x, y):
    """True if (x, y) holds a hazard.  Out-of-bounds counts as impassable."""
    if not in_bounds(x, y):
        return True
    return cells[y * W + x] == "hazard"


def _passable(cells, x, y):
    return in_bounds(x, y) and cells[y * W + x] != "hazard"


def _heuristic(x, y, gx, gy):
    """Manhattan distance -- admissible and consistent for 4-connected unit steps."""
    return abs(x - gx) + abs(y - gy)


def find_path(cells, start, goal):
    """A* from `start` to `goal` over a 64-entry row-major `cells` list.

    Returns [[x, y], ...] including both the start and the goal cell, or []
    when the goal is unreachable (or either endpoint is invalid/hazardous).
    """
    if cells is None or len(cells) != W * H:
        return []

    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        return []
    if is_hazard(cells, gx, gy):
        return []
    if (sx, sy) == (gx, gy):
        return [[sx, sy]]

    counter = 0
    start_h = _heuristic(sx, sy, gx, gy)
    open_heap = [(start_h, start_h, counter, sx, sy)]
    came_from = {}
    best_g = {(sx, sy): 0}
    closed = set()

    while open_heap:
        _f, _h, _c, cx, cy = heapq.heappop(open_heap)
        cur = (cx, cy)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == (gx, gy):
            path = []
            node = cur
            while node is not None:
                path.append([node[0], node[1]])
                node = came_from.get(node)
            path.reverse()
            return path

        g_next = best_g[cur] + 1
        for dx, dy, _name in NEIGHBOR_ORDER:
            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            if nxt in closed or not _passable(cells, nx, ny):
                continue
            if nxt in best_g and best_g[nxt] <= g_next:
                continue
            best_g[nxt] = g_next
            came_from[nxt] = cur
            counter += 1
            nh = _heuristic(nx, ny, gx, gy)
            heapq.heappush(open_heap, (g_next + nh, nh, counter, nx, ny))

    return []


def move_between(a, b):
    """Direction name that steps from cell `a` to adjacent cell `b`, else None."""
    dx = int(b[0]) - int(a[0])
    dy = int(b[1]) - int(a[1])
    for name, delta in MOVE_DELTA.items():
        if delta == (dx, dy):
            return name
    return None


def apply_move(pos, move):
    """[x, y] reached by applying `move` to `pos`.  Unknown move -> unchanged."""
    dx, dy = MOVE_DELTA.get(move, (0, 0))
    return [int(pos[0]) + dx, int(pos[1]) + dy]


def next_move(cells, pos, goal):
    """First step of the A* route from `pos` to `goal`.

    Returns "NORTH" | "SOUTH" | "EAST" | "WEST", or None when there is no path
    (or `pos` is already the goal).
    """
    path = find_path(cells, pos, goal)
    if len(path) < 2:
        return None
    return move_between(path[0], path[1])
