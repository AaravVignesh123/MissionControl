"""Local A* / BFS for the ML pipeline ONLY.

Deliberately self-contained: `lib/astar.py` is owned by another workstream and
may not exist when this runs. Nothing here imports it, and nothing outside
`ml/` should import this. Same grid conventions as CONTRACT.md.

On a 4-connected uniform-cost grid, BFS gives exact shortest-path distances, so
`dist_map` below is an exact optimal-cost oracle -- that is what generates the
behavioural-cloning labels.
"""

from collections import deque
import heapq

W = 8
H = 8
MOVES = ["NORTH", "SOUTH", "EAST", "WEST"]
DELTAS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
INF = float("inf")


def passable(cells, x, y):
    return 0 <= x < W and 0 <= y < H and cells[y * W + x] != "hazard"


def dist_map(cells, goal):
    """BFS from the goal. Returns a 64-list of hop counts (INF if unreachable)."""
    gx, gy = goal
    d = [INF] * (W * H)
    if not passable(cells, gx, gy):
        return d
    d[gy * W + gx] = 0
    q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        nd = d[y * W + x] + 1
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if passable(cells, nx, ny) and d[ny * W + nx] == INF:
                d[ny * W + nx] = nd
                q.append((nx, ny))
    return d


def optimal_moves(cells, pos, goal, d=None):
    """Every first move that lies on SOME shortest path. [] if unreachable."""
    if d is None:
        d = dist_map(cells, goal)
    px, py = pos
    if not passable(cells, px, py):
        return []
    here = d[py * W + px]
    if here == INF or here == 0:
        return []
    out = []
    for mv in MOVES:
        dx, dy = DELTAS[mv]
        nx, ny = px + dx, py + dy
        if passable(cells, nx, ny) and d[ny * W + nx] == here - 1:
            out.append(mv)
    return out


def best_move(cells, pos, goal, d=None):
    """The single labelled move: optimal, with a deterministic tie-break.

    Tie-break (documented so the net can actually learn it): among optimal
    moves, prefer the axis with the larger remaining |delta|; within an axis
    prefer the direction that closes the gap. Full priority order is therefore
    a pure function of (dx, dy), so it is reproducible from the features.
    """
    opts = optimal_moves(cells, pos, goal, d)
    if not opts:
        return None
    px, py = pos
    gx, gy = goal
    dx, dy = gx - px, gy - py
    horiz = "EAST" if dx > 0 else ("WEST" if dx < 0 else None)
    vert = "SOUTH" if dy > 0 else ("NORTH" if dy < 0 else None)
    h_away = "WEST" if horiz == "EAST" else "EAST"
    v_away = "NORTH" if vert == "SOUTH" else "SOUTH"

    if abs(dx) >= abs(dy):
        order = [horiz, vert, v_away, h_away]
    else:
        order = [vert, horiz, h_away, v_away]
    for mv in order:
        if mv is not None and mv in opts:
            return mv
    for mv in MOVES:  # dx == dy == 0 corner case
        if mv in opts:
            return mv
    return None


def find_path(cells, start, goal):
    """A* with Manhattan heuristic. [[x,y],...] incl. both ends; [] if none."""
    sx, sy = start
    gx, gy = goal
    if not passable(cells, sx, sy) or not passable(cells, gx, gy):
        return []
    if (sx, sy) == (gx, gy):
        return [[sx, sy]]

    def h(x, y):
        return abs(x - gx) + abs(y - gy)

    g = {(sx, sy): 0}
    came = {}
    pq = [(h(sx, sy), 0, (sx, sy))]
    seen = set()
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == (gx, gy):
            path = [list(cur)]
            while cur in came:
                cur = came[cur]
                path.append(list(cur))
            path.reverse()
            return path
        cx, cy = cur
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if not passable(cells, nx, ny):
                continue
            ng = gc + 1
            if ng < g.get((nx, ny), INF):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(pq, (ng + h(nx, ny), ng, (nx, ny)))
    return []


def next_move(cells, pos, goal):
    """A*'s own first move (path-derived), used for the seed-map sanity check."""
    p = find_path(cells, pos, goal)
    if len(p) < 2:
        return None
    (x0, y0), (x1, y1) = p[0], p[1]
    for mv, (dx, dy) in DELTAS.items():
        if (x0 + dx, y0 + dy) == (x1, y1):
            return mv
    return None


# --- canonical seed map from CONTRACT.md ----------------------------------
SEED_START = (0, 0)
SEED_TARGET = (7, 6)
SEED_HAZARDS = [
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7),
    (5, 2), (6, 5), (1, 6), (5, 7),
]


def seed_cells():
    cells = ["free"] * (W * H)
    for (x, y) in SEED_HAZARDS:
        cells[y * W + x] = "hazard"
    cells[SEED_START[1] * W + SEED_START[0]] = "start"
    cells[SEED_TARGET[1] * W + SEED_TARGET[0]] = "target"
    return cells
