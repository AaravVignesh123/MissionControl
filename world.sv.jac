"""Mission Control world graph.

Holds the frozen seed map, the 8x8 `Cell` lattice wired with typed `Adj` edges,
the `Mission` node that carries all mutable run state, and the graph-persistent
`LogEntry` / `Handoff` chains that hang off the Mission so the transcript
survives across requests.

Grid convention (CONTRACT.md): origin (0,0) top-left, x = column (-> east),
y = row (v south).  NORTH = y-1, SOUTH = y+1, EAST = x+1, WEST = x-1.
"""

# ---------------------------------------------------------------- frozen seed
glob GRID_W: int = 8;
glob GRID_H: int = 8;
glob CELL_COUNT: int = 64;

glob START_X: int = 0;
glob START_Y: int = 0;
glob TARGET_X: int = 7;
glob TARGET_Y: int = 6;
glob ENERGY_START: int = 40;

# Wall at x=3 with TWO gaps, at (3,2) and (3,4), plus four scattered hazards.
#
# The second gap is load-bearing, not decoration.  With a single gap the wall
# has choke points: blocking (2,4), (3,4) or (4,4) seals the survivor off
# entirely, so the disruption has to refuse those cells and drops its hazard
# somewhere the executor never walks -- the veto then never fires.  With two
# gaps the baseline A* route is byte-identical (still 14 cells through (3,4))
# but NO cell on it is a choke point, so the disruption can always block the
# executor's very next step and the veto is guaranteed.  Verified exhaustively.
glob HAZARDS: list[list[int]] = [
    [3, 0], [3, 1], [3, 3], [3, 5], [3, 6], [3, 7],
    [5, 2], [6, 5], [1, 6], [5, 7]
];

glob GOAL_TEXT: str = "Locate and reach the survivor at (7,6). Avoid all hazards.";

glob MOVES: list[str] = ["NORTH", "SOUTH", "EAST", "WEST"];


# ------------------------------------------------------------------ archetypes
node Cell {
    has x: int = 0,
        y: int = 0,
        kind: str = "free";
}

"""Orthogonal adjacency in the 8x8 lattice (both directions per pair)."""
edge Adj: Cell --> Cell {}


node LogEntry {
    has seq: int = 0,
        tick: int = 0,
        agent: str = "Commander",
        level: str = "info",
        text: str = "";
}


node Handoff {
    # `from` / `to` are Python keywords and cannot be `has` field names (E0067),
    # so they are stored as src/dst and renamed on the wire.
    has seq: int = 0,
        tick: int = 0,
        src: str = "",
        dst: str = "",
        task_id: str = "",
        payload: dict[str, any] = {};
}


node Mission {
    has goal: str = "",
        status: str = "running",
        tick: int = 0,
        energy: int = 40;

    has pos_x: int = 0,
        pos_y: int = 0,
        prev_x: int = -1,
        prev_y: int = -1,
        target_x: int = 7,
        target_y: int = 6;

    has path: list[list[int]] = [],
        visited: list[list[int]] = [],
        tasks: list[dict[str, any]] = [];

    # Empty dict means "null" on the wire (i.e. before the first tick).
    has proposals: dict[str, any] = {},
        verdict: dict[str, any] = {};

    has moves_committed: int = 0,
        vetoes: int = 0,
        llm_calls: int = 0,
        replans: int = 0,
        seq: int = 0;

    # Hazards that exist in ground truth but the planning agents do not know
    # about yet, as "x,y" keys.  The Verifier reads authoritative live state;
    # Commander / Executor / Planner reason over the last-known map.  A hazard
    # graduates out of this list the moment a veto teaches the team about it.
    has unknown_hazards: list[str] = [];

    # How many cells have been entered how often, as "x,y" -> count.  Feeds the
    # Verifier's `no_loop` livelock guard.
    has visit_counts: dict[str, int] = {};

    # Ticks for which the Commander has taken routing authority away from the
    # Executor's own proposers and handed it to the Planner's A* route.  Set
    # by a veto -- this is the "re-delegate" half of the veto loop.
    has defer_ticks: int = 0;
}


edge Holds: Mission --> Cell {}
edge Logged: Mission --> LogEntry {}
edge Handed: Mission --> Handoff {}


# ------------------------------------------------------------------- accessors
"""Every Cell of the mission's grid, in creation (row-major) order."""
def mission_cells(m: Mission) -> list[Cell] {
    return [m ->:Holds:->];
}


"""Row-major list of 64 kind strings -- exactly the `grid.cells` wire format."""
def grid_cells(m: Mission) -> list[str] {
    out: list[str] = ["free" for _i in range(CELL_COUNT)];
    for c in mission_cells(m) {
        idx = c.y * GRID_W + c.x;
        if idx >= 0 and idx < CELL_COUNT {
            out[idx] = c.kind;
        }
    }
    return out;
}


"""The grid as the *planning* agents believe it to be.

Identical to `grid_cells` except that hazards the team has not discovered yet
still read as "free".  This is the map the Commander, Planner, Executor and the
policy net reason over; the Verifier always checks against `grid_cells`, which
is ground truth.  The gap between the two is what a veto exists to catch.
"""
def believed_cells(m: Mission) -> list[str] {
    out = grid_cells(m);
    for key in m.unknown_hazards {
        parts = key.split(",");
        if len(parts) == 2 {
            bx = int(parts[0]);
            by = int(parts[1]);
            idx = by * GRID_W + bx;
            if idx >= 0 and idx < CELL_COUNT {
                out[idx] = "free";
            }
        }
    }
    return out;
}


"""The Cell node at (x, y), or None when out of bounds."""
def cell_at(m: Mission, x: int, y: int) -> Cell | None {
    for c in mission_cells(m) {
        if c.x == x and c.y == y {
            return c;
        }
    }
    return None;
}


"""Orthogonal neighbours of a cell, walked over the typed Adj lattice."""
def cell_neighbors(c: Cell) -> list[Cell] {
    return [c ->:Adj:->];
}


# ------------------------------------------------------------ transcript nodes
def add_log(m: Mission, agent: str, level: str, text: str) {
    m.seq = m.seq + 1;
    e = LogEntry(seq=m.seq, tick=m.tick, agent=agent, level=level, text=text);
    m +>:Logged():+> e;
}


def add_handoff(m: Mission, src: str, dst: str, task_id: str, payload: dict[str, any]) {
    m.seq = m.seq + 1;
    h = Handoff(seq=m.seq, tick=m.tick, src=src, dst=dst, task_id=task_id, payload=payload);
    m +>:Handed():+> h;
}


"""Log entries oldest-first, as plain wire dicts."""
def log_list(m: Mission) -> list[dict[str, any]] {
    entries = [m ->:Logged:->];
    ordered = sorted(entries, key=lambda (e: LogEntry) -> int { return e.seq; });
    out: list[dict[str, any]] = [];
    for e in ordered {
        out.append({"tick": e.tick, "agent": e.agent, "level": e.level, "text": e.text});
    }
    return out;
}


"""Handoffs oldest-first, as plain wire dicts (src/dst renamed to from/to)."""
def handoff_list(m: Mission) -> list[dict[str, any]] {
    entries = [m ->:Handed:->];
    ordered = sorted(entries, key=lambda (h: Handoff) -> int { return h.seq; });
    out: list[dict[str, any]] = [];
    for h in ordered {
        out.append({
            "tick": h.tick,
            "from": h.src,
            "to": h.dst,
            "task_id": h.task_id,
            "payload": h.payload
        });
    }
    return out;
}


# ------------------------------------------------------------- world lifecycle
"""The seed task board the Commander starts every mission with."""
def seed_tasks() -> list[dict[str, any]] {
    return [
        {"id": "t1", "desc": "Plot safe route to survivor", "owner": "Planner", "state": "done"},
        {"id": "t2", "desc": "Advance one cell along route", "owner": "Executor", "state": "active"},
        {"id": "t3", "desc": "Re-route around new hazard", "owner": "Commander", "state": "pending"}
    ];
}


"""Tear down any existing mission subgraph (cells, logs, handoffs, mission)."""
def destroy_world() {
    for m in [root -->[?:Mission]] {
        for c in [m ->:Holds:->] { del c; }
        for e in [m ->:Logged:->] { del e; }
        for h in [m ->:Handed:->] { del h; }
        del m;
    }
}


"""Build the 8x8 lattice + Mission from the frozen seed.  Returns the Mission."""
def build_world() -> Mission {
    destroy_world();

    m = (root ++> Mission(
        goal=GOAL_TEXT,
        status="running",
        tick=0,
        energy=ENERGY_START,
        pos_x=START_X,
        pos_y=START_Y,
        prev_x=-1,
        prev_y=-1,
        target_x=TARGET_X,
        target_y=TARGET_Y,
        path=[],
        visited=[[START_X, START_Y]],
        tasks=seed_tasks(),
        proposals={},
        verdict={}
    )) as Mission;

    hazard_keys: list[str] = [f"{h[0]},{h[1]}" for h in HAZARDS];

    # Row-major construction so Holds-edge order == wire order.
    lattice: list[Cell] = [];
    for y in range(GRID_H) {
        for x in range(GRID_W) {
            kind = "free";
            if f"{x},{y}" in hazard_keys { kind = "hazard"; }
            if x == START_X and y == START_Y { kind = "start"; }
            if x == TARGET_X and y == TARGET_Y { kind = "target"; }
            c = Cell(x=x, y=y, kind=kind);
            m +>:Holds():+> c;
            lattice.append(c);
        }
    }

    # Wire the 4-connected Adj lattice (one directed edge each way per pair).
    for y in range(GRID_H) {
        for x in range(GRID_W) {
            src = lattice[y * GRID_W + x];
            if x + 1 < GRID_W {
                dst = lattice[y * GRID_W + x + 1];
                src +>:Adj():+> dst;
                dst +>:Adj():+> src;
            }
            if y + 1 < GRID_H {
                dsty = lattice[(y + 1) * GRID_W + x];
                src +>:Adj():+> dsty;
                dsty +>:Adj():+> src;
            }
        }
    }

    return m;
}


"""The live Mission, building the world on first touch."""
def get_mission() -> Mission {
    existing = [root -->[?:Mission]];
    if len(existing) > 0 {
        return existing[0];
    }
    return build_world();
}
