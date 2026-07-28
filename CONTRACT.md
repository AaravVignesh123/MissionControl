# Mission Control — Frozen State Contract

**This file is the single source of truth for the wire format.** Backend, frontend, and ML
all code against it independently. Do not change a field name without updating every consumer.

Grid is **8×8**. Origin `(0,0)` is **top-left**. `x` = column (→ east), `y` = row (↓ south).
So `NORTH` = `y-1`, `SOUTH` = `y+1`, `EAST` = `x+1`, `WEST` = `x-1`.

## Endpoints

Server runs on **port 8800** (port 8000 is occupied by another app on this machine).

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/function/get_state` | `{}` | Full state snapshot. Frontend polls this ~2×/sec. |
| POST | `/function/tick` | `{}` | Advance exactly one perceive→reason→verify→act cycle. Returns new state. |
| POST | `/function/reset_mission` | `{}` | Rebuild grid from the known-good seed, tick 0. |
| POST | `/function/inject_hazard` | `{"x":int,"y":int}` | Drop a hazard at a specific cell. Refused if it would strand the mission. |
| POST | `/function/inject_hazard_ahead` | `{}` | **The disruption — use this one.** Drops a hazard on the cell the Executor is about to enter, so the veto fires on the very next tick regardless of which proposer is winning. |

Every response is wrapped in the Jac envelope — **the state object is at `data.result`**:

```json
{"ok":true,"type":"response","data":{"result": <STATE>, "reports":[]},"error":null,"meta":{...}}
```

Frontend must read `json.data.result`. Ignore `_jac_type` / `_jac_id` / `_jac_archetype` keys.

## The STATE object

```jsonc
{
  "tick": 12,
  "status": "running",         // "running" | "complete" | "failed"

  "mission": {
    "goal": "Locate and reach the survivor at (7,6). Avoid all hazards.",
    "tasks": [
      {"id":"t1","desc":"Plot safe route to survivor","owner":"Planner","state":"done"},
      {"id":"t2","desc":"Advance one cell along route","owner":"Executor","state":"active"}
      // state: "pending" | "active" | "done" | "blocked"
    ]
  },

  "grid": {
    "w": 8, "h": 8,
    // Row-major array of 64 strings. index = y*8 + x
    // "free" | "hazard" | "target" | "start"
    "cells": ["start","free","free","hazard", "..."]
  },

  "executor": {
    "pos": [3,4],              // [x,y]
    "energy": 27,              // decrements 1 per committed move; mission fails at 0
    "path": [[3,4],[3,5],[4,5]], // current planned route incl. current pos; [] if none
    "visited": [[0,0],[1,0]]   // trail for rendering
  },

  // What each of the three proposers wanted THIS tick. null before first tick.
  "proposals": {
    "llm":   "NORTH",          // "NORTH"|"SOUTH"|"EAST"|"WEST"|null
    "net":   "NORTH",
    "astar": "EAST",
    "chosen":"NORTH",
    "source":"llm",            // which proposer won: "llm"|"net"|"astar"
    "agreed": false,           // did all three non-null proposers agree?
    "net_confidence": 0.87     // policy-net softmax max, 0..1
  },

  // The Verifier's ruling on `proposals.chosen`. null before first tick.
  "verdict": {
    "vetoed": true,
    "rule": "no_hazard",       // "no_hazard"|"in_bounds"|"energy_budget"|"no_thrash"|"no_loop"|"ok"
    "reason": "Move NORTH enters hazard at (3,3)",
    "target_cell": [3,3]       // cell the rejected move would have entered; null if n/a
  },

  // Newest LAST. Frontend animates the edge for entries with tick == state.tick.
  "handoffs": [
    {"tick":12,"from":"Verifier","to":"Commander","task_id":"t2",
     "payload":{"rule":"no_hazard","blocked_move":"NORTH"}}
  ],

  // Newest LAST. Frontend renders the tail and auto-scrolls.
  "log": [
    {"tick":12,"agent":"Commander","level":"info","text":"Decomposed mission into 2 tasks"},
    {"tick":12,"agent":"Verifier","level":"veto","text":"VETO: move NORTH enters hazard at (3,3)"}
    // level: "info" | "veto" | "success" | "warn"
  ],

  "stats": {
    "moves_committed": 11,
    "vetoes": 2,
    "llm_calls": 4,
    "replans": 2,
    "llm_mode": "live"         // "live" | "mock"  — surfaced in the UI so the demo is honest
  }
}
```

## Agent names (exact strings — the UI keys panels off these)

`"Commander"`, `"Executor"`, `"Verifier"`, `"Planner"`

## Rules the Verifier enforces (pure logic, never an LLM call)

| `rule` | Vetoes when |
|---|---|
| `in_bounds` | target cell is outside the 8×8 grid |
| `no_hazard` | target cell `kind == "hazard"` |
| `energy_budget` | `energy <= 0` |
| `no_thrash` | the move revisits the immediately previous cell (oscillation guard) |
| `no_loop` | the move re-enters a cell for the 3rd+ time (livelock guard; exempt for the Planner's A\* route) |

## Two world models — why the veto is meaningful

Ground truth and the team's belief are deliberately different:

- **The Verifier** checks against live ground truth. It is the safety authority.
- **Commander / Planner / Executor / policy net** reason over the *last-known* map — a freshly
  injected hazard is not on it yet.

A disruption is therefore invisible to the planners at the moment it lands. They propose a move
that *was* safe, the Verifier catches it against live sensor truth, and **the veto is how the team
learns the hazard exists**. The blocked cell then graduates onto the known map and the Planner
routes around it. Without this split the proposers would silently avoid every new hazard and the
Verifier would have nothing to catch.

**Veto → re-delegate.** A veto also transfers routing authority from the Executor's own proposers
to the Planner's deterministic A\* route for the next tick (`no_loop` transfers it permanently).
This is what stops a confidently-wrong proposer from being vetoed on the same move forever.
| `ok` | not a veto — the move is approved |

## Canonical seed map (frozen — all modules must agree)

```
START  = (0,0)          TARGET = (7,6)        ENERGY_START = 40
HAZARDS = (3,0) (3,1) (3,3) (3,5) (3,6) (3,7)   <- wall at x=3, TWO gaps: (3,2) and (3,4)
          (5,2) (6,5) (1,6) (5,7)
```
Baseline A\* solution is **14 cells**, threading the gap at `(3,4)`.

**The second gap is load-bearing.** With a single gap the wall has choke points — blocking
`(2,4)`, `(3,4)` or `(4,4)` seals the survivor off entirely, so the disruption has to refuse
those cells and drops its hazard somewhere the executor never walks, and the veto never fires.
With two gaps the baseline route is byte-identical but **no cell on it is a choke point**, so
the executor's next step can always be blocked. Verified exhaustively.

**Use `/function/inject_hazard_ahead` for the demo** rather than a fixed coordinate: the three
proposers don't always agree on the route, so a hardcoded cell can miss. Verified: disrupting at
any tick from 1–11 produces a veto with position held, and the mission still completes every time.
At tick 12+ the executor is adjacent to the survivor and the disruption is correctly refused —
the survivor's own cell must never be blocked.

## Module interfaces (frozen — these are the seams between parallel workstreams)

### `lib/astar.py` — pure Python, stdlib only
```python
def find_path(cells: list[str], start: tuple, goal: tuple) -> list[list[int]]
    """cells = 64 row-major strings. Returns [[x,y],...] incl. start and goal; [] if unreachable."""

def next_move(cells: list[str], pos: tuple, goal: tuple) -> str | None
    """Returns "NORTH"|"SOUTH"|"EAST"|"WEST", or None if no path."""
```

### `lib/policy.py` — pure Python, **NO numpy** (must run inside Jac's bundled runtime)
```python
def predict(cells: list[str], pos: tuple, goal: tuple) -> tuple[str, float]
    """Returns (move, confidence 0..1). Loads ml/weights/policy.json at import time.
       Must degrade gracefully to ("", 0.0) if weights are missing."""
```

### `brain.sv.jac` — the byLLM layer
```jac
obj PlannedTask { has id: str; has desc: str; has owner: str; }
obj TaskPlan    { has tasks: list[PlannedTask]; }
obj MoveProposal{ has move: str; has rationale: str; }

def commander_decompose(goal: str) -> TaskPlan by llm();
def executor_propose(situation: str) -> MoveProposal by llm();
def brain_mode() -> str;   # "live" | "mock" — drives stats.llm_mode
```
`move` must be one of `NORTH|SOUTH|EAST|WEST`; the backend treats anything else as invalid
and falls through to the next proposer.

## Invariants (tests assert these)

1. A vetoed move **never** changes `executor.pos`.
2. Every veto appends a `Verifier → Commander` handoff **and** a `level:"veto"` log entry.
3. `executor.energy` decrements only on a committed move.
4. `grid.cells` is always exactly 64 entries.
5. `tick` increases by exactly 1 per `/function/tick` call, veto or not.
6. `status` flips to `"complete"` the tick `executor.pos` equals the target cell.
