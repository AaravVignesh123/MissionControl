"""The Verifier's constraint checks.

PURE LOGIC ONLY.  There is deliberately no byLLM import, no `by llm()`, and no
network call anywhere in this module: the veto is the climax of the demo and it
must be reproducible on every single run.  Do not add an LLM here.

Rules, in evaluation order (CONTRACT.md):
    in_bounds      target cell is outside the 8x8 grid
    no_hazard      target cell kind == "hazard"
    energy_budget  energy <= 0
    no_thrash      the move revisits the immediately previous cell
    no_loop        the move re-enters a cell for the 3rd+ time (livelock)
    ok             approved
"""

glob R_GRID_W: int = 8;
glob R_GRID_H: int = 8;

# Entering the same cell this many times (or more) is a livelock, not a route.
glob LOOP_LIMIT: int = 3;

glob DELTAS: dict[str, list[int]] = {
    "NORTH": [0, -1],
    "SOUTH": [0, 1],
    "EAST": [1, 0],
    "WEST": [-1, 0]
};


"""The Verifier's ruling on a single proposed move."""
obj Verdict {
    has vetoed: bool = False,
        rule: str = "ok",
        reason: str = "",
        target_cell: list[int] | None = None;
}


"""True when `move` is one of the four legal direction names."""
def is_move(move: str) -> bool {
    return move in DELTAS;
}


"""Cell [x, y] that `move` from `pos` would enter.  Unknown move -> `pos`."""
def target_of(pos: list[int], move: str) -> list[int] {
    if not is_move(move) {
        return [pos[0], pos[1]];
    }
    d = DELTAS[move];
    return [pos[0] + d[0], pos[1] + d[1]];
}


"""Verify a proposed move.  Deterministic; never calls out to anything."""
def verify(
    cells: list[str],
    pos: list[int],
    move: str,
    energy: int,
    prev: list[int] | None,
    visits: dict[str, int],
    trusted_route: bool
) -> Verdict {
    # A non-direction (no proposer produced a usable move) can never be
    # committed.  CONTRACT.md's `rule` enum has no slot for this, so it is
    # reported under `in_bounds` -- the move leads nowhere on the grid.
    if not is_move(move) {
        return Verdict(
            vetoed=True,
            rule="in_bounds",
            reason="No valid move proposed; nothing to execute",
            target_cell=None
        );
    }

    tgt = target_of(pos, move);
    tx = tgt[0];
    ty = tgt[1];

    # 1. in_bounds
    if tx < 0 or tx >= R_GRID_W or ty < 0 or ty >= R_GRID_H {
        return Verdict(
            vetoed=True,
            rule="in_bounds",
            reason=f"Move {move} leaves the grid at ({tx},{ty})",
            target_cell=tgt
        );
    }

    # 2. no_hazard
    idx = ty * R_GRID_W + tx;
    kind = "free";
    if idx >= 0 and idx < len(cells) {
        kind = cells[idx];
    }
    if kind == "hazard" {
        return Verdict(
            vetoed=True,
            rule="no_hazard",
            reason=f"Move {move} enters hazard at ({tx},{ty})",
            target_cell=tgt
        );
    }

    # 3. energy_budget
    if energy <= 0 {
        return Verdict(
            vetoed=True,
            rule="energy_budget",
            reason=f"Move {move} refused: energy budget exhausted",
            target_cell=tgt
        );
    }

    # 4. no_thrash -- oscillation guard against bouncing back one cell.
    if prev is not None {
        if len(prev) == 2 and prev[0] == tx and prev[1] == ty {
            return Verdict(
                vetoed=True,
                rule="no_thrash",
                reason=f"Move {move} back-tracks to the previous cell ({tx},{ty})",
                target_cell=tgt
            );
        }
    }

    # 5. no_loop -- livelock guard.  `no_thrash` only catches an immediate
    # A->B->A bounce; a proposer can still cycle round a 4-cell ring forever,
    # burning energy and never arriving.  (Observed for real: the Executor
    # looped (7,4)->(6,4)->(6,3)->(7,3) until the battery died.)  Re-entering
    # the same cell a third time is not navigation, it is a livelock.
    #
    # Exempt when the move comes from the Planner's A* route: that route is
    # recomputed from the live map, is provably acyclic, and following it
    # strictly cannot livelock.  Without this exemption a legitimate reroute
    # that has to re-cross ground the Executor already covered gets vetoed
    # forever, which is a livelock of the guard's own making.
    key = f"{tx},{ty}";
    seen = visits.get(key, 0) as int;
    if seen >= LOOP_LIMIT and not trusted_route {
        return Verdict(
            vetoed=True,
            rule="no_loop",
            reason=f"Move {move} re-enters ({tx},{ty}) for the {seen + 1}th time -- livelock",
            target_cell=tgt
        );
    }

    return Verdict(
        vetoed=False,
        rule="ok",
        reason=f"Move {move} to ({tx},{ty}) approved",
        target_cell=tgt
    );
}


"""Verdict -> the exact `verdict` wire dict from CONTRACT.md."""
def verdict_dict(v: Verdict) -> dict[str, any] {
    return {
        "vetoed": v.vetoed,
        "rule": v.rule,
        "reason": v.reason,
        "target_cell": v.target_cell
    };
}
