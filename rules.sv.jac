"""The Safety Officer's doctrine checks.

PURE LOGIC ONLY. There is deliberately no byLLM import, no `by llm()`, and no
network call anywhere in this module: the HALT is the climax of the demo and it
must be reproducible on every single run. Do not add an LLM here.

`lib.astar` is imported only for reachability math (escape-route / egress
distance); it is pure-stdlib pathfinding, not a reasoning call.

Rules, in evaluation order. Which are ACTIVE comes from the profile's doctrine
list; thresholds come from the risk tolerance (the `reserve` multiplier):
    in_bounds      target cell is outside the grid
    no_hazard      target cell is a hazard (collapse / fire / hydraulic)
    escape_route   the move severs the only safe path back to staging  -> WITHDRAW
    air_reserve    can't advance and still egress with reserve in hand  -> WITHDRAW
    no_thrash      the move oscillates back to the previous cell
    no_loop        the move re-enters a cell for the 3rd+ time (livelock)
    ok             approved
"""

import lib.astar as astar;

glob R_GRID_W: int = 8;
glob R_GRID_H: int = 8;
glob LOOP_LIMIT: int = 3;

glob DELTAS: dict[str, list[int]] = {
    "NORTH": [0, -1],
    "SOUTH": [0, 1],
    "EAST": [1, 0],
    "WEST": [-1, 0]
};


"""The Safety Officer's ruling on a single proposed move."""
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


"""Cell [x, y] that `move` from `pos` would enter. Unknown move -> `pos`."""
def target_of(pos: list[int], move: str) -> list[int] {
    if not is_move(move) {
        return [pos[0], pos[1]];
    }
    d = DELTAS[move];
    return [pos[0] + d[0], pos[1] + d[1]];
}


"""WITHDRAW rules end the operation safely (Return-To-Base) rather than reroute."""
def is_withdraw_rule(rule: str) -> bool {
    return rule == "air_reserve" or rule == "escape_route";
}


"""Verify a proposed move against the active doctrine. Deterministic.

`cells` is GROUND TRUTH passability ("hazard" vs "open"). `doctrine` is the set
of active rule ids for this profile. `staging` + `reserve` drive the two
withdrawal rules. Never calls out to anything but pure pathfinding.
"""
def verify(
    cells: list[str],
    pos: list[int],
    move: str,
    resource: int,
    prev: list[int] | None,
    visits: dict[str, int],
    trusted_route: bool,
    staging: list[int],
    reserve: float,
    doctrine: list[str]
) -> Verdict {
    if not is_move(move) {
        return Verdict(
            vetoed=True,
            rule="in_bounds",
            reason="No viable advance proposed; nothing to execute",
            target_cell=None
        );
    }

    tgt = target_of(pos, move);
    tx = tgt[0];
    ty = tgt[1];

    # in_bounds
    if tx < 0 or tx >= R_GRID_W or ty < 0 or ty >= R_GRID_H {
        return Verdict(vetoed=True, rule="in_bounds",
                       reason=f"Advance {move} leaves the operational area at ({tx},{ty})",
                       target_cell=tgt);
    }

    # no_hazard
    idx = ty * R_GRID_W + tx;
    kind = "open";
    if idx >= 0 and idx < len(cells) {
        kind = cells[idx];
    }
    if kind == "hazard" {
        return Verdict(vetoed=True, rule="no_hazard",
                       reason=f"Advance {move} enters a hazard at ({tx},{ty})",
                       target_cell=tgt);
    }

    # escape_route (WITHDRAW) -- after the move, is staging still reachable over
    # safe ground?  If a disruption has severed the way out, doctrine says do not
    # engage: withdraw.  (LCES for wildland; keep-a-route-to-the-bank for water.)
    if "escape_route" in doctrine {
        back = astar.find_path(cells, [tx, ty], staging) as list;
        if len(back) == 0 {
            return Verdict(vetoed=True, rule="escape_route",
                           reason=f"Advance {move} severs the escape route to staging -- withdraw",
                           target_cell=tgt);
        }
    }

    # air_reserve (WITHDRAW) -- rule of thirds.  Refuse to advance if, after this
    # step, the resource left can no longer cover the trip home with the required
    # reserve.  Conservative crews turn back with more in the tank.
    if "air_reserve" in doctrine {
        home = astar.find_path(cells, [tx, ty], staging) as list;
        egress = len(home);
        if egress > 0 {
            after = resource - 1;
            # Explicit float() calls: `x as float < y` mis-parses in Jac (the
            # cast binds looser than the comparison), which silently produced a
            # spurious veto on tick 1.  Do NOT revert to `as float` here.
            needed = float(egress) * reserve;
            afterf = float(after);
            if afterf < needed {
                return Verdict(vetoed=True, rule="air_reserve",
                               reason=f"advancing leaves {after} vs {int(needed)} needed to egress -- return to base",
                               target_cell=tgt);
            }
        }
    }

    # no_thrash -- oscillation guard against bouncing back one cell.
    if "no_thrash" in doctrine and prev is not None {
        if len(prev) == 2 and prev[0] == tx and prev[1] == ty {
            return Verdict(vetoed=True, rule="no_thrash",
                           reason=f"Advance {move} back-tracks to ({tx},{ty})",
                           target_cell=tgt);
        }
    }

    # no_loop -- livelock guard.  no_thrash only catches an immediate A->B->A
    # bounce; a proposer can still cycle a 4-cell ring forever.  Re-entering a
    # cell a third time is a livelock.  Exempt Operations' A* route: it is
    # recomputed from the live map and is provably acyclic, so a legitimate
    # reroute that re-crosses covered ground must not be blocked.
    if "no_loop" in doctrine and not trusted_route {
        key = f"{tx},{ty}";
        seen = visits.get(key, 0) as int;
        if seen >= LOOP_LIMIT {
            return Verdict(vetoed=True, rule="no_loop",
                           reason=f"Advance {move} re-enters ({tx},{ty}) a {seen + 1}th time -- livelock",
                           target_cell=tgt);
        }
    }

    return Verdict(vetoed=False, rule="ok",
                   reason=f"Advance {move} to ({tx},{ty}) cleared",
                   target_cell=tgt);
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
