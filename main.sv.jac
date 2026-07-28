"""Mission Control API service -- the four public endpoints and the tick loop.

One `POST /function/tick` == one full perceive -> reason -> verify -> act cycle:

    1. no-op (but still advance the clock) if the mission already ended
    2. gather three independent move proposals (llm, policy-net, A*)
    3. run the Verifier's pure rules on the chosen move
    4. veto  -> stay put, hand off Verifier -> Commander, replan
       approve-> commit the move, burn one energy, hand off along the chain
    5. settle status

Wire format is frozen in CONTRACT.md; `_snapshot` is the single place it is built.
"""

import lib.astar as astar;
import lib.policy as policy;

import from brain {
    executor_propose,
    brain_mode
}

import from world {
    Mission,
    Cell,
    GRID_W,
    GRID_H,
    build_world,
    grid_cells,
    believed_cells,
    cell_at,
    cell_neighbors,
    add_log,
    add_handoff,
    log_list,
    handoff_list
}

import from rules {
    Verdict,
    is_move,
    verify,
    verdict_dict
}


# ===========================================================================
# INTEGRATION POINTS -- brain.sv.jac / lib/policy.py are owned by other agents.
# Each stub below returns ("", 0.0) so the selection chain falls straight
# through to A*.  Swap the body; do not change the signature.
# ===========================================================================

"""Executor's byLLM move proposal.  Returns (move, confidence)."""
def _llm_proposal(situation: str) -> tuple[str, float] {
    # WIRED -> brain.sv.jac.  executor_propose is exception-proof by contract:
    # a network/auth/parse failure comes back as move="" and falls through to
    # the next proposer rather than raising into the tick loop.
    p = executor_propose(situation);
    if is_move(p.move) {
        return (p.move, 1.0);
    }
    return ("", 0.0);
}


"""Policy-net move proposal.  Returns (move, softmax confidence 0..1)."""
def _net_proposal(cells: list[str], pos: list[int], goal: list[int]) -> tuple[str, float] {
    # WIRED -> lib/policy.py (pure-Python inference, no numpy).
    # predict() degrades to ("", 0.0) on missing/corrupt weights, never raises.
    r = policy.predict(cells, pos, goal) as tuple;
    move = r[0] as str;
    conf = r[1] as float;
    if is_move(move) {
        return (move, conf);
    }
    return ("", 0.0);
}


""""live" | "mock" -- surfaced as stats.llm_mode so the demo stays honest."""
def _llm_mode() -> str {
    # WIRED -> brain.sv.jac.  Reports the model object actually wired up
    # (isinstance check), so it cannot claim "live" while running mocked.
    return brain_mode();
}


# ------------------------------------------------------------------- helpers
"""The live Mission, bootstrapping a fresh world the first time it is asked for."""
def _mission() -> Mission {
    found = [root -->[?:Mission]];
    if len(found) > 0 {
        return found[0];
    }
    return _bootstrap();
}


"""Rebuild the world from the frozen seed and lay down the opening transcript."""
def _bootstrap() -> Mission {
    m = build_world();
    cells = grid_cells(m);
    _compute_path(m, cells);
    n = len(m.path);
    add_log(m, "Commander", "info", "Mission received. Decomposed into 3 tasks.");
    add_handoff(m, "Commander", "Planner", "t1", {"order": "plot safe route"});
    if n > 0 {
        add_log(m, "Planner", "info", f"Route plotted: {n} cells threading the gap at (3,4)");
    } else {
        add_log(m, "Planner", "warn", "No safe route exists from the start cell");
    }
    add_handoff(m, "Planner", "Commander", "t1", {"route_cells": n});
    add_log(m, "Commander", "info", "Route accepted. Executor cleared to advance.");
    return m;
}


"""(Re)compute the cached A* plan from the executor's current position."""
def _compute_path(m: Mission, cells: list[str]) {
    p = astar.find_path(cells, [m.pos_x, m.pos_y], [m.target_x, m.target_y]) as list;
    m.path = p;
}


"""A replan is a counted _compute_path -- it only happens after a veto."""
def _replan(m: Mission, cells: list[str]) {
    _compute_path(m, cells);
    m.replans = m.replans + 1;

    # Forget where we came from.  `no_thrash` exists to stop the Executor
    # bouncing between two cells, but once the Planner has drawn a NEW route
    # the previous cell is no longer evidence of oscillation -- and a reroute
    # around a fresh obstruction very often has to step back the way it came.
    # Without this the pair of rules deadlock: the only escape is backwards,
    # and no_thrash vetoes it every tick forever.  (Observed for real.)
    m.prev_x = -1;
    m.prev_y = -1;
}


"""Next step along the CACHED plan, or "" if the plan is missing or stale.

Deliberately not a fresh A* call: the Executor follows the route the Planner
handed it, so a hazard injected after planning is caught by the Verifier
instead of being silently routed around.
"""
def _plan_move(m: Mission) -> str {
    p = m.path;
    if len(p) < 2 {
        return "";
    }
    if p[0][0] != m.pos_x or p[0][1] != m.pos_y {
        return "";
    }
    mv = astar.move_between(p[0], p[1]);
    if mv is None {
        return "";
    }
    return mv as str;
}


"""Human-readable surroundings, walked over the typed Adj lattice."""
def _situation(m: Mission) -> str {
    px = m.pos_x;
    py = m.pos_y;
    tx = m.target_x;
    ty = m.target_y;
    en = m.energy;
    here: Cell | None = cell_at(m, px, py);
    around: list[str] = [];
    # Reported from the LAST-KNOWN map, not ground truth: the Executor is not
    # allowed to see a hazard the team has not discovered yet.
    known = believed_cells(m);
    if here is not None {
        for n in cell_neighbors(here) {
            nx = n.x;
            ny = n.y;
            nk = known[ny * GRID_W + nx];
            d = astar.move_between([px, py], [nx, ny]);
            if d is not None {
                dname = d as str;
                around.append(f"{dname}=({nx},{ny}):{nk}");
            }
        }
    }
    joined = ", ".join(around);
    return f"At ({px},{py}), energy {en}, survivor at ({tx},{ty}). Adjacent: {joined}.";
}


"""Set one task's state, rewriting the list so the node is marked dirty."""
def _set_task(m: Mission, tid: str, state: str) {
    out: list[dict[str, any]] = [];
    for t in m.tasks {
        if t["id"] == tid {
            out.append({
                "id": t["id"],
                "desc": t["desc"],
                "owner": t["owner"],
                "state": state
            });
        } else {
            out.append(t);
        }
    }
    m.tasks = out;
}


"""Build the frozen STATE object (CONTRACT.md) from the Mission subgraph."""
def _snapshot(m: Mission) -> dict[str, any] {
    cells = grid_cells(m);

    props: dict[str, any] | None = None;
    if len(m.proposals) > 0 {
        props = m.proposals;
    }
    verd: dict[str, any] | None = None;
    if len(m.verdict) > 0 {
        verd = m.verdict;
    }

    return {
        "tick": m.tick,
        "status": m.status,
        "mission": {
            "goal": m.goal,
            "tasks": m.tasks
        },
        "grid": {
            "w": GRID_W,
            "h": GRID_H,
            "cells": cells
        },
        "executor": {
            "pos": [m.pos_x, m.pos_y],
            "energy": m.energy,
            "path": m.path,
            "visited": m.visited
        },
        "proposals": props,
        "verdict": verd,
        "handoffs": handoff_list(m),
        "log": log_list(m),
        "stats": {
            "moves_committed": m.moves_committed,
            "vetoes": m.vetoes,
            "llm_calls": m.llm_calls,
            "replans": m.replans,
            "llm_mode": _llm_mode()
        }
    };
}


# ----------------------------------------------------------------- endpoints
"""Full state snapshot.  The frontend polls this ~2x/sec."""
def:pub get_state() -> dict[str, any] {
    return _snapshot(_mission());
}


"""Rebuild the grid from the known-good seed and return to tick 0."""
def:pub reset_mission() -> dict[str, any] {
    return _snapshot(_bootstrap());
}


"""Drop a hazard on (x, y).  The planned route is deliberately NOT recomputed --
that is what gives the Verifier something to catch on the next tick."""
def:pub inject_hazard(x: int, y: int) -> dict[str, any] {
    m = _mission();

    if x == m.pos_x and y == m.pos_y {
        add_log(m, "Commander", "warn", f"Refused hazard at ({x},{y}): executor is standing there");
        return _snapshot(m);
    }
    if x == m.target_x and y == m.target_y {
        add_log(m, "Commander", "warn", f"Refused hazard at ({x},{y}): that is the survivor");
        return _snapshot(m);
    }

    target: Cell | None = cell_at(m, x, y);
    if target is None {
        add_log(m, "Commander", "warn", f"Refused hazard at ({x},{y}): outside the grid");
        return _snapshot(m);
    }

    # Refuse any hazard that would strand the executor.  A disruption is meant
    # to force a reroute, not to end the mission -- and an unsolvable grid
    # stalls forever (every tick vetoes, nothing changes).
    probe = grid_cells(m);
    probe[y * GRID_W + x] = "hazard";
    escape = astar.find_path(probe, [m.pos_x, m.pos_y], [m.target_x, m.target_y]) as list;
    if len(escape) == 0 {
        add_log(m, "Commander", "warn", f"Refused hazard at ({x},{y}): it would seal off the survivor");
        return _snapshot(m);
    }

    target.kind = "hazard";

    # The hazard is real immediately -- the Verifier's live sensor feed sees it
    # this instant -- but the planning agents are still working from the map
    # they last had.  They find out when the Verifier blocks them.
    key = f"{x},{y}";
    if key not in m.unknown_hazards {
        m.unknown_hazards = m.unknown_hazards + [key];
    }
    add_log(m, "Commander", "warn", f"DISRUPTION: new hazard at ({x},{y}) -- not yet on the team's map");
    return _snapshot(m);
}


"""The move the Executor would propose right now, over the last-known map.

Shares the exact selection chain the tick loop uses (llm -> net -> cached plan),
so `inject_hazard_ahead` and the next `tick` provably agree on where the
executor is headed.  Returns "" when no proposer offers a usable move.
"""
def _intended_move(m: Mission) -> str {
    known = believed_cells(m);
    pos = [m.pos_x, m.pos_y];

    llm_pair = _llm_proposal(_situation(m));
    llm_move = llm_pair[0] as str;
    if is_move(llm_move) {
        return llm_move;
    }

    net_pair = _net_proposal(known, pos, [m.target_x, m.target_y]);
    net_move = net_pair[0] as str;
    if is_move(net_move) {
        return net_move;
    }

    return _plan_move(m);
}


"""THE DISRUPTION: drop a hazard on the cell the executor is about to enter.

This is the demo climax and it is deliberately not a fixed coordinate -- it
targets whatever the Executor actually intends next, so the veto fires no
matter which proposer is winning or where the executor has got to.
"""
def:pub inject_hazard_ahead() -> dict[str, any] {
    m = _mission();

    if m.status != "running" {
        add_log(m, "Commander", "warn", "Disruption ignored: mission is not running");
        return _snapshot(m);
    }

    # Candidates, best first: the cell the Executor intends to enter next
    # (a veto on the very next tick), then cells further along the route
    # (the executor walks into it a few ticks later).  The first candidate
    # that does not seal off the survivor wins.
    candidates: list[list[int]] = [];

    mv = _intended_move(m);
    if is_move(mv) {
        candidates.append(astar.apply_move([m.pos_x, m.pos_y], mv) as list);
    }
    for step in m.path {
        if step[0] != m.pos_x or step[1] != m.pos_y {
            candidates.append(step);
        }
    }

    ground = grid_cells(m);
    for c in candidates {
        cx = c[0] as int;
        cy = c[1] as int;
        if cx == m.target_x and cy == m.target_y {
            continue;
        }
        if ground[cy * GRID_W + cx] == "hazard" {
            continue;
        }
        probe = grid_cells(m);
        probe[cy * GRID_W + cx] = "hazard";
        escape = astar.find_path(probe, [m.pos_x, m.pos_y], [m.target_x, m.target_y]) as list;
        if len(escape) > 0 {
            return inject_hazard(cx, cy);
        }
    }

    add_log(m, "Commander", "warn", "Disruption skipped: no cell ahead can be blocked without stranding the mission");
    return _snapshot(m);
}


"""Advance exactly one perceive -> reason -> verify -> act cycle."""
def:pub tick() -> dict[str, any] {
    m = _mission();

    # 1 -- terminal mission: no-op, but the clock still advances.
    if m.status != "running" {
        m.tick = m.tick + 1;
        return _snapshot(m);
    }

    # The clock advances first so every log line and handoff written below
    # carries `tick == state.tick` (the frontend animates on that equality).
    m.tick = m.tick + 1;

    # `cells` is GROUND TRUTH -- only the Verifier may look at it.
    # `known` is the team's last-known map -- every proposer reasons over this.
    cells = grid_cells(m);
    known = believed_cells(m);
    pos = [m.pos_x, m.pos_y];

    # 2 -- gather three independent proposals, all from the last-known map.
    situation = _situation(m);
    llm_pair = _llm_proposal(situation);
    net_pair = _net_proposal(known, pos, [m.target_x, m.target_y]);

    llm_move = llm_pair[0] as str;
    net_move = net_pair[0] as str;
    net_conf = net_pair[1] as float;

    if is_move(llm_move) {
        m.llm_calls = m.llm_calls + 1;
    }

    astar_move = _plan_move(m);
    replanned = False;
    if not is_move(astar_move) {
        # The cached plan ran out or no longer starts here -- the Planner
        # redraws it before the Executor is allowed to propose.
        _replan(m, known);
        replanned = True;
        astar_move = _plan_move(m);
        pc = len(m.path);
        add_log(m, "Planner", "info", f"Route (re)plotted from current position: {pc} cells");
    }

    # Selection chain: llm -> net -> A* fallback.
    #
    # ...unless the Commander has pulled routing authority.  After a veto the
    # Executor's own proposers are exactly the thing that just got the team
    # blocked, so for the next tick the Planner's deterministic route wins.
    # This is the "re-delegate" half of the veto loop, and it is what stops a
    # confidently-wrong proposer from being vetoed on the same move forever.
    chosen = "";
    source = "";
    deferring = m.defer_ticks > 0;

    if deferring and is_move(astar_move) {
        chosen = astar_move;
        source = "astar";
        m.defer_ticks = m.defer_ticks - 1;
        add_log(m, "Commander", "info", "Routing authority held by Planner this tick (post-veto)");
    } elif is_move(llm_move) {
        chosen = llm_move;
        source = "llm";
    } elif is_move(net_move) {
        chosen = net_move;
        source = "net";
    } elif is_move(astar_move) {
        chosen = astar_move;
        source = "astar";
    }
    if deferring and source != "astar" {
        # No usable A* route to defer to -- drop back to the normal chain.
        m.defer_ticks = 0;
    }

    offered: list[str] = [];
    if is_move(llm_move) { offered.append(llm_move); }
    if is_move(net_move) { offered.append(net_move); }
    if is_move(astar_move) { offered.append(astar_move); }

    agreed = False;
    if len(offered) > 1 {
        agreed = True;
        for o in offered {
            if o != offered[0] { agreed = False; }
        }
    }

    m.proposals = {
        "llm": llm_move if is_move(llm_move) else None,
        "net": net_move if is_move(net_move) else None,
        "astar": astar_move if is_move(astar_move) else None,
        "chosen": chosen if chosen != "" else None,
        "source": source if source != "" else None,
        "agreed": agreed,
        "net_confidence": net_conf
    };

    if chosen != "" {
        add_log(m, "Executor", "info", f"Proposing {chosen} (source: {source}, net conf {net_conf})");
    } else {
        add_log(m, "Executor", "warn", "No proposer produced a usable move");
    }

    # 3 -- the Verifier. Pure rules, no LLM, fully deterministic.
    prev: list[int] | None = None;
    if m.prev_x >= 0 and m.prev_y >= 0 {
        prev = [m.prev_x, m.prev_y];
    }
    # A move that came from the Planner's route under post-veto deference is
    # exempt from the livelock guard -- see rules.sv.jac.
    trusted = deferring and source == "astar";
    v: Verdict = verify(cells, pos, chosen, m.energy, prev, m.visit_counts, trusted);
    m.verdict = verdict_dict(v);

    # 4 -- act.
    if v.vetoed {
        _veto(m, cells, v, chosen, replanned);
    } else {
        _commit(m, cells, v, chosen);
    }

    # 5 -- settle status.
    if m.pos_x == m.target_x and m.pos_y == m.target_y {
        m.status = "complete";
        _set_task(m, "t2", "done");
        _set_task(m, "t3", "done");
        add_log(m, "Commander", "success", "Survivor reached. Mission complete.");
    } elif m.energy <= 0 {
        m.status = "failed";
        _set_task(m, "t2", "blocked");
        add_log(m, "Commander", "warn", "Energy budget exhausted. Mission failed.");
    }

    return _snapshot(m);
}


"""Vetoed move: position is untouched, the Commander is told, the route is redrawn."""
def _veto(m: Mission, cells: list[str], v: Verdict, chosen: str, already_replanned: bool) {
    m.vetoes = m.vetoes + 1;
    _set_task(m, "t2", "blocked");
    _set_task(m, "t3", "active");

    rule = v.rule;

    # The Commander takes routing authority off the Executor's proposers.  One
    # tick is enough to break out of a blocked move; a livelock means the
    # proposer is systematically lost, so the Planner keeps the wheel.
    if rule == "no_loop" {
        m.defer_ticks = 999;
        add_log(m, "Commander", "warn",
                "Executor livelock detected -- routing authority transferred to Planner");
    } else {
        m.defer_ticks = 1;
    }
    reason = v.reason;
    add_handoff(m, "Verifier", "Commander", "t2", {"rule": rule, "blocked_move": chosen});
    add_log(m, "Verifier", "veto", f"VETO ({rule}): {reason}");
    add_log(m, "Commander", "info", "Veto received. Re-delegating route planning.");

    # The veto is how the team LEARNS about the hazard: the blocked cell
    # graduates onto the known map, so the replan below routes around it and
    # the same move is never proposed twice.
    tgt = v.target_cell;
    if tgt is not None and rule == "no_hazard" {
        tk = f"{tgt[0]},{tgt[1]}";
        if tk in m.unknown_hazards {
            remaining: list[str] = [];
            for k in m.unknown_hazards {
                if k != tk { remaining.append(k); }
            }
            m.unknown_hazards = remaining;
            add_log(m, "Planner", "info", f"Map updated: ({tgt[0]},{tgt[1]}) marked impassable");
        }
    }

    # The proposal step may already have redrawn the route this tick; replanning
    # twice would double-count stats.replans on a stalled mission.
    if already_replanned {
        return;
    }

    _replan(m, believed_cells(m));
    n = len(m.path);
    if n > 0 {
        add_log(m, "Planner", "info", f"Route re-plotted around the obstruction: {n} cells");
    } else {
        add_log(m, "Planner", "warn", "No safe route remains from the current position");
    }
}


"""Approved move: commit it, burn one energy, advance the cached plan."""
def _commit(m: Mission, cells: list[str], v: Verdict, chosen: str) {
    tgt = v.target_cell;
    if tgt is None {
        return;
    }

    _set_task(m, "t2", "active");
    _set_task(m, "t3", "pending");

    add_handoff(m, "Commander", "Executor", "t2", {"order": "advance", "move": chosen});
    add_handoff(m, "Executor", "Verifier", "t2", {"move": chosen});

    m.prev_x = m.pos_x;
    m.prev_y = m.pos_y;
    m.pos_x = tgt[0];
    m.pos_y = tgt[1];
    m.energy = m.energy - 1;
    m.moves_committed = m.moves_committed + 1;
    m.visited = m.visited + [[m.pos_x, m.pos_y]];

    # Feeds the Verifier's no_loop guard.  Rebuilt rather than mutated so the
    # node is marked dirty and the new count actually persists.
    vkey = f"{m.pos_x},{m.pos_y}";
    counts = dict(m.visit_counts);
    counts[vkey] = (counts.get(vkey, 0) as int) + 1;
    m.visit_counts = counts;

    # Walk the cached plan forward one step; only rebuild if it desynced.
    p = m.path;
    if len(p) >= 2 and p[1][0] == m.pos_x and p[1][1] == m.pos_y {
        m.path = p[1:];
    } else {
        _compute_path(m, cells);
    }

    nx = m.pos_x;
    ny = m.pos_y;
    en = m.energy;
    add_log(m, "Executor", "success", f"Committed {chosen} -> ({nx},{ny}); energy {en}");
}
