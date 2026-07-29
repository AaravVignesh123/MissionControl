"""Mission Control API — SAR operations simulator endpoints + the tick loop.

One `POST /function/tick` == one perceive -> plan -> verify -> act cycle:

    1. no-op (but advance the clock) if the operation already ended
    2. gather three independent advance proposals (LLM, policy-net, A*)
    3. the Safety Officer runs pure doctrine rules on the chosen advance
    4. HALT -> stay put, hand off Safety Officer -> IC, re-task Operations
       WITHDRAW (air/escape) -> Return-To-Base, operation aborts safely
       CLEARED -> commit the advance, burn one resource unit
    5. progress objectives (secure victims, then egress) / settle status

Wire format is frozen in CONTRACT.md v2; `_snapshot` is the single place it is
built. ICS roles: Incident Commander, Operations, Rescue Team, Safety Officer.
"""

import lib.astar as astar;
import lib.policy as policy;
import lib.profiles as profiles;

import from brain {
    commander_decompose,
    executor_propose,
    brain_mode
}

import from world {
    Mission,
    Cell,
    GRID_W,
    GRID_H,
    clockstr,
    build_from_scenario,
    get_mission,
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
    is_withdraw_rule,
    verify,
    verdict_dict
}

# ICS role names -- the exact strings the UI and the wire contract key off.
glob IC: str = "Incident Commander";
glob OPS: str = "Operations";
glob RESCUE: str = "Rescue Team";
glob SAFETY: str = "Safety Officer";


# ===========================================================================
# Proposers (LLM + policy net wired; A* is computed inline in the tick loop).
# ===========================================================================
"""Rescue Team's byLLM advance proposal.  Returns (move, confidence)."""
def _llm_proposal(situation: str) -> tuple[str, float] {
    # executor_propose is exception-proof by contract: a network/auth/parse
    # failure comes back as move="" and falls through to the next proposer.
    p = executor_propose(situation);
    if is_move(p.move) {
        return (p.move, 1.0);
    }
    return ("", 0.0);
}


"""Policy-net advance proposal.  Returns (move, softmax confidence 0..1)."""
def _net_proposal(cells: list[str], pos: list[int], goal: list[int]) -> tuple[str, float] {
    # Pure-Python inference (no numpy); degrades to ("", 0.0), never raises.
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
    return brain_mode();
}


# ------------------------------------------------------------------- helpers
"""The live Mission, opening a default urban operation on first touch."""
def _mission() -> Mission {
    found = [root -->[?:Mission]];
    if len(found) > 0 {
        return found[0];
    }
    m = get_mission();
    _open_mission(m);
    return m;
}


"""Lay the opening ICS transcript for a freshly built mission."""
def _open_mission(m: Mission) {
    prof = profiles.profile(m.profile_id);
    known = believed_cells(m);
    _compute_path(m, known);
    n = len(m.path);
    vc = len(m.victims_remaining);
    onoun = prof["objective_noun"] as str;
    add_log(m, IC, "info", f"{prof['code']} operation. {vc} {onoun}(s) reported. Command established.");

    # The Incident Commander decomposes the objective with the live reasoner
    # (Groq/Anthropic, or the offline mock). Total function — never raises — and
    # the mechanical task board (t1/t2/t3) is authoritative regardless, so a
    # slow or failed decomposition only costs us the briefing lines.
    plan = commander_decompose(m.objective_text);
    for pt in plan.tasks {
        add_log(m, IC, "info", f"Tasking {pt.owner}: {pt.desc}.");
    }

    add_handoff(m, IC, OPS, "t1", {"order": "plan search route"});
    if n > 0 {
        add_log(m, OPS, "info", f"Search route to nearest {onoun} plotted: {n} cells.");
    } else {
        add_log(m, OPS, "warn", "No safe route to the objective from staging.");
    }
    add_handoff(m, OPS, IC, "t1", {"route_cells": n});
    add_log(m, IC, "info", "Route accepted. Rescue Team cleared to advance.");
}


"""(Re)compute the cached A* plan from the team's current position to objective."""
def _compute_path(m: Mission, cells: list[str]) {
    p = astar.find_path(cells, [m.pos_x, m.pos_y], [m.target_x, m.target_y]) as list;
    m.path = p;
}


"""A counted replan -- happens after a HALT or when the plan desyncs."""
def _replan(m: Mission, cells: list[str]) {
    _compute_path(m, cells);
    m.replans = m.replans + 1;
    # Forget the previous cell: once Operations draws a NEW route the old cell is
    # no longer oscillation evidence, and a reroute around a fresh obstruction
    # often must step back the way it came.  Otherwise no_thrash deadlocks.
    m.prev_x = -1;
    m.prev_y = -1;
}


"""Next step along the CACHED plan, or "" if the plan is missing or stale.

Deliberately not a fresh A* call: the Rescue Team follows the route Operations
handed it, so a hazard that appeared after planning is caught by the Safety
Officer instead of being silently routed around.
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


"""Human-readable surroundings for the Rescue Team, over the LAST-KNOWN map."""
def _situation(m: Mission) -> str {
    prof = profiles.profile(m.profile_id);
    px = m.pos_x;
    py = m.pos_y;
    tx = m.target_x;
    ty = m.target_y;
    rlabel = prof["resource_label"] as str;
    hnoun = prof["hazard_noun"] as str;
    here: Cell | None = cell_at(m, px, py);
    around: list[str] = [];
    known = believed_cells(m);
    if here is not None {
        for n in cell_neighbors(here) {
            nx = n.x;
            ny = n.y;
            raw = known[ny * GRID_W + nx];
            # "BLOCKED" is a stable token every proposer (incl. the mock parser)
            # keys off; the domain noun rides along for the LLM's benefit.
            nk = f"BLOCKED ({hnoun})" if raw == "hazard" else "clear";
            d = astar.move_between([px, py], [nx, ny]);
            if d is not None {
                dname = d as str;
                around.append(f"{dname}=({nx},{ny}):{nk}");
            }
        }
    }
    joined = ", ".join(around);
    objn = "staging" if m.phase == "egress" else "victim";
    return f"At ({px},{py}), {rlabel} {m.resource} remaining, {objn} at ({tx},{ty}). Adjacent: {joined}.";
}


"""Set one task's state, rewriting the list so the node is marked dirty."""
def _set_task(m: Mission, tid: str, state: str) {
    out: list[dict[str, any]] = [];
    for t in m.tasks {
        if t["id"] == tid {
            out.append({"id": t["id"], "desc": t["desc"], "owner": t["owner"], "state": state});
        } else {
            out.append(t);
        }
    }
    m.tasks = out;
}


"""Build the frozen STATE object (CONTRACT.md v2) from the Mission subgraph."""
def _snapshot(m: Mission) -> dict[str, any] {
    cells = grid_cells(m);

    props: dict[str, any] | None = None;
    if len(m.proposals) > 0 { props = m.proposals; }
    verd: dict[str, any] | None = None;
    if len(m.verdict) > 0 { verd = m.verdict; }

    total = len(m.victims_remaining) + len(m.victims_rescued);

    return {
        "tick": m.tick,
        "clock": clockstr(m.tick),
        "status": m.status,
        "profile": profiles.profile_wire(m.profile_id),
        "config": m.config,
        "mission": {
            "objective": m.objective_text,
            "phase": m.phase,
            "tasks": m.tasks
        },
        "grid": {
            "w": GRID_W,
            "h": GRID_H,
            "cells": cells,
            "staging": [m.staging_x, m.staging_y]
        },
        "responder": {
            "pos": [m.pos_x, m.pos_y],
            "resource": m.resource,
            "resource_max": m.resource_max,
            "objective": [m.target_x, m.target_y],
            "path": m.path,
            "visited": m.visited
        },
        "victims": {
            "total": total,
            "reached": len(m.victims_rescued),
            "remaining": m.victims_remaining,
            "rescued": m.victims_rescued
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
            "rescued": len(m.victims_rescued),
            "llm_mode": _llm_mode()
        }
    };
}


# ----------------------------------------------------------------- endpoints
"""The 3 domain profiles + tunable specs, for the mission-setup screen."""
def:pub list_profiles() -> dict[str, any] {
    return profiles.list_profiles() as dict;
}


"""Configure and launch an operation: pick a domain profile + tunables."""
def:pub configure_mission(profile: str, tunables: dict[str, any], seed: int) -> dict[str, any] {
    sc = profiles.generate_scenario(profile, tunables, seed) as dict;
    config = {"profile": profile, "tunables": profiles.clamp_tunables(tunables), "seed": seed};
    m = build_from_scenario(sc, config);
    _open_mission(m);
    return _snapshot(m);
}


"""Full state snapshot.  The frontend polls this ~2x/sec."""
def:pub get_state() -> dict[str, any] {
    return _snapshot(_mission());
}


"""Rebuild the SAME scenario (deterministic) and return to tick 0."""
def:pub reset_mission() -> dict[str, any] {
    m = _mission();
    cfg = m.config;
    pid = cfg["profile"] as str if "profile" in cfg else m.profile_id;
    tun = cfg["tunables"] as dict if "tunables" in cfg else {};
    seed = cfg["seed"] as int if "seed" in cfg else 7;
    sc = profiles.generate_scenario(pid, tun, seed) as dict;
    m2 = build_from_scenario(sc, {"profile": pid, "tunables": tun, "seed": seed});
    _open_mission(m2);
    return _snapshot(m2);
}


"""Every remaining victim AND staging reachable from `pos` in `probe`?"""
def _reachable_all(m: Mission, probe: list[str], pos: list[int]) -> bool {
    for v in m.victims_remaining {
        vx = v[0] as int;
        vy = v[1] as int;
        if len(astar.find_path(probe, pos, [vx, vy]) as list) == 0 {
            return False;
        }
    }
    if len(astar.find_path(probe, pos, [m.staging_x, m.staging_y]) as list) == 0 {
        return False;
    }
    return True;
}


"""Drop a hazard on (x, y).  The planned route is deliberately NOT recomputed --
that is what gives the Safety Officer something to catch on the next tick."""
def:pub inject_hazard(x: int, y: int) -> dict[str, any] {
    m = _mission();
    prof = profiles.profile(m.profile_id);
    label = prof["disruption_label"] as str;

    if x == m.pos_x and y == m.pos_y {
        add_log(m, IC, "warn", f"Disruption refused at ({x},{y}): the team is there.");
        return _snapshot(m);
    }
    target: Cell | None = cell_at(m, x, y);
    if target is None {
        add_log(m, IC, "warn", f"Disruption refused at ({x},{y}): outside the area.");
        return _snapshot(m);
    }
    if target.kind == "victim" or target.kind == "staging" {
        add_log(m, IC, "warn", f"Disruption refused at ({x},{y}): objective / staging cell.");
        return _snapshot(m);
    }

    probe = grid_cells(m);
    probe[y * GRID_W + x] = "hazard";
    if not _reachable_all(m, probe, [m.pos_x, m.pos_y]) {
        add_log(m, IC, "warn", f"Disruption refused at ({x},{y}): would strand the operation.");
        return _snapshot(m);
    }

    target.kind = "hazard";
    # Real immediately (Safety Officer's live sensors see it now) but not yet on
    # the team's map -- they discover it when the Safety Officer halts them.
    key = f"{x},{y}";
    if key not in m.unknown_hazards {
        m.unknown_hazards = m.unknown_hazards + [key];
    }
    add_log(m, IC, "warn", f"{label}: new {prof['hazard_noun']} at ({x},{y}) -- not yet on the team's map.");
    return _snapshot(m);
}


"""The advance the Rescue Team would propose right now (same selection chain as
tick), so the disruption and the next tick agree on where the team is headed."""
def _intended_move(m: Mission) -> str {
    known = believed_cells(m);
    pos = [m.pos_x, m.pos_y];
    llm_pair = _llm_proposal(_situation(m));
    llm_move = llm_pair[0] as str;
    if is_move(llm_move) { return llm_move; }
    net_pair = _net_proposal(known, pos, [m.target_x, m.target_y]);
    net_move = net_pair[0] as str;
    if is_move(net_move) { return net_move; }
    return _plan_move(m);
}


"""THE DISRUPTION: a domain hazard on the cell the team is about to enter.

Not a fixed coordinate -- it targets whatever the Rescue Team actually intends
next (falling forward along the route), so the HALT fires regardless of which
proposer is winning, and never strands the operation."""
def:pub inject_disruption() -> dict[str, any] {
    m = _mission();
    if m.status != "running" {
        add_log(m, IC, "warn", "Disruption ignored: no active operation.");
        return _snapshot(m);
    }

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
        k = ground[cy * GRID_W + cx];
        if k == "hazard" or k == "victim" or k == "staging" { continue; }
        probe = grid_cells(m);
        probe[cy * GRID_W + cx] = "hazard";
        if _reachable_all(m, probe, [m.pos_x, m.pos_y]) {
            return inject_hazard(cx, cy);
        }
    }

    add_log(m, IC, "warn", "Disruption skipped: nothing ahead can be blocked without stranding the operation.");
    return _snapshot(m);
}


"""Back-compat alias for the old endpoint name."""
def:pub inject_hazard_ahead() -> dict[str, any] {
    return inject_disruption();
}


"""Advance exactly one perceive -> plan -> verify -> act cycle."""
def:pub tick() -> dict[str, any] {
    m = _mission();

    # 1 -- terminal operation: no-op, but the clock still advances.
    if m.status != "running" {
        m.tick = m.tick + 1;
        return _snapshot(m);
    }

    m.tick = m.tick + 1;

    prof = profiles.profile(m.profile_id);
    doctrine = prof["doctrine"] as list;

    # `cells` is GROUND TRUTH (kinds); only the Safety Officer looks at it.
    # `known` is the last-known passability map every proposer reasons over.
    cells = grid_cells(m);
    known = believed_cells(m);
    pos = [m.pos_x, m.pos_y];

    # 2 -- three independent proposals, all from the last-known map.
    situation = _situation(m);
    llm_pair = _llm_proposal(situation);
    net_pair = _net_proposal(known, pos, [m.target_x, m.target_y]);

    llm_move = llm_pair[0] as str;
    net_move = net_pair[0] as str;
    net_conf = net_pair[1] as float;

    if is_move(llm_move) { m.llm_calls = m.llm_calls + 1; }

    astar_move = _plan_move(m);
    replanned = False;
    if not is_move(astar_move) {
        _replan(m, known);
        replanned = True;
        astar_move = _plan_move(m);
        pc = len(m.path);
        add_log(m, OPS, "info", f"Route (re)plotted from current position: {pc} cells.");
    }

    # Selection chain llm -> net -> A*, unless the IC has pulled routing
    # authority after a HALT (then Operations' deterministic route wins).
    chosen = "";
    source = "";
    deferring = m.defer_ticks > 0;

    if deferring and is_move(astar_move) {
        chosen = astar_move;
        source = "astar";
        m.defer_ticks = m.defer_ticks - 1;
        add_log(m, IC, "info", "Operations holds routing authority this tick (post-HALT).");
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
        add_handoff(m, IC, RESCUE, "t2", {"order": "advance", "move": chosen});
        add_log(m, RESCUE, "info", f"Advancing {chosen} (source {source}, net conf {net_conf}).");
        add_handoff(m, RESCUE, SAFETY, "t2", {"move": chosen});
    } else {
        add_log(m, RESCUE, "warn", "No viable advance produced.");
    }

    # 3 -- the Safety Officer. Pure doctrine, no LLM, fully deterministic.
    prev: list[int] | None = None;
    if m.prev_x >= 0 and m.prev_y >= 0 { prev = [m.prev_x, m.prev_y]; }
    trusted = deferring and source == "astar";
    staging = [m.staging_x, m.staging_y];
    v: Verdict = verify(cells, pos, chosen, m.resource, prev, m.visit_counts,
                        trusted, staging, m.reserve, doctrine);
    m.verdict = verdict_dict(v);

    # 4 -- act.
    if v.vetoed {
        if is_withdraw_rule(v.rule) {
            _withdraw(m, v);
            return _snapshot(m);
        }
        _halt(m, v, chosen, replanned);
    } else {
        _commit(m, cells, chosen, v);
        _progress(m);
    }

    # 5 -- backstop: resource fully spent in the field.
    if m.status == "running" and m.resource <= 0 {
        m.status = "failed";
        _set_task(m, "t2", "blocked");
        add_log(m, IC, "warn", "Resource exhausted in the field. Operation failed.");
    }

    return _snapshot(m);
}


"""WITHDRAW ruling (air_reserve / escape_route): Return-To-Base, operation aborts
safely.  Doctrine, not defeat -- the Safety Officer pulled the team out in time."""
def _withdraw(m: Mission, v: Verdict) {
    m.vetoes = m.vetoes + 1;
    m.status = "aborted";
    _set_task(m, "t2", "blocked");
    _set_task(m, "t3", "done");
    add_handoff(m, SAFETY, IC, "t2", {"rule": v.rule, "action": "RTB"});
    add_log(m, SAFETY, "veto", f"WITHDRAW ({v.rule}): {v.reason}.");
    add_log(m, IC, "warn", "Return-To-Base ordered. Operation aborted -- crew safe.");
}


"""HALT ruling (recoverable): position untouched, IC re-tasks Operations, reroute."""
def _halt(m: Mission, v: Verdict, chosen: str, already_replanned: bool) {
    m.vetoes = m.vetoes + 1;
    _set_task(m, "t2", "blocked");
    _set_task(m, "t3", "active");
    rule = v.rule;

    if rule == "no_loop" {
        m.defer_ticks = 999;
        add_log(m, IC, "warn", "Rescue Team livelock -- routing authority transferred to Operations.");
    } else {
        m.defer_ticks = 1;
    }

    add_handoff(m, SAFETY, IC, "t2", {"rule": rule, "blocked_move": chosen});
    add_log(m, SAFETY, "veto", f"HALT ({rule}): {v.reason}.");
    add_log(m, IC, "info", "HALT acknowledged. Re-tasking Operations to re-route.");

    # The HALT is how the team LEARNS the hazard: the blocked cell graduates onto
    # the known map, so the reroute avoids it and it is never proposed again.
    tgt = v.target_cell;
    if tgt is not None and rule == "no_hazard" {
        tk = f"{tgt[0]},{tgt[1]}";
        if tk in m.unknown_hazards {
            remaining: list[str] = [];
            for k in m.unknown_hazards {
                if k != tk { remaining.append(k); }
            }
            m.unknown_hazards = remaining;
            add_log(m, OPS, "info", f"Map updated: ({tgt[0]},{tgt[1]}) marked impassable.");
        }
    }

    if already_replanned { return; }
    _replan(m, believed_cells(m));
    n = len(m.path);
    if n > 0 {
        add_log(m, OPS, "info", f"Route re-plotted around the obstruction: {n} cells.");
    } else {
        add_log(m, OPS, "warn", "No safe route remains from the current position.");
    }
}


"""Cleared advance: commit it, burn one resource unit, advance the cached plan."""
def _commit(m: Mission, cells: list[str], chosen: str, v: Verdict) {
    tgt = v.target_cell;
    if tgt is None { return; }

    _set_task(m, "t2", "active");
    _set_task(m, "t3", "pending");

    m.prev_x = m.pos_x;
    m.prev_y = m.pos_y;
    m.pos_x = tgt[0];
    m.pos_y = tgt[1];
    m.resource = m.resource - 1;
    m.moves_committed = m.moves_committed + 1;
    m.visited = m.visited + [[m.pos_x, m.pos_y]];

    vkey = f"{m.pos_x},{m.pos_y}";
    counts = dict(m.visit_counts);
    counts[vkey] = (counts.get(vkey, 0) as int) + 1;
    m.visit_counts = counts;

    p = m.path;
    if len(p) >= 2 and p[1][0] == m.pos_x and p[1][1] == m.pos_y {
        m.path = p[1:];
    } else {
        _compute_path(m, cells);
    }

    nx = m.pos_x;
    ny = m.pos_y;
    add_log(m, SAFETY, "success", f"Advance {chosen} to ({nx},{ny}) cleared.");
}


"""Nearest not-yet-reached victim from the current position, over `known`."""
def _nearest_victim(m: Mission, known: list[str]) -> list[int] | None {
    best: list[int] | None = None;
    bestd = 999999;
    for v in m.victims_remaining {
        vx = v[0] as int;
        vy = v[1] as int;
        d = len(astar.find_path(known, [m.pos_x, m.pos_y], [vx, vy]) as list);
        if d > 0 and d < bestd {
            bestd = d;
            best = [vx, vy];
        }
    }
    return best;
}


"""Objective progression: secure a reached victim, pick the next, then egress."""
def _progress(m: Mission) {
    reached_obj = m.pos_x == m.target_x and m.pos_y == m.target_y;
    if not reached_obj { return; }

    if m.phase == "search" {
        # Secure the victim at this cell.
        here = [m.pos_x, m.pos_y];
        rem: list[list[int]] = [];
        for v in m.victims_remaining {
            if not (v[0] == here[0] and v[1] == here[1]) { rem.append(v); }
        }
        m.victims_remaining = rem;
        m.victims_rescued = m.victims_rescued + [here];
        secured: Cell | None = cell_at(m, here[0], here[1]);
        if secured is not None { secured.kind = "open"; }
        add_log(m, RESCUE, "success", f"Victim reached and secured at ({here[0]},{here[1]}).");
        add_handoff(m, RESCUE, IC, "t2", {"victim_secured": here});

        known = believed_cells(m);
        if len(m.victims_remaining) == 0 {
            m.phase = "egress";
            m.target_x = m.staging_x;
            m.target_y = m.staging_y;
            _replan(m, known);
            add_log(m, IC, "info", "All victims secured. Egress to staging.");
            add_handoff(m, IC, OPS, "t1", {"order": "egress"});
        } else {
            nv = _nearest_victim(m, known);
            if nv is not None {
                m.target_x = nv[0];
                m.target_y = nv[1];
                _replan(m, known);
                add_log(m, OPS, "info", f"Next victim assigned at ({nv[0]},{nv[1]}).");
                add_handoff(m, OPS, RESCUE, "t2", {"objective": nv});
            }
        }
    } elif m.phase == "egress" {
        m.phase = "done";
        m.status = "complete";
        _set_task(m, "t2", "done");
        _set_task(m, "t3", "done");
        add_log(m, IC, "success", "All victims secured and team egressed. Mission complete.");
    }
}
