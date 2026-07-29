"""Domain profiles — the specialization engine for the SAR simulator.

A department picks a profile (urban collapse / wildland fire / swiftwater) and a
handful of tunables, and this module produces a concrete, *solvable* scenario:
terrain, victim locations, a resource budget, and the set of Safety-Officer
doctrine rules that apply. The same tick engine runs every profile; what changes
is the terrain semantics, the doctrine, and the language of the operation.

Pure Python, stdlib only. This module is imported by the Jac runtime (which
bundles its own CPython with no numpy), so it must stay dependency-free.

Grid convention matches CONTRACT.md: origin (0,0) top-left, x = column (east),
y = row (south). Internally every cell is one of:
    "open"     traversable
    "hazard"   impassable / lethal (collapse | fire | hydraulic)
    "staging"  the responders' entry point / safety zone (also traversable)
    "victim"   a person to reach (traversable once reached)
The profile supplies the human labels the UI shows for each.
"""

import heapq

GRID_W = 8
GRID_H = 8
CELL_COUNT = GRID_W * GRID_H

# ICS (Incident Command System) roles — the same four across every domain.
# These are the real command structure US emergency response runs on.
ROLE_IC = "Incident Commander"
ROLE_OPS = "Operations"
ROLE_RESCUE = "Rescue Team"
ROLE_SAFETY = "Safety Officer"
ROLES = [ROLE_IC, ROLE_OPS, ROLE_RESCUE, ROLE_SAFETY]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
# Each profile defines terrain language, the resource being managed, the active
# Safety-Officer doctrine rules, and how a disruption reads. Doctrine rule ids:
#   in_bounds     never leave the operational area
#   no_hazard     never enter a lethal cell (collapse / fire / hydraulic)
#   air_reserve   turn back while enough resource remains to egress (rule of thirds)
#   escape_route  never sever the path back to the safety zone (LCES)
#   no_thrash     don't oscillate one cell back and forth
#   no_loop       don't re-enter the same cell repeatedly (livelock guard)

PROFILES = {
    "urban": {
        "id": "urban",
        "name": "Urban Search & Rescue",
        "code": "US&R",
        "agency": "FEMA US&R Task Force",
        "summary": (
            "Locate and reach victims trapped in the void spaces of a collapsed "
            "structure. Unstable rubble is lethal; monitor air and keep an egress path."
        ),
        "resource_label": "SCBA air",
        "resource_unit": "min",
        "hazard_noun": "collapse zone",
        "safe_noun": "cleared void",
        "objective_noun": "trapped victim",
        "staging_noun": "staging area",
        "disruption_label": "SECONDARY COLLAPSE",
        "disruption_verb": "collapses",
        "cell_labels": {
            "open": "Cleared void",
            "hazard": "Collapse zone",
            "staging": "Staging area",
            "victim": "Trapped victim",
        },
        "doctrine": ["in_bounds", "no_hazard", "air_reserve", "escape_route",
                     "no_thrash", "no_loop"],
    },
    "wildland": {
        "id": "wildland",
        "name": "Wildland Fire Rescue",
        "code": "WILDLAND",
        "agency": "Wildland Fire Crew",
        "summary": (
            "Reach a trapped party ahead of an advancing fire. Active fire is lethal. "
            "LCES doctrine: never let the fire sever your escape route to the safety zone."
        ),
        "resource_label": "Egress window",
        "resource_unit": "min",
        "hazard_noun": "active fire",
        "safe_noun": "black / burned",
        "objective_noun": "trapped party",
        "staging_noun": "safety zone",
        "disruption_label": "FIRE SPREAD",
        "disruption_verb": "ignites",
        "cell_labels": {
            "open": "Black / safe",
            "hazard": "Active fire",
            "staging": "Safety zone",
            "victim": "Trapped party",
        },
        "doctrine": ["in_bounds", "no_hazard", "escape_route", "air_reserve",
                     "no_thrash", "no_loop"],
    },
    "swiftwater": {
        "id": "swiftwater",
        "name": "Swiftwater Rescue",
        "code": "SWIFTWATER",
        "agency": "Swiftwater Rescue Team",
        "summary": (
            "Reach victims in moving water. Hydraulics and strainers are lethal. "
            "Manage cold-water exposure and keep a route back to the bank."
        ),
        "resource_label": "Exposure budget",
        "resource_unit": "min",
        "hazard_noun": "hydraulic / strainer",
        "safe_noun": "slack water",
        "objective_noun": "victim in water",
        "staging_noun": "bank staging",
        "disruption_label": "RISING WATER",
        "disruption_verb": "floods",
        "cell_labels": {
            "open": "Slack water",
            "hazard": "Hydraulic / strainer",
            "staging": "Bank staging",
            "victim": "Victim in water",
        },
        "doctrine": ["in_bounds", "no_hazard", "air_reserve", "escape_route",
                     "no_thrash", "no_loop"],
    },
}

DEFAULT_PROFILE = "urban"

# Risk tolerance → the air-management reserve multiplier (rule of thirds).
# The Safety Officer orders Return-To-Base once remaining resource can no longer
# cover (egress distance x multiplier). Conservative crews turn back with more
# air in the bottle; aggressive crews push nearly to empty.
RISK_LEVELS = {
    "conservative": {"reserve": 2.0, "label": "Conservative"},
    "standard": {"reserve": 1.3, "label": "Standard"},
    "aggressive": {"reserve": 1.05, "label": "Aggressive"},
}
DEFAULT_RISK = "standard"

# Tunable ranges, surfaced to the setup screen. Every one of these changes the
# simulation mechanically — none are cosmetic.
TUNABLES = {
    "team_size":      {"min": 1, "max": 6, "default": 3,
                       "label": "Team size", "help": "Responders on the operation. More crew extends the resource window (rotation)."},
    "resource_budget":{"min": 16, "max": 72, "default": 44,
                       "label": "Resource budget", "help": "Operational window (reach victims AND egress to staging)."},
    "hazard_density": {"min": 0, "max": 6, "default": 3,
                       "label": "Hazard density", "help": "How much of the area is impassable. Higher = harder routing."},
    "victim_count":   {"min": 1, "max": 3, "default": 2,
                       "label": "Victims", "help": "People to locate and reach."},
    "risk_tolerance": {"values": ["conservative", "standard", "aggressive"],
                       "default": "standard",
                       "label": "Risk tolerance", "help": "Safety Officer's air-reserve doctrine (rule of thirds)."},
}


def profile(pid):
    return PROFILES.get(pid, PROFILES[DEFAULT_PROFILE])


# Keys copied verbatim onto the wire STATE.profile object (see CONTRACT.md v2).
_WIRE_KEYS = ("id", "name", "code", "agency", "summary", "resource_label",
              "resource_unit", "hazard_noun", "safe_noun", "objective_noun",
              "staging_noun", "disruption_label", "cell_labels")


def profile_wire(pid):
    """The profile object exactly as the frontend expects it in STATE.profile."""
    p = profile(pid)
    return {k: p[k] for k in _WIRE_KEYS}


def list_profiles():
    """Payload for the setup screen: all profiles + tunable specs."""
    return {
        "profiles": [profile_wire(pid) for pid in ("urban", "wildland", "swiftwater")],
        "tunables": TUNABLES,
        "default_profile": DEFAULT_PROFILE,
    }


def clamp_tunables(raw):
    """Coerce a raw tunables dict to valid, in-range values."""
    raw = raw or {}
    out = {}
    for key, spec in TUNABLES.items():
        v = raw.get(key, spec["default"])
        if "values" in spec:
            out[key] = v if v in spec["values"] else spec["default"]
        else:
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = spec["default"]
            out[key] = max(spec["min"], min(spec["max"], v))
    return out


# --------------------------------------------------------------------------- #
# Scenario generation (deterministic, always solvable)
# --------------------------------------------------------------------------- #
def _bfs_reachable(cells, start):
    """Set of (x,y) reachable from start over non-hazard cells (4-connected)."""
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < GRID_W and 0 <= n[1] < GRID_H and n not in seen:
                if cells[n[1] * GRID_W + n[0]] != "hazard":
                    seen.add(n)
                    stack.append(n)
    return seen


def path_len(cells, start, goal):
    """A* path length (cell count) over non-hazard cells, or -1 if unreachable.

    Mirrors lib/astar.py's neighbour order so generation agrees with the engine.
    """
    order = ((0, 1), (1, 0), (0, -1), (-1, 0))  # SOUTH, EAST, NORTH, WEST

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    pq = [(h(start), 0, start)]
    seen = set()
    while pq:
        f, g, cur = heapq.heappop(pq)
        if cur == goal:
            return g + 1
        if cur in seen:
            continue
        seen.add(cur)
        for dx, dy in order:
            n = (cur[0] + dx, cur[1] + dy)
            if 0 <= n[0] < GRID_W and 0 <= n[1] < GRID_H and n not in seen:
                if cells[n[1] * GRID_W + n[0]] != "hazard":
                    heapq.heappush(pq, (g + 1 + h(n), g + 1, n))
    return -1


def _lcg(seed):
    """Tiny deterministic PRNG (no `random` import needed; fully reproducible)."""
    state = [seed & 0x7FFFFFFF or 1]

    def nxt(n):
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return state[0] % n
    return nxt


def generate_scenario(profile_id, tunables, seed=7):
    """Build a solvable scenario for a profile + tunables.

    Returns a plain dict the Jac backend materialises into the graph:
        cells      64 row-major kind strings
        staging    [x,y]
        victims    [[x,y], ...]  (ordered by the search sequence)
        resource   int (starting budget)
        reserve    float (air-reserve multiplier from risk tolerance)
        risk       str
    Deterministic in (profile_id, tunables, seed): same inputs -> same map, so
    a rehearsal replays identically and a department can share a scenario seed.
    """
    t = clamp_tunables(tunables)
    prof = profile(profile_id)
    staging = (0, GRID_H - 1)  # bottom-left entry / safety zone

    # Retry generation until every victim is reachable and reachable in sequence.
    for attempt in range(200):
        rnd = _lcg(seed + attempt * 131)
        cells = ["open"] * CELL_COUNT

        # A partial wall down a middle column with gaps — forces real routing —
        # scaled by hazard_density, plus scattered hazards.
        density = t["hazard_density"]
        if density > 0:
            wall_x = 3 + (rnd(2))  # column 3 or 4
            gaps = {rnd(GRID_H), rnd(GRID_H)}  # 1–2 gaps
            for y in range(GRID_H):
                if y not in gaps:
                    cells[y * GRID_W + wall_x] = "hazard"
            scattered = density * 2
            for _ in range(scattered):
                x = rnd(GRID_W)
                y = rnd(GRID_H)
                cells[y * GRID_W + x] = "hazard"

        cells[staging[1] * GRID_W + staging[0]] = "open"  # never wall the entry

        # Place victims in the reachable region, away from staging.
        reachable = _bfs_reachable(cells, staging)
        candidates = sorted(
            [p for p in reachable
             if abs(p[0] - staging[0]) + abs(p[1] - staging[1]) >= 5
             and cells[p[1] * GRID_W + p[0]] == "open"],
            key=lambda p: (p[1], p[0]),
        )
        if len(candidates) < t["victim_count"]:
            continue

        victims = []
        pool = list(candidates)
        for _ in range(t["victim_count"]):
            victims.append(pool.pop(rnd(len(pool))))

        # Order victims into a greedy nearest-next search sequence from staging,
        # and confirm each leg is traversable (so the mission always completes).
        cur = staging
        ordered = []
        remaining = list(victims)
        ok = True
        while remaining:
            remaining.sort(key=lambda v: path_len(cells, cur, v))
            nxt = remaining.pop(0)
            if path_len(cells, cur, nxt) < 0:
                ok = False
                break
            ordered.append(nxt)
            cur = nxt
        if not ok:
            continue

        # Materialise kinds.
        cells[staging[1] * GRID_W + staging[0]] = "staging"
        for v in ordered:
            cells[v[1] * GRID_W + v[0]] = "victim"

        # Resource: budget scaled slightly by crew (rotation extends the window).
        resource = t["resource_budget"] + (t["team_size"] - 1) * 2
        reserve = RISK_LEVELS[t.get("risk_tolerance", DEFAULT_RISK)]["reserve"]

        return {
            "profile": prof["id"],
            "cells": cells,
            "staging": [staging[0], staging[1]],
            "victims": [[v[0], v[1]] for v in ordered],
            "resource": resource,
            "reserve": reserve,
            "risk": t.get("risk_tolerance", DEFAULT_RISK),
            "tunables": t,
        }

    # Fallback: an empty area with victims on the far side (always solvable).
    cells = ["open"] * CELL_COUNT
    cells[staging[1] * GRID_W + staging[0]] = "staging"
    fallback_victims = [[GRID_W - 1, 0], [GRID_W - 1, GRID_H - 1], [GRID_W - 1, 3]]
    victims = fallback_victims[: t["victim_count"]]
    for v in victims:
        cells[v[1] * GRID_W + v[0]] = "victim"
    return {
        "profile": prof["id"],
        "cells": cells,
        "staging": [staging[0], staging[1]],
        "victims": victims,
        "resource": t["resource_budget"],
        "reserve": RISK_LEVELS[t.get("risk_tolerance", DEFAULT_RISK)]["reserve"],
        "risk": t.get("risk_tolerance", DEFAULT_RISK),
        "tunables": t,
    }
