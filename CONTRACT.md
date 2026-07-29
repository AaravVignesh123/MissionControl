# Mission Control — Wire Contract (v2, SAR simulator)

**Single source of truth for the wire format.** Backend and frontend code against this.

Mission Control is a **search-and-rescue operations simulator**. A department picks a domain
**profile**, sets **tunables**, and rehearses the operation. Four ICS agents run it; the
**Safety Officer** enforces doctrine and can halt the team. Grid is 8×8, origin `(0,0)`
top-left, `x`=column (east), `y`=row (south). NORTH=y-1, SOUTH=y+1, EAST=x+1, WEST=x-1.

Server runs on **port 8800**.

## ICS roles (exact strings — panels/log/handoffs key off these)

`"Incident Commander"` · `"Operations"` · `"Rescue Team"` · `"Safety Officer"`

(Formerly Commander / Planner / Executor / Verifier.)

## Endpoints

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/function/list_profiles` | `{}` | The 3 domain profiles + tunable specs, for the setup screen. |
| POST | `/function/configure_mission` | `{"profile":str,"tunables":{...},"seed":int}` | Build a scenario and start it. Returns state. |
| POST | `/function/get_state` | `{}` | Full state snapshot. Frontend polls this. |
| POST | `/function/tick` | `{}` | Advance one perceive→plan→verify→act cycle. |
| POST | `/function/reset_mission` | `{}` | Rebuild with the last config. |
| POST | `/function/inject_disruption` | `{}` | **The disruption** — domain hazard on the cell the team is about to enter. |

Every response is the Jac envelope; **the payload is at `data.result`**. Ignore `_jac_*` keys.

## `list_profiles` result

```jsonc
{
  "profiles": [
    {"id":"urban","name":"Urban Search & Rescue","code":"US&R","agency":"FEMA US&R Task Force",
     "summary":"...","resource_label":"SCBA air","hazard_noun":"collapse zone",
     "cell_labels":{"open":"Cleared void","hazard":"Collapse zone","staging":"Staging area","victim":"Trapped victim"}},
    {"id":"wildland", ...}, {"id":"swiftwater", ...}
  ],
  "tunables": {
    "team_size":{"min":1,"max":6,"default":3,"label":"Team size","help":"..."},
    "resource_budget":{"min":16,"max":72,"default":44,"label":"Resource budget","help":"..."},
    "hazard_density":{"min":0,"max":6,"default":3,"label":"Hazard density","help":"..."},
    "victim_count":{"min":1,"max":3,"default":2,"label":"Victims","help":"..."},
    "risk_tolerance":{"values":["conservative","standard","aggressive"],"default":"standard","label":"Risk tolerance","help":"..."}
  },
  "default_profile":"urban"
}
```

## STATE object (`get_state` / `tick` / `configure_mission` / `reset_mission` / `inject_disruption`)

```jsonc
{
  "tick": 12,
  "clock": "T+00:12",              // mission clock derived from tick (MM:SS-ish)
  "status": "running",             // "running" | "complete" | "aborted" | "failed"
                                   //   complete = all victims reached AND team egressed to staging
                                   //   aborted  = Safety Officer ordered Return-To-Base (air reserve)
                                   //   failed   = resource hit 0 in the field (backstop)

  "profile": { /* the chosen profile object from list_profiles, verbatim */
    "id":"urban","name":"Urban Search & Rescue","code":"US&R","agency":"...","summary":"...",
    "resource_label":"SCBA air","resource_unit":"min","hazard_noun":"collapse zone",
    "safe_noun":"cleared void","objective_noun":"trapped victim","staging_noun":"staging area",
    "disruption_label":"SECONDARY COLLAPSE",
    "cell_labels":{"open":"...","hazard":"...","staging":"...","victim":"..."}
  },

  "config": {"team_size":3,"resource_budget":44,"hazard_density":3,"victim_count":2,
             "risk_tolerance":"standard","seed":7},

  "mission": {
    "objective": "Reach 2 trapped victims and egress to staging.",
    "phase": "search",             // "search" (heading to a victim) | "egress" (heading home) | "done"
    "tasks": [ {"id":"t1","desc":"...","owner":"Operations","state":"done"},
               {"id":"t2","desc":"...","owner":"Rescue Team","state":"active"} ]
    // state: "pending" | "active" | "done" | "blocked"
  },

  "grid": {
    "w":8, "h":8,
    "cells": [ /* 64 row-major kind strings: "open"|"hazard"|"staging"|"victim" */ ],
    "staging": [0,7]
  },

  "responder": {                   // the deployed team's position (formerly executor)
    "pos":[3,4],
    "resource": 27,                // remaining; decrements 1 per committed move
    "resource_max": 44,
    "objective":[7,0],             // cell being headed to (current victim, or staging on egress)
    "path":[[3,4],[3,5]],          // planned route incl. current pos; [] if none
    "visited":[[0,7],[1,7]]
  },

  "victims": {
    "total": 2,
    "reached": 1,
    "remaining": [[7,0]],          // not yet reached
    "rescued":   [[0,1]]           // reached
  },

  "proposals": {                   // what each proposer wanted this tick; null before first tick
    "llm":"NORTH","net":"NORTH","astar":"EAST","chosen":"NORTH",
    "source":"llm","agreed":false,"net_confidence":0.87
  },

  "verdict": {                     // Safety Officer's ruling; null before first tick
    "vetoed": true,
    "rule": "no_hazard",           // in_bounds|no_hazard|air_reserve|escape_route|no_thrash|no_loop|ok
    "reason": "Advance NORTH enters collapse zone at (3,3)",
    "target_cell": [3,3]
  },

  "handoffs": [ {"tick":12,"from":"Safety Officer","to":"Incident Commander",
                 "task_id":"t2","payload":{"rule":"no_hazard"}} ],

  "log": [ {"tick":12,"clock":"T+00:12","agent":"Safety Officer","level":"veto",
            "text":"HALT: advance NORTH enters collapse zone at (3,3)"} ],
            // level: "info" | "veto" | "success" | "warn"

  "stats": {"moves_committed":11,"vetoes":2,"llm_calls":4,"replans":2,
            "rescued":1,"llm_mode":"live"}   // llm_mode: "live" | "mock"
}
```

## Safety Officer doctrine rules (pure logic, never an LLM call)

Which rules are active comes from the profile; thresholds come from `risk_tolerance`.

| `rule` | Vetoes when | Domain framing |
|---|---|---|
| `in_bounds` | target cell is outside the grid | leaving the operational area |
| `no_hazard` | target cell is `hazard` | entering collapse / active fire / hydraulic |
| `air_reserve` | remaining resource < (egress distance × reserve multiplier) → **orders RTB, status→aborted** | rule of thirds; conservative=2.0, standard=1.3, aggressive=1.05 |
| `escape_route` | the move severs the only safe path back to staging | LCES / never lose your way out |
| `no_thrash` | the move bounces straight back to the previous cell | oscillation guard |
| `no_loop` | the move re-enters a cell for the 3rd+ time | livelock guard (exempt for Operations' A\* route) |
| `ok` | approved | — |

## Two world models — why the veto is meaningful

Ground truth vs. the team's belief are deliberately separated: the **Safety Officer** checks
live ground truth; **Incident Commander / Operations / Rescue Team / policy net** reason over
the *last-known* map. A disruption is real immediately but not on the team's map — they advance
into what *was* safe, the Safety Officer catches it against live sensors, and **the halt is how
the team learns the hazard exists**. A veto also transfers routing authority to Operations' A\*
route for the next tick (`no_loop` transfers it permanently).

## Invariants (tests assert these)

1. A vetoed move never changes `responder.pos`.
2. Every veto appends a `Safety Officer → Incident Commander` handoff and a `level:"veto"` log.
3. `responder.resource` decrements only on a committed move.
4. `grid.cells` is always exactly 64 entries.
5. `tick` increases by exactly 1 per `/tick`, veto or not.
6. `status` → `complete` only when all victims reached **and** the team is back at staging.
7. `inject_disruption` never places a hazard that would strand the mission.
