# Mission Control

**A team of specialized AI agents — not one chatbot — that plan and run a rescue mission
together on a live grid, with a safety agent that vetoes unsafe moves before they commit.**

Built on [Jac](https://www.jaseci.org/)'s object-spatial model: grid cells are graph nodes,
agents are walkers, and coordination happens through shared graph state.

---

## The 30-second version

A mission comes in: *locate and reach the survivor, avoid the hazards*. A **Commander**
decomposes it and delegates. An **Executor** proposes one move per tick. A **Verifier**
checks every proposed action against hard rules and **can block it**. A **Planner** holds
the A\* route. No human drives it.

Three properties make this genuinely multi-agent rather than one model in a loop:

1. **Heterogeneous roles** — different agents, different jobs, different tools.
2. **Structured handoffs** — typed messages between agents, rendered as lit-up edges.
3. **A guardian with real power** — the Verifier can veto, and the veto changes what happens next.

## Three proposers, one arbiter

Each tick, three independent systems propose a move:

| Proposer | What it is |
|---|---|
| **LLM** | A real `by llm()` call (Anthropic `claude-sonnet-4-6`), prompted via `sem` semstrings |
| **Policy net** | A neural net we trained from scratch by behavioral cloning on A\* (96.2% held-out) |
| **A\*** | Deterministic ground truth — always available as a floor |

The first valid proposal wins. Then the **Verifier** — pure rules, **never an LLM call** —
rules on it. That is deliberate: the veto is the climax of the demo and it must not flake.

```
llm_move  = byllm_propose(situation)     # real LLM decision
net_move  = policy_net(state)            # trained MLP
safe_move = astar(state)                 # deterministic floor

chosen  = first valid of (llm, net, astar)
verdict = verifier_rules(chosen)         # PURE RULES — cannot flake
if verdict.vetoed:
    hold position; handoff Verifier -> Commander; re-delegate to Planner; replan
```

## Why the veto is meaningful (the part that took the longest to get right)

Ground truth and the team's belief are **deliberately different**:

- The **Verifier** checks against live ground truth. It is the safety authority.
- **Commander / Planner / Executor / policy net** reason over the *last-known* map.

When a hazard is injected it is real immediately, but it is **not on the planners' map yet**.
They propose a move that *was* safe; the Verifier catches it against live sensor truth; and
**the veto is how the team learns the hazard exists**. The cell then graduates onto the known
map and the Planner routes around it.

Without this split the proposers would silently avoid every new hazard and the Verifier would
have nothing to catch. We found that out the hard way — the first integration produced a
system that worked perfectly and demonstrated nothing.

**Veto → re-delegate.** A veto also transfers routing authority from the Executor's proposers
to the Planner's deterministic route for the next tick. That is what stops a confidently-wrong
proposer from being vetoed on the same move forever.

## Verifier rules (deterministic, no LLM)

| Rule | Vetoes when |
|---|---|
| `in_bounds` | target cell is outside the 8×8 grid |
| `no_hazard` | target cell is a hazard |
| `energy_budget` | energy exhausted |
| `no_thrash` | the move bounces straight back to the previous cell |
| `no_loop` | the move re-enters a cell for the 3rd+ time (livelock guard) |

`no_loop` exists because we watched a proposer cycle (7,4)→(6,4)→(6,3)→(7,3) until the
battery died. `no_thrash` only catches a two-cell bounce; a four-cell ring slips straight past it.

---

## Run it

**Requires nothing but the Jac binary** — it bundles its own CPython, so your system Python
version is irrelevant.

```bash
curl -fsSL https://raw.githubusercontent.com/jaseci-labs/jaseci/main/scripts/install.sh | bash
```

Pinned to **jac 0.34.7**. Then, from the repo root:

```bash
export PATH="$HOME/.local/bin:$PATH"
jac start main.sv.jac --no-client --port 8800
```

The server takes ~20-25s to become ready. In a second terminal:

```bash
cd web && python3 -m http.server 8901
```

Open **http://localhost:8901/index.html**.

> **Port 8800, not 8000.** Port 8000 is commonly already taken; if the API 404s on every
> endpoint, something else owns the port. `jac run` serves the same thing but does **not**
> accept `--port`.

### With a real LLM

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

That is the only switch. With it set, `brain_mode()` reports `live` and the on-screen badge
reads **LIVE LLM**; without it the system runs on byLLM's `MockLLM` and the badge reads
**MOCK**. The badge is driven by an `isinstance` check on the model object actually wired up,
so it cannot claim "live" while running mocked.

### Drive it

| Control | Does |
|---|---|
| **START / SPACE** | auto-tick ~2×/sec |
| **STEP / N** | one tick |
| **RESET / R** | back to the seeded start |
| **INJECT HAZARD AHEAD / H** | **the disruption** |

The disruption drops a hazard on the cell the Executor is *actually about to enter*, rather
than a fixed coordinate — the three proposers don't always agree on the route, so a hardcoded
cell can miss and the veto never fires. The backend also refuses any cell that would strand
the mission.

### If everything is on fire

Three independent fallbacks, all verified:

| Layer | How | Needs |
|---|---|---|
| **Mock LLM** | just don't set the API key | no network, no key |
| **Replay** | `index.html?mode=replay` | **no backend at all** — 19 frames recorded off the real server |
| **Fixture** | automatic when :8800 is unreachable | nothing |

---

## Layout

```
main.sv.jac        the four endpoints + the tick loop
world.sv.jac       graph schema (Cell/Adj lattice, Mission, LogEntry, Handoff) + seed
rules.sv.jac       the Verifier. pure rules, zero LLM imports
brain.sv.jac       byLLM functions + sem prompts + live/mock switch
lib/astar.py       A* pathfinding (stdlib only)
lib/policy.py      policy-net inference — PURE PYTHON, no numpy (runs inside Jac's runtime)
ml/                training: data gen, policy net, router distillation
web/               single-page UI: board, agent org panel, reasoning log
CONTRACT.md        the frozen wire format — the seam every workstream coded against
docs/PROMPTS.md    prompt design writeup
```

`lib/policy.py` is numpy-free by hard requirement: training runs on system Python with numpy,
but inference runs inside Jac's bundled interpreter where numpy isn't installed.

## The trained models

**Executor policy net** — 126 → ReLU(64) → softmax(4), Adam, 59k samples from 20k random grids.
**96.17% held-out top-1** agreement with A\*; **97.63%** of its moves lie on *some* shortest path.
Both numbers are reported because 48.3% of states have more than one optimal first move, so
strict top-1 structurally understates it.

**Commander router** — 10 → ReLU(16) → softmax(4). 98.08% held-out.
**Read that number carefully:** no LLM decision logs existed yet, so it is distilled from a
documented rule-based teacher. 98% means "the net learned our if-chain" — it is *not* evidence
that LLM judgement was captured. `router.json` records this as `"trained_from":
"synthetic_rule_teacher"`. Drop a JSONL at `ml/data/router_log.jsonl` to retrain from real logs
with no code change.

The policy net is **not** a safety mechanism. It has no notion of energy, bounds, or thrash,
and it can propose moves into hazards. The Verifier's rules are the only thing preventing an
illegal move.

## Verified

- Disruption at **every tick from 1–11** produces a veto with the executor's position held,
  and the mission still completes. At tick 12+ the executor is adjacent to the survivor and
  the disruption is correctly refused — the survivor's cell must never be blocked.
- Two consecutive runs produce **byte-identical** transcripts.
- Grid is always 64 cells; corner/edge/interior node degrees are 2/3/4.
- Replay runs the full veto sequence with the backend process killed.

## Jac features used

Object-Spatial Programming (`node` / `edge` / `walker`, typed `+>:Adj():+>` connections,
filtered traversals `[root -->[?:Cell, x == 3]]`), **`by llm()`** for agent reasoning,
**`sem`** semstrings for prompt wiring, **`MockLLM`** for offline runs, `def:pub`
auto-generated REST endpoints, and automatic graph persistence.
