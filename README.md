# Mission Control — SAR Operations Simulator

**A search-and-rescue operations simulator run by a team of specialized AI agents — not one
chatbot. A department picks a rescue domain, configures the operation, and rehearses it under
the Incident Command System, with a Safety Officer agent that enforces doctrine and can halt
the team before an unsafe move commits.**

Built on [Jac](https://www.jaseci.org/)'s object-spatial model: the sector is a graph of cells,
the agents are walkers, and coordination flows through shared graph state.

---

## Why this exists

Emergency-response departments rehearse operations constantly — tabletop exercises, pre-plans,
after-action reviews. Mission Control is a **mission-rehearsal simulator**: choose a domain
(urban collapse, wildland fire, swiftwater), set the parameters of *your* operation, and watch
an agentic Incident Command team plan and execute a search under real doctrine. The point isn't
a clever demo — it's a tool a department could specialize to model the kind of rescue they
actually run, and to see where doctrine forces a halt or a withdrawal.

## The team is an ICS org chart, not four copies of one bot

| ICS role | Job | How it decides |
|---|---|---|
| **Incident Commander** | Decompose the objective, re-task after a halt | `by llm()` decomposition (Groq/Anthropic) |
| **Operations** | Plan and re-plan the search route | classic **A\*** pathfinding |
| **Rescue Team** | Propose the next one-cell advance | `by llm()` + a **trained policy net** + A\* fallback |
| **Safety Officer** | Clear or **HALT** every advance against doctrine | pure rules — **never an LLM call** |

Three properties make this genuinely multi-agent: heterogeneous roles, typed handoffs between
them (rendered as an incident log + roster), and a guardian with real blocking power.

## Three proposers, one arbiter

Each tick the Rescue Team's advance is chosen from three independent proposers, then the Safety
Officer rules on it:

```
llm_move  = groq_propose(situation)   # a real LLM advance
net_move  = policy_net(state)         # trained from scratch on A* demos (96% agreement)
astar     = operations_route()        # deterministic floor

chosen  = first valid of (llm, net, astar)
verdict = safety_officer(chosen)      # PURE DOCTRINE — cannot flake
if verdict is HALT:      hold; hand off Safety Officer -> IC; re-task Operations; reroute
if verdict is WITHDRAW:  Return-To-Base; operation aborts safely (crew out in time)
```

## Safety Officer doctrine (deterministic, no LLM)

Which rules are active comes from the domain profile; thresholds come from the risk tolerance.

| Rule | Halts / withdraws when | Real doctrine |
|---|---|---|
| `no_hazard` | advance enters collapse / active fire / hydraulic | don't walk into the lethal cell |
| `escape_route` | advance severs the only safe path back to staging → **withdraw** | LCES — never lose your way out |
| `air_reserve` | can't advance and still egress with reserve in hand → **withdraw** | rule of thirds (SCBA air management) |
| `in_bounds` · `no_thrash` · `no_loop` | leaves the area / oscillates / livelocks | operational sanity |

`air_reserve` is where the risk tolerance bites, and it produces a genuinely useful result:

| Risk | Reserve | Typical outcome on a tight budget |
|---|---|---|
| Aggressive | 1.05× | pushes to near-empty, reaches more victims, higher exposure |
| Standard | 1.3× | balanced; turns back with a working margin |
| Conservative | 2.0× | Return-To-Base earliest, most air left, fewest victims reached |

A department can see, before anyone deploys, how their air-management doctrine trades victims
reached against crew safety on *their* scenario.

## Why the HALT is meaningful (the hard part)

Ground truth and the team's belief are deliberately separated. The **Safety Officer** checks
live ground truth; the **IC / Operations / Rescue Team / policy net** reason over the
*last-known* map. A disruption (a secondary collapse, a fire spread, rising water) is real
immediately but **not on the team's map yet** — so they advance into what *was* safe, the
Safety Officer catches it against live sensors, and **the HALT is how the team learns the
hazard exists**. The blocked cell then graduates onto the known map and Operations reroutes.
Without this split the planners would silently avoid every new hazard and the Safety Officer
would have nothing to catch.

A HALT also transfers routing authority from the Rescue Team's proposers to Operations' A\*
route for the next tick (a livelock transfers it for good) — so a confidently-wrong proposer
can't be halted on the same move forever.

---

## Run it

Requires only the Jac binary — it bundles its own CPython, so your system Python is irrelevant.

```bash
curl -fsSL https://raw.githubusercontent.com/jaseci-labs/jaseci/main/scripts/install.sh | bash
```

Pinned to **jac 0.34.7**. Then from the repo root:

```bash
export PATH="$HOME/.local/bin:$PATH"
jac start main.sv.jac --no-client --port 8800
```

The server takes ~20–25s to become ready. In a second terminal:

```bash
cd web && python3 -m http.server 8901
```

Open **http://localhost:8901/index.html**, pick a profile, set the tunables, and launch.

> **Port 8800, not 8000.** If every endpoint 404s, another app owns the port.
> `jac run` serves the same thing but does not accept `--port`.

### With a live LLM (Groq — fast, ideal for the per-tick loop)

```bash
export GROQ_API_KEY=gsk_...
```

Then start the server in that shell. The on-screen badge reads **LIVE** vs **MOCK**, driven by
an `isinstance` check on the model object actually wired up — it can't claim live while mocked.
`ANTHROPIC_API_KEY` (`claude-sonnet-4-6`) also works; Groq wins if both are set. With no key at
all the whole system runs on byLLM's `MockLLM`, fully offline.

### Drive it

Start/Pause auto-tick · Step · Reset · **Inject Disruption** (a domain hazard on the cell the
team is about to enter — the Safety Officer HALT is guaranteed and never strands the operation)
· New mission.

### If everything is on fire

| Fallback | How | Needs |
|---|---|---|
| Mock reasoner | don't set a key | no network, no key |
| Replay | `index.html?mode=replay` | **no backend** — a full recorded run |
| Fixture | automatic when :8800 is unreachable | nothing |

---

## Configure your operation

`POST /function/list_profiles` exposes the tunables; the setup screen builds controls from it.

- **Profile** — urban US&R / wildland fire / swiftwater. Sets terrain language, the resource
  being managed, the active doctrine, and how a disruption reads.
- **Team size** — crew extends the operational window (rotation).
- **Resource budget** — the operational window; must cover reaching victims *and* egress.
- **Hazard density** — how contested the sector is.
- **Victims** — 1–3 to locate and reach.
- **Risk tolerance** — the Safety Officer's air-reserve doctrine (see the table above).

Scenarios are deterministic in `(profile, tunables, seed)`, so a rehearsal replays identically
and a department can share a scenario by its seed.

## Layout

```
main.sv.jac        endpoints (list_profiles, configure_mission, tick, inject_disruption) + tick loop
world.sv.jac       graph schema, multi-victim Mission node, scenario materialisation
rules.sv.jac       the Safety Officer — pure doctrine, zero LLM imports
brain.sv.jac       byLLM reasoners + ICS prompts + Groq/Anthropic/mock switch
lib/profiles.py    the 3 domain profiles + deterministic scenario generator (pure stdlib)
lib/astar.py       A* pathfinding
lib/policy.py      policy-net inference — PURE PYTHON, no numpy (runs in Jac's runtime)
ml/                training: data gen, policy net, router distillation
web/               the two-screen UI (mission setup + operations dashboard)
CONTRACT.md        the frozen v2 wire format
```

## The trained model

The Rescue Team's policy net is a from-scratch numpy MLP (126→64→4) trained by behavioral
cloning on A\* demonstrations: **96% held-out agreement** with A\*'s optimal first move. It is a
*proposer*, not a safety mechanism — it has no notion of hazards, air, or doctrine, and can
propose an unsafe move. The Safety Officer's rules are the only thing that prevents one.
(A second model distils a routing classifier from a documented rule-based teacher; `ml/README.md`
is explicit that it learned the if-chain, not live LLM judgment.)

## Verified

- Every domain × disruption timing produces a Safety Officer HALT with the team's position held,
  and the operation still reaches a safe terminal state.
- Conservative / standard / aggressive risk produce visibly different Return-To-Base outcomes.
- Live Groq completes a mission in ~7s (0.2s/tick); the offline mock completes with no key.
- `lib/policy.py` confirmed numpy-free; scenarios deterministic and always solvable.

## Jac features used

Object-Spatial Programming (`node`/`edge`/`walker`, typed `+>:Adj():+>` connections, filtered
traversals), **`by llm()`** for agent reasoning, **`sem`** semstrings for prompt wiring,
**`MockLLM`** for offline runs, `def:pub` auto-generated REST endpoints, `import`ed pure-Python
`lib/*`, and automatic graph persistence.
