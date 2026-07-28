# Prompt design — `brain.sv.jac`

The byLLM layer has exactly two reasoners. Neither has a function body: `by llm(...)`
replaces it, and every word the model sees comes from a `sem` statement. There is not a
single docstring in the LLM-visible surface — `sem` *is* the prompt.

```
commander_decompose(goal)      -> TaskPlan       # once per mission
executor_propose(situation)    -> MoveProposal   # once per tick
brain_mode()                   -> "live" | "mock"
```

## Where the prompt actually lives

byLLM builds the request from three places, and all three are `sem`:

| `sem` target | What it becomes |
|---|---|
| `sem fn = ...` | the system/task instruction |
| `sem fn.param = ...` | the description of each input |
| `sem Obj.field = ...` | a per-field description inside the JSON output schema |

The field-level `sem` is the highest-leverage of the three: it is attached to the
structured-output schema, so the constraint sits next to the slot the model is filling
rather than three paragraphs up in a wall of instructions. That is why the "exactly one of
NORTH, SOUTH, EAST, WEST" rule lives on `MoveProposal.move` and the "exactly one of
Commander, Executor, Verifier, Planner" rule lives on `PlannedTask.owner`, not in the
function-level text.

## The Commander

`_llm_decompose` gets a role, the board size, a hard count (2–4), and — critically — a
one-line job description for each of the four agents, because the model cannot pick a
correct `owner` unless it knows what a Verifier *does* here:

> Planner (computes a safe route), Executor (advances one cell per tick), Verifier
> (checks every proposed move against the grid safety rules), Commander (re-plans when a
> move is rejected).

Two negative constraints earn their place: **do not restate the goal as a task** (models
love to emit "Reach the survivor at (7,6)" as task 1, which is not a task, it is the
mission) and **do not invent agents outside the four listed** (the UI keys its panels off
those four exact strings — an invented "Scout" would render nowhere).

`PlannedTask.desc` is told it is "rendered live on a mission-control board" and capped at
a short imperative phrase. Telling the model where the string will be displayed is worth
more than any adjective about style.

## The Executor

This is the per-tick call, so it is the one that has to be short, cheap and boring.

The function-level `sem` spends its first sentence on **coordinate semantics**, because
this is the single thing a model gets wrong here:

> The origin (0,0) is the top-left cell, x is the column and y is the row, so NORTH
> decreases y, SOUTH increases y, EAST increases x and WEST decreases x.

Screen coordinates are y-down; the model's prior is that "north" means "y increases".
Stating the mapping explicitly removes an entire class of inverted-move failures.

`MoveProposal.rationale` is written as a **UI spec, not a reasoning instruction**:

> One short sentence, at most 100 characters, telling a human watching the mission-control
> screen why this step was chosen. Plain present-tense English, no step-by-step reasoning,
> no coordinate dumps, no markdown.

It ships with a worked example (`'Heading east closes the gap while the hazard wall stays
clear.'`) because one exemplar pins tone and length better than three adjectives. This is
deliberately **not** chain-of-thought: it is a caption. A model asked to "explain its
reasoning" writes 200 words of deliberation that overflows a log panel; a model asked to
write a caption for a screen writes a caption.

## Call parameters

`temperature=0.0` on both, so the same board yields the same move — a demo that replays
differently each run is not a demo. `max_tokens` is 400 for the Commander (once per
mission) and 150 for the Executor (every tick). `max_output_retries` is 2 for the
Commander and **1** for the Executor: a per-tick call must not sit through three corrective
round-trips, and one bad tick costs nothing because A\* is right there.

## Live / mock switch

Decided once, at import, by `_make_llm()`:

- `ANTHROPIC_API_KEY` present and non-blank → `Model(model_name="claude-sonnet-4-6")`
- absent, blank, or the `Model` constructor throws → `MockLLM(model_name="mockllm")`

`brain_mode()` does not read the env var. It does `isinstance(llm, MockLLM)` — it reports
the object that is *actually wired up*, so it structurally cannot claim "live" while
running mocked, including in the case where a key is present but the model failed to
construct.

Mock outputs are canned but not static. `MockLLM` **pops** from its `outputs` list, so an
exhausted list is an `IndexError` mid-demo; `_prime()` therefore refills the queue with
exactly one value immediately before every call, and the queue can never run dry. The
canned `MoveProposal` is derived from the situation text: it pulls the two coordinate pairs
out of the report, reads each neighbour's status, steps greedily along the larger axis,
skips any direction the report flags as `hazard` / out of bounds, skips directions the
report does not list at all (grid edge), and refuses to immediately reverse itself. On the
canonical seed map that walks (0,0) → (7,6) in 13 moves with zero vetoes — the A\* optimum
— so the offline demo looks like reasoning rather than like a stub.

## Failure behaviour

Both public functions are total. They never raise, and there is no case in which the
backend has to catch anything.

| What happens | What the caller gets |
|---|---|
| Network error, auth error, rate limit, timeout | `MoveProposal(move="", ...)` / `TaskPlan(tasks=[])` |
| Output that will not parse into the schema (after retries) | same |
| `move` is not one of the four directions (`"north-east"`, `"up-left"`, `""`) | `move=""` |
| Plan comes back with fewer than 2 usable tasks | `TaskPlan(tasks=[])` |
| Circuit breaker open (see below) | same, with no LLM call at all |

Everything that does come back is normalised before it leaves the module: `move` is
stripped to letters and upper-cased, with a small alias table (`N`, `UP`, `LEFT`, …) so a
near-miss is recovered rather than discarded; `rationale` is whitespace-collapsed, cut to
its first sentence and clamped to 100 characters; `owner` is snapped onto one of the four
exact agent strings (defaulting to `Planner`); task ids are **renumbered** `t1..tn` so they
are guaranteed unique even if the model repeats one; the plan is truncated at 4 tasks.

An empty `move` is the contract's own "invalid" signal — the backend already treats
anything that is not one of the four directions as unusable and falls through to the policy
net and then to A\*. So the failure path needs no special-casing on either side.

**Circuit breaker.** In live mode a dead network costs seconds per call (litellm retries
internally), which would stall a loop that ticks twice a second. After 3 consecutive
failures the reasoner is skipped outright and returns the invalid value immediately; every
10th skipped call lets one probe through, so a model that comes back is picked up again
without a restart. In mock mode nothing ever fails, so the breaker never arms.

## Verified

- `jac check brain.sv.jac` → clean, no errors, no warnings.
- Mock path end-to-end with no `ANTHROPIC_API_KEY`: `brain_mode() == "mock"`, a 4-task
  `TaskPlan` with valid owners, and a valid `MoveProposal` for every situation string
  `main.sv.jac::_situation` emits.
- Injected dispatch exception → `move=""`, no exception escapes; breaker arms on the 4th
  consecutive failure.
- Injected malformed output (`move="north-east!"`, 1-task plan) → `move=""`,
  `TaskPlan(tasks=[])`.
- With a dummy key set, `brain_mode()` reports `"live"` and a refused connection is
  swallowed into the same invalid result.
