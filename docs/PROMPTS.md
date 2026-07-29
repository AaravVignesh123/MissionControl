# Prompt design — the byLLM reasoning layer

All prompts live in `brain.sv.jac` as `sem` statements (never docstrings) — the `sem` text is
exactly what the model sees. Two reasoners sit behind total, exception-proof wrappers:
`commander_decompose(goal) -> TaskPlan` and `executor_propose(situation) -> MoveProposal`.

## Model selection

Chosen once at import, honest by construction: `GROQ_API_KEY` → `groq/llama-3.3-70b-versatile`,
else `ANTHROPIC_API_KEY` → `claude-sonnet-4-6`, else byLLM `MockLLM`. `brain_mode()` reports the
object actually wired up (an `isinstance` check), so it cannot claim "live" while mocked. Groq is
preferred because its sub-second latency suits a per-tick loop. Both reasoners run at
`temperature=0.0` — a rehearsal that replays differently isn't a rehearsal.

## Three tiers of `sem`

1. **Function-level** — the role and its hard constraints. The **Incident Commander** prompt names
   the four ICS roles and what each does (a model can't assign an `owner` correctly without
   knowing a Safety Officer's job), and forbids restating the objective or inventing roles. The
   **Rescue Team** prompt spends its first sentence on coordinate semantics — origin top-left,
   y increases south, so NORTH decreases y — because the model's default prior is the opposite,
   and that one sentence removes a whole class of inverted advances.

2. **Field-level** — the highest-leverage tier, since it rides inside the output schema next to the
   slot being filled: "exactly one of NORTH/SOUTH/EAST/WEST" on `MoveProposal.move`; "exactly one
   of Incident Commander / Operations / Rescue Team / Safety Officer" on `PlannedTask.owner`.

3. **`rationale` is written as a UI spec, not a reasoning instruction**: ≤100 chars, one clipped
   field-radio sentence, no step-by-step, no coordinate dumps. Asking a model to "explain its
   reasoning" overflows the incident log; asking for a radio call gets a radio call.

## Robustness over cleverness

Every LLM output is schema-validated downstream and falls back to the policy net, then A*. So the
job here is: make the happy path good and the failure path harmless.

- Neither public function ever raises. Network/auth/parse failure, an out-of-set `move`, or a plan
  with <2 usable tasks returns a *clearly invalid* value (empty move / empty task list) and the
  backend falls through to the next proposer.
- Survivable near-misses are repaired, not discarded: `move` is stripped, upper-cased, and
  alias-mapped (`N`, `UP`, `LEFT`, …); `rationale` is collapsed to one clamped sentence; `owner`
  snaps to one of the four ICS roles; task ids are renumbered `t1..tn`; plans truncate at 4.
- A **circuit breaker** skips the reasoner after repeated failures (with periodic probes) so a dead
  network can't stall the ~1.5 Hz tick loop; in mock mode nothing fails, so it never arms.
- The offline **mock** reasoner is not a stub: it parses the situation report, steps greedily
  toward the objective, honours `BLOCKED` neighbours, avoids reversals, and keeps a short trail
  memory so it doesn't ring around a wall — so the no-key demo still looks like reasoning.

## Situation report format

The Rescue Team sees, from the *last-known* map (undiscovered hazards read as clear — that is what
makes the Safety Officer's HALT meaningful): current cell, objective cell (a victim during search,
staging during egress), remaining resource, and each neighbour tagged `clear` or
`BLOCKED (<hazard noun>)`. `BLOCKED` is a stable token every proposer keys off; the domain noun
(collapse zone / active fire / hydraulic) rides along for the LLM's benefit.
