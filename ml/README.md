# Mission Control — trained-model layer

Two small hand-written networks. No torch, no sklearn, no GPU: training is plain
numpy on system `python3` (3.9.6 / numpy 2.0.2), and **inference is pure Python**.

```
ml/gen_data.py        20k random grids -> A* labels -> ml/data/policy_data.npz
ml/train_policy.py    MLP 126->64->4, Adam. Writes ml/weights/policy.json
ml/distill_router.py  MLP 10->16->4,  Adam. Writes ml/weights/router.json
ml/_astar.py          local A*/BFS oracle, used ONLY by the ML pipeline
lib/policy.py         pure-python inference (predict / predict_route)
```

Reproduce end to end (~15s total):

```bash
python3 ml/gen_data.py && python3 ml/train_policy.py && python3 ml/distill_router.py
```

---

## The numpy split — the one rule that matters

| | runtime | numpy? |
|---|---|---|
| `ml/*.py` | system `python3` | **yes**, used freely |
| `lib/policy.py` | Jac's bundled CPython | **NO — hard constraint** |

Jac's bundled interpreter has no numpy. A numpy import in `lib/policy.py` takes
down the whole backend at server start. `lib/policy.py` imports **only `json`,
`math`, `os`**. Verified by importing it under a meta-path hook that raises on
any of numpy/scipy/pandas/sklearn/torch/jaclang and auditing `sys.modules`
afterwards: zero third-party modules, zero site-packages.

`ml/gen_data.py` **imports `encode_features` from `lib/policy.py`** rather than
reimplementing it. The encoder is defined exactly once, so training features and
inference features cannot drift apart.

---

## Model 1 — Executor policy net

Behavioural cloning of A* on an 8×8 grid. Predicts the next move from
`(cells, pos, goal)`.

### Feature encoding — `exec-v1`, 126 floats, all in [-1, 1]

| block | dims | contents |
|---|---|---|
| A | 25 | 5×5 egocentric passability window (`dy,dx ∈ -2..2`), 1.0 = passable |
| B | 25 | 5×5 egocentric goal indicator, 1.0 in the goal cell |
| C | 6 | `dx/7, dy/7, |dx|/7, |dy|/7, sign(dx), sign(dy)` |
| D | 4 | free-ray length N/S/E/W from the agent, ÷7 |
| E | 2 | absolute `px/7, py/7` (lets the net feel the border) |
| F | 64 | global passability map, row-major |

Out-of-bounds encodes as 0.0, identically to a hazard. A cell is passable iff
`kind != "hazard"` — so `start`, `target` and `free` are all passable.

Blocks A–E are egocentric (sample-efficient, translation-invariant); block F adds
the global obstacle layout, without which a purely local view cannot tell a
detour from a dead end. Most of the vector is zero, which the pure-Python forward
pass exploits by skipping zero inputs.

### Architecture

`126 → Linear → ReLU(64) → Linear → softmax(4)` over `[NORTH, SOUTH, EAST, WEST]`.
Adam (lr 3e-3, batch 256, 60 epochs, L2 1e-6), He init. ~8.4k parameters.
Best-test-epoch checkpoint is the one saved (epoch 54 of 60).

### Data

20,000 random grids (hazard density sampled uniformly in [0.05, 0.30], random
start/goal, unsolvable grids discarded), up to 3 agent positions sampled per grid
→ **59,197 samples**. 85/15 train/test split. Labels are near-balanced
(NORTH .225 / SOUTH .231 / EAST .276 / WEST .267).

### Labels, and why there are two accuracy numbers

**48.3% of states have more than one optimal first move.** A single label has to
pick one, so strict top-1 understates the net. `_astar.best_move` tie-breaks
deterministically — prefer the axis with the larger remaining `|delta|`, then the
direction that closes the gap — a pure function of `(dx, dy)`, so it is learnable
rather than arbitrary. Both numbers are reported:

| metric | held-out |
|---|---|
| **top-1 vs the A* label** (strict, target was ≥90%) | **96.17%** |
| **predicted move lies on *some* shortest path** | **97.63%** |

Measured on 8,879 held-out samples. Training curve (test top-1): 0.904 @ ep1 →
0.952 @ ep10 → 0.954 @ ep30 → **0.962 @ ep54** → 0.954 @ ep60. Mild overfit late
(train 0.979 vs test 0.954), which is why the best-epoch checkpoint is saved. The
full per-epoch curve is stored in `ml/weights/policy.json` under `"curve"`.

### Behaviour on the frozen seed map (CONTRACT.md)

Checked from 12 positions spanning all four answers, via the pure-Python path:

| | seed map | after hazard injected at (4,5) |
|---|---|---|
| matches the label oracle | 12/12 | 12/12 |
| move is on a shortest path | **12/12** | **12/12** |
| matches A*'s *heap* tie-break | 6/12 | 7/12 |

That last row is not an error and is the reason the check reports three columns.
`_astar.next_move` breaks ties via heap order and prefers SOUTH from (0,0);
the net was trained to prefer the larger-`|delta|` axis and says EAST. Both are
first steps on a 14-cell shortest path. Judging the net against a tie-break it
was never trained on would report a misleading 50%.

Greedy rollout, following the net alone with no A* and no verifier:

- seed map: reaches (7,6) in **13 moves — optimal**
- after injecting the demo hazard at (4,5): reaches (7,6) in **13 moves — optimal**
- 1,899 fresh random solvable grids: **90.3%** reach the goal, **89.8%** by a
  shortest path

Inference cost: **~0.29 ms/call** in pure Python.

### What this model is and is not

- It **is** a fast learned approximation of A*'s first move — a plausible second
  opinion for the proposal-vs-verify loop, and it is right about 96% of the time.
- It is **not** a planner and **not** a safety mechanism. ~10% of random grids it
  fails to solve on its own (dead ends, oscillation). It has no notion of energy,
  thrash, or bounds. The Verifier's rules remain the only thing preventing an
  illegal move — the net can and does propose moves into hazards.
- It is trained on *uniformly random* hazards. The seed map's structured wall is
  out-of-distribution; it happens to handle it, as measured above, but that is a
  spot check on one map, not a guarantee.

---

## Model 2 — Commander router

Maps a mission-situation vector to the agent that should own the next task.

### ⚠️ Distilled from a SYNTHETIC teacher, not from live LLM output

**The real byLLM decision logs do not exist yet.** `distill_router.py` reads
`ml/data/router_log.jsonl` when that file is present, and otherwise falls back to
a corpus generated by the documented rule-based teacher in `teacher()`.
**The committed `ml/weights/router.json` is currently distilled from that
synthetic rule.** `router.json` records this in its `"trained_from"` field.

So the accuracy below measures *how faithfully the net reproduces a rule we wrote
ourselves*. It is **not** evidence that the router matches real LLM routing, and
should not be described that way in the demo. To make it real, drop a JSONL of
`{"features": {...} or [...], "agent": "Planner"}` at `ml/data/router_log.jsonl`
and re-run — real logs are used automatically (≥200 valid rows required), no code
change needed.

### Features — 10 floats, order fixed by `lib.policy.ROUTER_FEATURE_NAMES`

`tick_norm`, `has_plan`, `last_vetoed`, `veto_rate`, `net_confidence`, `agreed`,
`energy_frac`, `dist_norm`, `blocked_frac`, `pending_frac`

`predict_route` accepts either a list in this order or a dict keyed by these
names (missing keys default to 0.0).

### The teacher rule — first match wins, order *is* the policy

1. `has_plan == 0` → **Planner** (no route exists)
2. `energy_frac < 0.15` → **Commander** (resource crisis outranks everything)
3. `last_vetoed and veto_rate > 0.25` → **Commander** (chronic failure: re-decompose)
4. `last_vetoed` → **Planner** (one-off obstacle: just replan)
5. `blocked_frac >= 0.75` → **Planner** (boxed in)
6. `not agreed or net_confidence < 0.5` → **Verifier** (disagreement: check first)
7. `pending_frac > 0.6 and tick_norm > 0.3` → **Commander** (backlog: re-prioritise)
8. otherwise → **Executor**

### Architecture and result

`10 → ReLU(16) → softmax(4)`, Adam (lr 5e-3, batch 128, 120 epochs), 12,000
synthetic samples, 80/20 split.

**Held-out accuracy: 98.08%** (2,400 test samples, best epoch 116).

Per-class recall: Commander 0.975 (n=526), Executor 0.917 (n=168),
Verifier 0.996 (n=484), Planner 0.986 (n=1222). Reported per class because the
sampler deliberately over-weights the interesting non-Executor corners, so
Executor is only ~7% of the corpus and a collapsed class could otherwise hide
behind a good average.

### What this model is and is not

- It **is** a working, wired-up router with a real interface, honest about being
  a placeholder, and swappable for real logs without touching code.
- It is **not** distilled from an LLM yet. 98% means "it learned our `if`-chain",
  which is close to tautological — a rule this simple could just be executed
  directly. Its value is the seam, not the intelligence.
- Do not present the router as evidence that the LLM's judgement was captured.

---

## Files and artifacts

| path | committed | notes |
|---|---|---|
| `ml/weights/policy.json` | yes | ~90 KB, includes metrics + full training curve |
| `ml/weights/router.json` | yes | ~10 KB, includes `trained_from` provenance |
| `ml/data/policy_data.npz` | no | 1.6 MB regenerated intermediate — safe to `.gitignore` |
| `ml/data/router_log.jsonl` | n/a | optional real-LLM input; absent today |

Both weight files carry `in_dim`, `hidden`, `classes` and are shape-validated on
load; `policy.json` also carries `feature_version: "exec-v1"`.

## Failure behaviour (backend-critical)

`lib/policy.py` is imported at server start and **never raises**. On any problem
`predict` returns `("", 0.0)` and `predict_route` returns `("", 0.0)`, letting the
backend fall through to the next proposer. Verified against: weights directory
absent, truncated JSON, non-JSON bytes, valid JSON with wrong shapes, valid JSON
with wrong class names, unreadable file (`chmod 000`), `cells=None`, wrong-length
`cells`, non-numeric positions, and garbage router input.

Weight lookup tries, in order: `$MISSIONCONTROL_WEIGHTS`, `<repo>/ml/weights/`,
`<module dir>/ml/weights/`, `<module dir>/weights/`, `<cwd>/ml/weights/` — so it
still resolves if Jac loads a copy of `policy.py` from somewhere other than
`repo/lib/`. `lib.policy.weights_status()` reports which files actually loaded and
from where.
