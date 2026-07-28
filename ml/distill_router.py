"""Distil the Commander's task-routing decision into a tiny classifier.

Maps a mission-situation feature vector -> which agent owns the next task
("Commander" | "Executor" | "Verifier" | "Planner").

*** HONEST STATUS ***
The real byLLM decision logs do not exist yet. This script reads
`ml/data/router_log.jsonl` if that file is present, and OTHERWISE falls back to
a SYNTHETIC corpus produced by the rule-based teacher in `teacher()` below.
The committed `ml/weights/router.json` is currently distilled from that
synthetic teacher, NOT from live LLM output. A high accuracy number here means
"the net reproduces the rule", not "the net routes well in the real world".
Re-point it at real logs by dropping a JSONL of
    {"features": {...} or [...], "agent": "Planner"}
at ml/data/router_log.jsonl and re-running -- no code change needed.

Run:  python3 ml/distill_router.py
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from lib.policy import ROUTER_FEATURE_NAMES, ROUTE_AGENTS  # noqa: E402

LOG = os.path.join(_HERE, "data", "router_log.jsonl")
OUT = os.path.join(_HERE, "weights", "router.json")

N_SYNTH = 12000
HIDDEN = 16
EPOCHS = 120
BATCH = 128
LR = 5e-3
TEST_FRAC = 0.2
SEED = 11

AGENT_IDX = {a: i for i, a in enumerate(ROUTE_AGENTS)}


# --------------------------------------------------------------------------
# The rule-based teacher. FIRST MATCHING RULE WINS -- order is the policy.
# Feature order is lib.policy.ROUTER_FEATURE_NAMES:
#   tick_norm, has_plan, last_vetoed, veto_rate, net_confidence,
#   agreed, energy_frac, dist_norm, blocked_frac, pending_frac
# --------------------------------------------------------------------------
def teacher(f):
    (tick_norm, has_plan, last_vetoed, veto_rate, net_conf,
     agreed, energy_frac, dist_norm, blocked_frac, pending_frac) = f

    # 1. No route at all -> somebody has to plan one.
    if has_plan < 0.5:
        return "Planner"
    # 2. Energy crisis outranks everything else: escalate, don't burn moves.
    if energy_frac < 0.15:
        return "Commander"
    # 3. Repeated vetoes = the plan itself is wrong -> escalate to re-decompose.
    if last_vetoed >= 0.5 and veto_rate > 0.25:
        return "Commander"
    # 4. A single veto = local obstacle -> replan the route.
    if last_vetoed >= 0.5:
        return "Planner"
    # 5. Boxed in -> replan before stepping.
    if blocked_frac >= 0.75:
        return "Planner"
    # 6. Proposers disagree, or the net is unsure -> check before committing.
    if agreed < 0.5 or net_conf < 0.5:
        return "Verifier"
    # 7. Backlog piling up while things are calm -> Commander re-prioritises.
    if pending_frac > 0.6 and tick_norm > 0.3:
        return "Commander"
    # 8. Otherwise just move.
    return "Executor"


def synth(rng, n):
    """Sample situations, biased toward the interesting (non-Executor) corners."""
    X = np.zeros((n, len(ROUTER_FEATURE_NAMES)))
    X[:, 0] = rng.random(n)                                    # tick_norm
    X[:, 1] = (rng.random(n) > 0.25).astype(float)             # has_plan
    X[:, 2] = (rng.random(n) > 0.72).astype(float)             # last_vetoed
    X[:, 3] = rng.random(n) * 0.6                              # veto_rate
    X[:, 4] = rng.random(n)                                    # net_confidence
    X[:, 5] = (rng.random(n) > 0.35).astype(float)             # agreed
    X[:, 6] = rng.random(n)                                    # energy_frac
    X[:, 7] = rng.random(n)                                    # dist_norm
    X[:, 8] = rng.integers(0, 5, n) / 4.0                      # blocked_frac
    X[:, 9] = rng.random(n)                                    # pending_frac
    y = np.array([AGENT_IDX[teacher(row)] for row in X], dtype=np.int64)
    return X, y


def load_log():
    """Real LLM logs, if anyone has dropped them in. Returns None if absent."""
    if not os.path.exists(LOG):
        return None
    X, y = [], []
    with open(LOG, "r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
                ft = rec["features"]
                if isinstance(ft, dict):
                    ft = [float(ft.get(k, 0.0)) for k in ROUTER_FEATURE_NAMES]
                else:
                    ft = [float(v) for v in ft]
                if len(ft) != len(ROUTER_FEATURE_NAMES):
                    continue
                ag = rec["agent"]
                if ag not in AGENT_IDX:
                    continue
                X.append(ft)
                y.append(AGENT_IDX[ag])
            except Exception:
                continue
    if len(y) < 200:
        return None
    return np.array(X), np.array(y, dtype=np.int64)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(SEED)

    real = load_log()
    if real is not None:
        X, y = real
        source = "ml/data/router_log.jsonl (REAL logged decisions)"
    else:
        X, y = synth(rng, N_SYNTH)
        source = "SYNTHETIC rule-based teacher (no LLM logs present)"
    print("training source: %s" % source)
    print("samples: %d, features: %d" % (len(y), X.shape[1]))
    counts = np.bincount(y, minlength=len(ROUTE_AGENTS))
    print("label balance  : " + ", ".join(
        "%s=%.3f" % (a, counts[i] / len(y)) for i, a in enumerate(ROUTE_AGENTS)))

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_test = int(len(y) * TEST_FRAC)
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]

    in_dim, n_out = X.shape[1], len(ROUTE_AGENTS)
    p = {
        "W1": rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, n_out)),
        "b2": np.zeros(n_out),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1c, b2c, eps, step = 0.9, 0.999, 1e-8, 0
    onehot = np.eye(n_out)[ytr]

    def fwd(par, xb):
        h = xb @ par["W1"] + par["b1"]
        a = np.maximum(h, 0.0)
        return h, a, a @ par["W2"] + par["b2"]

    def acc(par, xb, yb):
        return float((fwd(par, xb)[2].argmax(axis=1) == yb).mean())

    best = None
    curve = []
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            xb, tb = Xtr[idx], onehot[idx]
            h, a, z = fwd(p, xb)
            dz = (softmax(z) - tb) / len(idx)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                g = {"W2": a.T @ dz, "b2": dz.sum(axis=0)}
                dh = (dz @ p["W2"].T) * (h > 0)
                g["W1"] = xb.T @ dh
                g["b1"] = dh.sum(axis=0)
            step += 1
            for k in p:
                m[k] = b1c * m[k] + (1 - b1c) * g[k]
                v[k] = b2c * v[k] + (1 - b2c) * g[k] ** 2
                p[k] -= LR * (m[k] / (1 - b1c ** step)) / (
                    np.sqrt(v[k] / (1 - b2c ** step)) + eps)

        te = acc(p, Xte, yte)
        curve.append({"epoch": ep, "train_acc": round(acc(p, Xtr, ytr), 4),
                      "test_acc": round(te, 4)})
        if best is None or te > best[0]:
            best = (te, {k: val.copy() for k, val in p.items()}, ep)
        if ep % 20 == 0 or ep == 1:
            print("epoch %3d  train %.4f  test %.4f"
                  % (ep, curve[-1]["train_acc"], te))

    te_acc, bp, bep = best
    print("\n" + "=" * 62)
    print("ROUTER HELD-OUT ACCURACY : %.4f  (%.2f%%)  [best epoch %d]"
          % (te_acc, 100 * te_acc, bep))
    print("=" * 62)
    if real is None:
        print("This measures fidelity to the SYNTHETIC teacher rule only.")
        print("It is NOT evidence the router matches real LLM routing.")

    # per-class recall, so a collapsed class can't hide behind the average
    pred = (np.maximum(Xte @ bp["W1"] + bp["b1"], 0) @ bp["W2"] + bp["b2"]).argmax(axis=1)
    print("\nper-class held-out recall:")
    for i, a in enumerate(ROUTE_AGENTS):
        sel = yte == i
        r = float((pred[sel] == i).mean()) if sel.sum() else float("nan")
        print("  %-10s n=%-5d recall=%.4f" % (a, int(sel.sum()), r))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "kind": "commander_router",
            "trained_from": "real_llm_logs" if real else "synthetic_rule_teacher",
            "feature_names": ROUTER_FEATURE_NAMES,
            "in_dim": in_dim,
            "hidden": HIDDEN,
            "classes": ROUTE_AGENTS,
            "W1": np.round(bp["W1"], 6).tolist(),
            "b1": np.round(bp["b1"], 6).tolist(),
            "W2": np.round(bp["W2"], 6).tolist(),
            "b2": np.round(bp["b2"], 6).tolist(),
            "metrics": {"test_acc": round(te_acc, 4),
                        "n_train": int(len(ytr)), "n_test": int(len(yte)),
                        "best_epoch": bep},
            "curve": curve,
        }, fh)
    print("\nwrote %s (%.0f KB)" % (OUT, os.path.getsize(OUT) / 1e3))

    # smoke-test through the pure-python inference path
    import importlib
    import lib.policy as P
    importlib.reload(P)
    print("\nspot checks via lib.policy.predict_route:")
    cases = {
        "no plan yet": {"has_plan": 0, "net_confidence": 0.9, "agreed": 1,
                        "energy_frac": 0.9},
        "single veto": {"has_plan": 1, "last_vetoed": 1, "veto_rate": 0.1,
                        "net_confidence": 0.9, "agreed": 1, "energy_frac": 0.8},
        "chronic vetoes": {"has_plan": 1, "last_vetoed": 1, "veto_rate": 0.5,
                           "net_confidence": 0.9, "agreed": 1, "energy_frac": 0.8},
        "energy crisis": {"has_plan": 1, "net_confidence": 0.9, "agreed": 1,
                          "energy_frac": 0.05},
        "proposers disagree": {"has_plan": 1, "net_confidence": 0.9, "agreed": 0,
                               "energy_frac": 0.8},
        "all calm": {"has_plan": 1, "net_confidence": 0.95, "agreed": 1,
                     "energy_frac": 0.8, "dist_norm": 0.4},
    }
    ok = 0
    for name, feats in cases.items():
        vec = [float(feats.get(k, 0.0)) for k in ROUTER_FEATURE_NAMES]
        want = teacher(vec)
        got, conf = P.predict_route(feats)
        ok += got == want
        print("  %-20s -> %-10s (%.2f)  teacher=%-10s %s"
              % (name, got, conf, want, "ok" if got == want else "MISMATCH"))
    print("  spot checks passed: %d/%d" % (ok, len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
