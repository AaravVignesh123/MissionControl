"""Mission Control — trained-model inference layer.

PURE PYTHON. NO numpy. NO third-party imports. Standard library only.

This module runs inside Jac's bundled CPython where numpy is NOT installed, so
importing numpy here would break the whole backend. Everything below is plain
loops and lists on purpose.

It exposes two entry points (see CONTRACT.md):

    predict(cells, pos, goal)   -> (move: str, confidence: float)
    predict_route(features)     -> (agent: str, confidence: float)

Both degrade to a neutral answer -- ("", 0.0) / ("Commander", 0.0) -- if the
JSON weight files are missing or malformed. They never raise; the backend
imports this module at server start and must not die because of it.

This file is ALSO the canonical definition of the executor feature encoding:
``ml/gen_data.py`` imports ``encode_features`` from here so training and
inference can never drift apart.
"""

import json
import math
import os

# --------------------------------------------------------------------------
# Grid conventions (CONTRACT.md)
#   8x8, origin (0,0) top-left, x = column (east +), y = row (south +)
#   NORTH = y-1, SOUTH = y+1, EAST = x+1, WEST = x-1
#   cells is 64 row-major strings; index = y*8 + x
# --------------------------------------------------------------------------
GRID_W = 8
GRID_H = 8

MOVES = ["NORTH", "SOUTH", "EAST", "WEST"]
DELTAS = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

ROUTE_AGENTS = ["Commander", "Executor", "Verifier", "Planner"]

# Order of the router feature vector. Kept here so callers can pass a dict.
ROUTER_FEATURE_NAMES = [
    "tick_norm",
    "has_plan",
    "last_vetoed",
    "veto_rate",
    "net_confidence",
    "agreed",
    "energy_frac",
    "dist_norm",
    "blocked_frac",
    "pending_frac",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _candidates(name):
    """Where the weights might be.

    Jac's loader may pick this file up from a copy next to the .jac entry point
    rather than from repo/lib/, so don't bet on a single hard-coded path. The
    env var wins if it is set.
    """
    out = []
    env = os.environ.get("MISSIONCONTROL_WEIGHTS")
    if env:
        out.append(os.path.join(env, name))
    out.append(os.path.join(_REPO, "ml", "weights", name))   # repo/lib/policy.py
    out.append(os.path.join(_HERE, "ml", "weights", name))   # policy.py at repo root
    out.append(os.path.join(_HERE, "weights", name))
    try:
        out.append(os.path.join(os.getcwd(), "ml", "weights", name))
    except Exception:
        pass
    return out

FEATURE_VERSION = "exec-v1"
FEATURE_DIM = 126


# --------------------------------------------------------------------------
# Feature encoding -- executor policy net
# --------------------------------------------------------------------------
# 126 floats, all in [-1, 1]:
#   A) 25  5x5 egocentric passability window   (dy,dx in -2..2; 1.0 = passable)
#   B) 25  5x5 egocentric goal indicator       (1.0 in the goal cell, else 0)
#   C)  6  goal direction: dx/7, dy/7, |dx|/7, |dy|/7, sign(dx), sign(dy)
#   D)  4  free-ray length N,S,E,W from the agent, normalised by 7
#   E)  2  absolute agent position px/7, py/7  (lets the net feel the border)
#   F)  64 global passability map, row-major   (1.0 = passable)
# Out-of-bounds always encodes as 0.0 (impassable), same as a hazard.


def _passable(cells, x, y):
    if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
        return False
    try:
        return cells[y * GRID_W + x] != "hazard"
    except (IndexError, TypeError):
        return False


def _sign(v):
    if v > 0:
        return 1.0
    if v < 0:
        return -1.0
    return 0.0


def encode_features(cells, pos, goal):
    """(cells, pos, goal) -> list[float] of length FEATURE_DIM.

    Identical at train time and inference time -- ml/gen_data.py imports this
    exact function.
    """
    px, py = int(pos[0]), int(pos[1])
    gx, gy = int(goal[0]), int(goal[1])
    f = []

    # A) egocentric passability
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            f.append(1.0 if _passable(cells, px + dx, py + dy) else 0.0)

    # B) egocentric goal indicator
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            f.append(1.0 if (px + dx == gx and py + dy == gy) else 0.0)

    # C) goal direction
    ddx = gx - px
    ddy = gy - py
    f.append(ddx / 7.0)
    f.append(ddy / 7.0)
    f.append(abs(ddx) / 7.0)
    f.append(abs(ddy) / 7.0)
    f.append(_sign(ddx))
    f.append(_sign(ddy))

    # D) free-ray length in each move direction
    for mv in MOVES:
        sx, sy = DELTAS[mv]
        n = 0
        cx, cy = px + sx, py + sy
        while _passable(cells, cx, cy):
            n += 1
            cx += sx
            cy += sy
        f.append(n / 7.0)

    # E) absolute position
    f.append(px / 7.0)
    f.append(py / 7.0)

    # F) global passability map
    for y in range(GRID_H):
        for x in range(GRID_W):
            f.append(1.0 if _passable(cells, x, y) else 0.0)

    return f


# --------------------------------------------------------------------------
# Tiny pure-python MLP: x -> ReLU(x@W1 + b1) -> @W2 + b2 -> softmax
# W1 is stored row-major by INPUT (in_dim rows x hidden cols) so we can skip
# zero inputs -- most of the 126 features are zero, so this is a real win.
# --------------------------------------------------------------------------
def _load_model(name, expect_classes=None):
    for path in _candidates(name):
        m = _try_load(path, expect_classes)
        if m is not None:
            return m
    return None


def _try_load(path, expect_classes=None):
    try:
        with open(path, "r") as fh:
            m = json.load(fh)
        w1 = m["W1"]
        b1 = m["b1"]
        w2 = m["W2"]
        b2 = m["b2"]
        classes = m["classes"]
        in_dim = int(m["in_dim"])
        hidden = int(m["hidden"])
        if len(w1) != in_dim or len(b1) != hidden:
            return None
        if len(w2) != hidden or len(b2) != len(classes):
            return None
        if len(w1[0]) != hidden or len(w2[0]) != len(classes):
            return None
        if expect_classes is not None and list(classes) != list(expect_classes):
            return None
        return {
            "W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2,
            "classes": list(classes),
            "in_dim": in_dim,
            "hidden": hidden,
            "path": path,
        }
    except Exception:
        return None


def _forward(model, x):
    """Returns (best_index, confidence) or None."""
    hidden = model["hidden"]
    b1 = model["b1"]
    w1 = model["W1"]
    h = list(b1)
    for i, xi in enumerate(x):
        if xi == 0.0:
            continue
        row = w1[i]
        for j in range(hidden):
            h[j] += xi * row[j]
    for j in range(hidden):
        if h[j] < 0.0:
            h[j] = 0.0

    n_out = len(model["b2"])
    o = list(model["b2"])
    w2 = model["W2"]
    for j in range(hidden):
        hj = h[j]
        if hj == 0.0:
            continue
        row = w2[j]
        for k in range(n_out):
            o[k] += hj * row[k]

    mx = max(o)
    exps = [math.exp(v - mx) for v in o]
    s = sum(exps)
    if s <= 0.0:
        return None
    probs = [e / s for e in exps]
    best = 0
    for k in range(1, n_out):
        if probs[k] > probs[best]:
            best = k
    return best, probs[best]


_POLICY = _load_model("policy.json", expect_classes=MOVES)
_ROUTER = _load_model("router.json", expect_classes=ROUTE_AGENTS)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def predict(cells, pos, goal):
    """Executor policy net.

    Returns ("NORTH"|"SOUTH"|"EAST"|"WEST", confidence 0..1).
    Returns ("", 0.0) if the weights are missing/malformed or the inputs are
    not a usable 64-cell grid. Never raises.
    """
    if _POLICY is None:
        return ("", 0.0)
    try:
        if cells is None or len(cells) != GRID_W * GRID_H:
            return ("", 0.0)
        x = encode_features(cells, pos, goal)
        if len(x) != _POLICY["in_dim"]:
            return ("", 0.0)
        r = _forward(_POLICY, x)
        if r is None:
            return ("", 0.0)
        idx, conf = r
        return (_POLICY["classes"][idx], float(conf))
    except Exception:
        return ("", 0.0)


def predict_route(features):
    """Commander router.

    ``features`` may be a list/tuple of floats in ROUTER_FEATURE_NAMES order,
    or a dict keyed by those names (missing keys default to 0.0).

    Returns ("Commander"|"Executor"|"Verifier"|"Planner", confidence 0..1),
    or ("", 0.0) if the weights are missing/malformed. Never raises.
    """
    if _ROUTER is None:
        return ("", 0.0)
    try:
        if isinstance(features, dict):
            x = [float(features.get(n, 0.0)) for n in ROUTER_FEATURE_NAMES]
        else:
            x = [float(v) for v in features]
        if len(x) != _ROUTER["in_dim"]:
            return ("", 0.0)
        r = _forward(_ROUTER, x)
        if r is None:
            return ("", 0.0)
        idx, conf = r
        return (_ROUTER["classes"][idx], float(conf))
    except Exception:
        return ("", 0.0)


def weights_status():
    """Small helper for the backend / debugging: which models actually loaded."""
    return {
        "policy_loaded": _POLICY is not None,
        "router_loaded": _ROUTER is not None,
        "policy_path": _POLICY["path"] if _POLICY else None,
        "router_path": _ROUTER["path"] if _ROUTER else None,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
    }
