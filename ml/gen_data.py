"""Generate the behavioural-cloning dataset for the executor policy net.

Samples ~20k random 8x8 grids (randomised hazard density, random start/goal),
runs the local A*/BFS oracle on each, and records the OPTIMAL FIRST MOVE as the
label. Writes ml/data/policy_data.npz.

Run:  python3 ml/gen_data.py

Features come from lib/policy.encode_features -- the same function the pure
python inference path uses -- so train and inference cannot drift.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import _astar as A  # noqa: E402
from lib.policy import encode_features, FEATURE_DIM, MOVES  # noqa: E402

N_GRIDS = 20000
SAMPLES_PER_GRID = 3          # start cell + up to 2 extra reachable cells
SEED = 20260728
OUT = os.path.join(_HERE, "data", "policy_data.npz")

MOVE_IDX = {m: i for i, m in enumerate(MOVES)}


def random_grid(rng):
    """A random solvable grid. Returns (cells, start, goal, dist_map) or None."""
    density = rng.uniform(0.05, 0.30)
    mask = rng.random(64) < density
    cells = ["hazard" if mask[i] else "free" for i in range(64)]

    free = [i for i in range(64) if cells[i] == "free"]
    if len(free) < 4:
        return None
    si, gi = rng.choice(free, size=2, replace=False)
    if si == gi:
        return None
    start = (int(si) % 8, int(si) // 8)
    goal = (int(gi) % 8, int(gi) // 8)

    d = A.dist_map(cells, goal)
    if d[start[1] * 8 + start[0]] == A.INF:
        return None  # unsolvable, throw it away

    cells[start[1] * 8 + start[0]] = "start"
    cells[goal[1] * 8 + goal[0]] = "target"
    return cells, start, goal, d


def main():
    rng = np.random.default_rng(SEED)
    X, Y, M = [], [], []
    t0 = time.time()
    grids = 0
    tries = 0

    while grids < N_GRIDS:
        tries += 1
        if tries > N_GRIDS * 20:
            break
        g = random_grid(rng)
        if g is None:
            continue
        cells, start, goal, d = g
        grids += 1

        # candidate agent positions: the start, plus other reachable cells
        reach = [i for i in range(64) if d[i] != A.INF and d[i] > 0]
        picks = [start[1] * 8 + start[0]]
        if len(reach) > 1:
            extra = rng.choice(reach, size=min(SAMPLES_PER_GRID - 1, len(reach)),
                               replace=False)
            for e in extra:
                if int(e) not in picks:
                    picks.append(int(e))

        for idx in picks:
            pos = (idx % 8, idx // 8)
            lbl = A.best_move(cells, pos, goal, d)
            if lbl is None:
                continue
            opts = A.optimal_moves(cells, pos, goal, d)
            X.append(encode_features(cells, pos, goal))
            Y.append(MOVE_IDX[lbl])
            M.append([1 if m in opts else 0 for m in MOVES])

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    M = np.asarray(M, dtype=np.uint8)

    assert X.shape[1] == FEATURE_DIM, (X.shape, FEATURE_DIM)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=Y, opt_mask=M)

    counts = np.bincount(Y, minlength=4)
    ties = (M.sum(axis=1) > 1).mean()
    print("grids generated      : %d (from %d attempts)" % (grids, tries))
    print("samples              : %d" % X.shape[0])
    print("feature dim          : %d" % X.shape[1])
    print("label balance        : " + ", ".join(
        "%s=%.3f" % (m, counts[i] / len(Y)) for i, m in enumerate(MOVES)))
    print("samples with >1 optimal first move: %.1f%%" % (100 * ties))
    print("wrote %s (%.1f MB) in %.1fs" % (
        OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
