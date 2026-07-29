"""Mission Control - the byLLM reasoning layer.

Two LLM-backed reasoners sit behind plain, exception-proof wrappers:

  commander_decompose(goal)      -> TaskPlan
  executor_propose(situation)    -> MoveProposal
  brain_mode()                   -> "live" | "mock"

Every prompt lives in a `sem` statement (never a docstring) - the `sem` text IS
what the model sees.  Both public entry points are total functions: they never
raise, and on any failure they return a *clearly invalid* value (an empty task
list / an empty move) so the backend can fall through to its deterministic A*
result without special-casing exceptions.

The model is chosen once at import time: Groq (fast, low-latency — ideal for a
live per-tick sim loop) when GROQ_API_KEY is present, else Anthropic when
ANTHROPIC_API_KEY is present, else byLLM's MockLLM so the whole system runs
offline with no key at all.  `brain_mode()` inspects the object actually in use,
so it can never claim "live" while running mocked.
"""

import os;
import re;
import from jaclang.byllm.lib { Model, MockLLM }
import from jaclang.byllm.llm { BaseLLM }

# ---------------------------------------------------------------------------
# Frozen wire types (see CONTRACT.md)
# ---------------------------------------------------------------------------

obj PlannedTask {
    has id: str;
    has desc: str;
    has owner: str;
}

obj TaskPlan {
    has tasks: list[PlannedTask];
}

obj MoveProposal {
    has move: str;
    has rationale: str;
}

sem PlannedTask.id =
    "Short stable identifier in execution order: t1, t2, t3, t4.";
sem PlannedTask.desc =
    "One short imperative phrase naming a single concrete action, under 60 "
    "characters, written for a mission-control display. Example: 'Plot a "
    "hazard-free route to the survivor'. No agent name, no trailing period.";
sem PlannedTask.owner =
    "Exactly one of the four ICS roles: Incident Commander, Operations, Rescue "
    "Team, Safety Officer. Assign by whose job it is - Operations plans the "
    "search route, the Rescue Team advances cell by cell, the Safety Officer "
    "clears or halts each advance against doctrine, the Incident Commander "
    "re-tasks after a halt. Never invent another role.";
sem TaskPlan.tasks =
    "Two to four tasks in the order they should be carried out, together "
    "covering route planning, advancing to victims, and safety verification.";

sem MoveProposal.move =
    "Exactly one of NORTH, SOUTH, EAST, WEST in upper case and nothing else - "
    "no diagonals, no coordinates, no punctuation, no explanation.";
sem MoveProposal.rationale =
    "One short sentence, at most 100 characters, in the clipped voice of a "
    "field radio call, telling an incident commander watching the board why "
    "this step was chosen. Plain present-tense English, no step-by-step "
    "reasoning, no coordinate dumps, no markdown. "
    "Example: 'Advancing east toward the victim; collapse zone holds to the south.'";

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

glob GROQ_MODEL: str = "groq/llama-3.3-70b-versatile";
glob ANTHROPIC_MODEL: str = "claude-sonnet-4-6";
glob DIRECTIONS: list[str] = ["NORTH", "SOUTH", "EAST", "WEST"];
# ICS roles — the exact role strings the rest of the system uses.
glob AGENTS: list[str] = ["Incident Commander", "Operations", "Rescue Team", "Safety Officer"];
glob _ALIASES: dict[str, str] = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "UP": "NORTH", "DOWN": "SOUTH", "RIGHT": "EAST", "LEFT": "WEST",
    "NORTHWARD": "NORTH", "SOUTHWARD": "SOUTH",
    "EASTWARD": "EAST", "WESTWARD": "WEST"
};
glob _OPPOSITE: dict[str, str] = {
    "NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"
};

# ---------------------------------------------------------------------------
# Live / mock switch
# ---------------------------------------------------------------------------

def _make_llm -> BaseLLM {
    # A bad key or missing provider must never take the demo down — every
    # branch degrades to the offline mock and lets brain_mode() say so.
    gkey: str = os.environ.get("GROQ_API_KEY") or "";
    if gkey.strip() != "" {
        try {
            return Model(model_name=GROQ_MODEL);
        } except Exception {}
    }
    akey: str = os.environ.get("ANTHROPIC_API_KEY") or "";
    if akey.strip() != "" {
        try {
            return Model(model_name=ANTHROPIC_MODEL);
        } except Exception {}
    }
    return MockLLM(model_name="mockllm", config={"outputs": []});
}

glob llm: BaseLLM = _make_llm();


def brain_mode -> str {
    return "mock" if isinstance(llm, MockLLM) else "live";
}

sem brain_mode =
    "Report which reasoning backend is actually wired up: 'live' for a real "
    "Anthropic model, 'mock' for the offline canned-output backend.";

# ---------------------------------------------------------------------------
# The LLM functions.  `by llm(...)` REPLACES the body - the sem text is the prompt.
# ---------------------------------------------------------------------------

def _llm_decompose(goal: str) -> TaskPlan by llm(
    temperature=0.0,
    max_tokens=400,
    max_output_retries=2
);

sem _llm_decompose =
    "You are the Incident Commander of a search-and-rescue task force running "
    "an operation on an 8x8 sector grid under the Incident Command System. "
    "Break the mission objective into 2 to 4 concrete, ordered tasks the team "
    "will carry out, and assign each to exactly one role: Operations (plans the "
    "search route), Rescue Team (advances one cell at a time toward victims), "
    "Safety Officer (clears or halts each advance against safety doctrine), "
    "Incident Commander (re-tasks after a halt). Cover route planning, "
    "advancing to victims, and safety verification. Keep each task a single "
    "short imperative phrase for a command-post board. Do not restate the "
    "objective as a task and do not invent roles outside the four listed.";
sem _llm_decompose.goal =
    "The operation objective in plain English, including how many victims and "
    "any standing constraints such as avoiding hazards and keeping an egress.";


def _llm_move(situation: str) -> MoveProposal by llm(
    temperature=0.0,
    max_tokens=150,
    max_output_retries=1
);

sem _llm_move =
    "You are the Rescue Team advancing through a search-and-rescue sector on an "
    "8x8 grid. The origin (0,0) is the top-left cell, x is the column and y is "
    "the row, so NORTH decreases y, SOUTH increases y, EAST increases x and "
    "WEST decreases x. Read the situation report and choose the single one-cell "
    "advance that makes the most progress toward the stated objective (a victim "
    "during search, or the staging area during egress). Never advance into a "
    "cell the report marks BLOCKED and never step off the grid. If the direct "
    "step is blocked, pick the best sideways step that keeps a route open. "
    "Answer with one direction and one clipped field-radio sentence.";
sem _llm_move.situation =
    "A plain-text situation report: the team's current cell, the objective "
    "cell, remaining resource, and the state of each neighbouring cell "
    "(BLOCKED or clear).";

# ---------------------------------------------------------------------------
# Sanitisers - the happy path is good, the failure path is harmless
# ---------------------------------------------------------------------------

def _tidy(raw: str, limit: int) -> str {
    text: str = " ".join(str(raw).split());
    head: str = text.partition(". ")[0];
    if head != text {
        text = head + ".";
    }
    if len(text) > limit {
        text = text[:limit].rstrip() + "...";
    }
    return text;
}

sem _tidy = "Collapse whitespace, keep the first sentence, clamp the length.";
sem _tidy.raw = "Raw model text.";
sem _tidy.limit = "Maximum characters to keep.";


def _clean_move(raw: str) -> str {
    token: str = re.sub(r"[^A-Za-z]", "", str(raw)).upper();
    if token in DIRECTIONS {
        return token;
    }
    return _ALIASES.get(token, "");
}

sem _clean_move = "Normalise a model direction to NORTH/SOUTH/EAST/WEST or ''.";
sem _clean_move.raw = "Raw model direction text.";


def _clean_owner(raw: str) -> str {
    text: str = str(raw).strip().lower();
    for name in AGENTS {
        if name.lower() == text {
            return name;
        }
    }
    for name in AGENTS {
        if name.lower() in text {
            return name;
        }
    }
    return "Operations";
}

sem _clean_owner = "Snap a model-supplied owner onto one of the four agents.";
sem _clean_owner.raw = "Raw model owner text.";


def _clean_plan(plan: TaskPlan) -> TaskPlan {
    kept: list[PlannedTask] = [];
    for task in plan.tasks {
        desc: str = _tidy(task.desc, 72);
        if desc == "" {
            continue;
        }
        kept.append(
            PlannedTask(
                id=f"t{len(kept) + 1}",
                desc=desc,
                owner=_clean_owner(task.owner)
            )
        );
        if len(kept) == 4 {
            break;
        }
    }
    if len(kept) < 2 {
        return TaskPlan(tasks=[]);
    }
    return TaskPlan(tasks=kept);
}

sem _clean_plan = "Clamp a plan to 2-4 well-formed tasks with renumbered ids.";
sem _clean_plan.plan = "Raw plan straight off the model.";


def _clean_proposal(proposal: MoveProposal) -> MoveProposal {
    move: str = _clean_move(proposal.move);
    if move == "" {
        return MoveProposal(move="", rationale="");
    }
    why: str = _tidy(proposal.rationale, 100);
    if why == "" {
        why = f"{move} is the best available step toward the survivor.";
    }
    return MoveProposal(move=move, rationale=why);
}

sem _clean_proposal = "Validate a move proposal; empty move means unusable.";
sem _clean_proposal.proposal = "Raw proposal straight off the model.";

# ---------------------------------------------------------------------------
# Canned mock outputs (offline path)
# ---------------------------------------------------------------------------

glob _mock_prev: str = "";
glob _mock_step: int = 0;
glob _mock_trail: list[list[int]] = [];
glob _MOCK_DELTA: dict[str, list[int]] = {
    "NORTH": [0, -1], "SOUTH": [0, 1], "EAST": [1, 0], "WEST": [-1, 0]
};


def _mock_plan -> TaskPlan {
    return TaskPlan(
        tasks=[
            PlannedTask(id="t1", desc="Plan a hazard-free search route to victims", owner="Operations"),
            PlannedTask(id="t2", desc="Advance one cell toward the objective", owner="Rescue Team"),
            PlannedTask(id="t3", desc="Clear each advance against safety doctrine", owner="Safety Officer"),
            PlannedTask(id="t4", desc="Re-task the team after a halt", owner="Incident Commander")
        ]
    );
}

sem _mock_plan = "Canned four-task plan used when no API key is present.";


def _pairs(text: str) -> list[list[int]] {
    found: list[list[int]] = [];
    for hit in re.findall(r"(\d+)\s*,\s*(\d+)", text) {
        found.append([int(hit[0]), int(hit[1])]);
    }
    return found;
}

sem _pairs = "Pull every 'x,y' coordinate pair out of a situation report.";
sem _pairs.text = "Situation report text.";


def _pair_near(text: str, keys: list[str]) -> list[int] {
    low: str = text.lower();
    at: int = -1;
    for key in keys {
        if key not in low {
            continue;
        }
        found: int = low.index(key);
        if at < 0 or found < at {
            at = found;
        }
    }
    if at < 0 {
        return [];
    }
    hit = re.search(r"(\d+)\s*,\s*(\d+)", text[at:at + 90]);
    if hit is None {
        return [];
    }
    return [int(hit.group(1)), int(hit.group(2))];
}

sem _pair_near = "Find the first coordinate pair following any of the keywords.";
sem _pair_near.text = "Situation report text.";
sem _pair_near.keys = "Lower-case keywords to anchor on.";


def _blocked(text: str) -> list[str] {
    # Reports look like "NORTH=(3,3):hazard, SOUTH=(3,5):hazard, EAST=(4,4):free"
    # or "NORTH is out of bounds." - so read the window that follows each
    # direction word, stopping at the next direction word.
    up: str = text.upper();
    spans: list[list[int]] = [];
    names: list[str] = [];
    for hit in re.finditer(r"NORTH|SOUTH|EAST|WEST", up) {
        spans.append([hit.start(), hit.end()]);
        names.append(hit.group(0));
    }
    bad: list[str] = [];
    for (i, span) in enumerate(spans) {
        stop: int = len(up);
        if i + 1 < len(spans) {
            stop = spans[i + 1][0];
        }
        if stop > span[1] + 40 {
            stop = span[1] + 40;
        }
        window: str = up[span[1]:stop];
        flagged: bool = (
            "HAZARD" in window or "BLOCKED" in window or "WALL" in window
            or "BOUNDS" in window or "OFF-GRID" in window or "UNSAFE" in window
            or "OCCUPIED" in window or "FIRE" in window or "DEBRIS" in window
        );
        if flagged and names[i] not in bad {
            bad.append(names[i]);
        }
    }
    return bad;
}

sem _blocked = "Directions the report describes as hazardous or off-grid.";
sem _blocked.text = "Situation report text.";


def _mentioned(text: str) -> list[str] {
    seen: list[str] = [];
    for hit in re.finditer(r"NORTH|SOUTH|EAST|WEST", text.upper()) {
        name: str = hit.group(0);
        if name not in seen {
            seen.append(name);
        }
    }
    return seen;
}

sem _mentioned = "Every direction the report names at all - a listed neighbour.";
sem _mentioned.text = "Situation report text.";


def _mock_proposal(situation: str) -> MoveProposal {
    _mock_step += 1;
    goal: list[int] = _pair_near(situation, ["victim", "objective", "staging", "survivor", "target", "goal"]);
    here: list[int] = _pair_near(situation, ["at (", "current", "position", "you are", "team", "robot"]);
    seen: list[list[int]] = _pairs(situation);
    if len(here) < 2 and len(seen) > 0 {
        here = seen[0];
    }
    if len(goal) < 2 and len(seen) > 1 {
        goal = seen[len(seen) - 1];
    }
    # Short memory of recently-occupied cells, so the greedy heuristic doesn't
    # ring around a wall it can't cross (the offline path stays clean without
    # leaning on the Safety Officer's livelock guard to bail it out).
    if len(here) >= 2 {
        _mock_trail = (_mock_trail + [[here[0], here[1]]])[-6:];
    }
    bad: list[str] = _blocked(situation);
    # A direction the report never lists is not a neighbour at all (grid edge),
    # so when the report does enumerate neighbours, only trust that set.
    okay: list[str] = [];
    for name in _mentioned(situation) {
        if name not in bad {
            okay.append(name);
        }
    }
    back: str = _OPPOSITE.get(_mock_prev, "");

    order: list[str] = [];
    if len(here) >= 2 and len(goal) >= 2 {
        dx: int = goal[0] - here[0];
        dy: int = goal[1] - here[1];
        horiz: str = "EAST" if dx > 0 else ("WEST" if dx < 0 else "");
        vert: str = "SOUTH" if dy > 0 else ("NORTH" if dy < 0 else "");
        order = [horiz, vert] if abs(dx) >= abs(dy) else [vert, horiz];
    } else {
        # No fix on either cell: sweep toward the far corner, alternating.
        order = ["EAST", "SOUTH"] if _mock_step % 2 == 1 else ["SOUTH", "EAST"];
    }
    for name in DIRECTIONS {
        if name not in order {
            order.append(name);
        }
    }

    allow: list[str] = okay if len(okay) > 0 else DIRECTIONS;
    pick: str = "";
    # Pass 1: best forward step that is not blocked, not a reversal, and does
    # not re-enter a cell we were just on (breaks rings around a wall).
    for name in order {
        if name in allow and name not in bad and name != back {
            tc: list[int] = [];
            if len(here) >= 2 {
                d = _MOCK_DELTA[name];
                tc = [here[0] + d[0], here[1] + d[1]];
            }
            looped: bool = False;
            for c in _mock_trail {
                if len(tc) == 2 and c[0] == tc[0] and c[1] == tc[1] { looped = True; }
            }
            if not looped {
                pick = name;
                break;
            }
        }
    }
    # Pass 2: drop the trail guard (allowed to backtrack across old ground).
    if pick == "" {
        for name in order {
            if name in allow and name not in bad and name != back {
                pick = name;
                break;
            }
        }
    }
    # Pass 3: dead end -- drop the anti-thrash guard too.
    if pick == "" {
        for name in order {
            if name in allow and name not in bad {
                pick = name;
                break;
            }
        }
    }
    if pick == "" {
        pick = "EAST";
    }
    _mock_prev = pick;

    why: str = "";
    if len(goal) >= 2 {
        gx: int = goal[0];
        gy: int = goal[1];
        why = f"Advancing {pick} toward the objective at ({gx},{gy}).";
        if pick in bad or (len(order) > 0 and order[0] != "" and order[0] != pick) {
            why = f"Direct approach blocked, so {pick} keeps a route open.";
        }
    } else {
        why = f"No fix on the objective yet - sweeping {pick} across the sector.";
    }
    return MoveProposal(move=pick, rationale=why);
}

sem _mock_proposal = "Canned move used when no API key is present.";
sem _mock_proposal.situation = "Situation report text.";


def _prime(value: object) {
    if isinstance(llm, MockLLM) {
        llm.config["outputs"] = [value];
    }
}

sem _prime = "Reload the MockLLM output queue so it can never run dry.";
sem _prime.value = "The canned value the next `by llm` call should return.";

# ---------------------------------------------------------------------------
# Circuit breaker - a dead network must not stall the per-tick demo loop
# ---------------------------------------------------------------------------

glob TRIP_AFTER: int = 3;
glob COOLDOWN: int = 10;
glob _fails: int = 0;
glob _skips: int = 0;


def _tripped -> bool {
    if _fails < TRIP_AFTER {
        return False;
    }
    _skips += 1;
    if _skips >= COOLDOWN {
        # Periodically let one probe through so a recovered model is picked up.
        _skips = 0;
        _fails = 0;
        return False;
    }
    return True;
}

sem _tripped = "True while the reasoner is being skipped after repeated failures.";


def _record(ok: bool) {
    if ok {
        _fails = 0;
        _skips = 0;
    } else {
        _fails += 1;
    }
}

sem _record = "Update the failure streak that drives the circuit breaker.";
sem _record.ok = "Whether the last reasoner call produced a usable answer.";

# ---------------------------------------------------------------------------
# Public API - total functions, never raise
# ---------------------------------------------------------------------------

def commander_decompose(goal: str) -> TaskPlan {
    if _tripped() {
        return TaskPlan(tasks=[]);
    }
    try {
        _prime(_mock_plan());
        plan: TaskPlan = _clean_plan(_llm_decompose(goal));
        _record(len(plan.tasks) > 0);
        return plan;
    } except Exception {
        _record(False);
        return TaskPlan(tasks=[]);
    }
}

sem commander_decompose =
    "Decompose a rescue mission goal into 2-4 owned tasks. Returns an empty "
    "task list if the model is unreachable or its answer is unusable.";
sem commander_decompose.goal = "The mission objective in plain English.";


def executor_propose(situation: str) -> MoveProposal {
    if _tripped() {
        return MoveProposal(
            move="",
            rationale="Reasoner offline - deferring to the planned route."
        );
    }
    try {
        _prime(_mock_proposal(situation));
        pick: MoveProposal = _clean_proposal(_llm_move(situation));
        _record(pick.move != "");
        return pick;
    } except Exception {
        _record(False);
        return MoveProposal(
            move="",
            rationale="Reasoner unavailable - deferring to the planned route."
        );
    }
}

sem executor_propose =
    "Propose the next one-cell move from a situation report. `move` is one of "
    "NORTH/SOUTH/EAST/WEST, or '' when the answer is unusable and the caller "
    "should fall through to the next proposer.";
sem executor_propose.situation = "The situation report shown to the reasoner.";
