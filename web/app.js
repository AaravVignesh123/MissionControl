/* =============================================================================
   Mission Control — frontend
   Vanilla JS. No build step, no npm, no CDN. Works offline.
   Wire format: see CONTRACT.md. State object lives at json.data.result.
   ========================================================================== */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- config */
  var PARAMS   = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get("api") || "http://localhost:8800";
  var FORCED   = PARAMS.get("mode");            // "fixture" | "live" | null
  var W = 8, H = 8;
  var ENERGY_START = 40;
  var TARGET = [7, 6], START = [0, 0];
  var DEMO_HAZARD = [4, 5];                     // the frozen demo disruption cell
  var TICK_MS = 500;                            // ~2 ticks/sec
  var AGENTS = ["Commander", "Executor", "Verifier", "Planner"];
  var AGENT_COLOR = {
    Commander: "#ffb84d", Executor: "#22d3ee",
    Verifier:  "#ff4d6d", Planner:  "#a78bfa"
  };
  var ANCHOR = {                                // % coords inside #org
    Commander: [50, 13], Planner: [15, 50], Executor: [85, 50], Verifier: [50, 87]
  };
  var ARROW = { NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←" };
  var DELTA = { NORTH: [0, -1], SOUTH: [0, 1], EAST: [1, 0], WEST: [-1, 0] };

  /* ------------------------------------------------------------- dom cache */
  var $ = function (id) { return document.getElementById(id); };
  var boardEl = $("board"), cellEls = [];
  var pathLine = $("pathLine"), visitedLine = $("visitedLine"), agentDot = $("agentDot");
  var logEl = $("log"), orgEl = $("org"), wiresEl = $("orgWires");

  /* --------------------------------------------------------- runtime state */
  var state = null;          // last good STATE object
  var prevCells = null;      // for new-hazard detection
  var mode = "fixture";      // "live" | "fixture" | "replay"
  var replayFrames = [];     // ?mode=replay -- states recorded off the real backend
  var replayAt = 0;
  var running = false;
  var timer = null;
  var busy = false;
  var liveFailures = 0;
  var renderedLog = [];      // signatures of log lines already in the DOM
  var lastHandoffFire = 0;
  var lastHandoffSig = "";
  var handoffTimeouts = [];

  /* ========================================================================
     UTILITIES
     ===================================================================== */
  function idx(x, y) { return y * W + x; }
  function inBounds(x, y) { return x >= 0 && x < W && y >= 0 && y < H; }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function eq(a, b) { return !!a && !!b && a[0] === b[0] && a[1] === b[1]; }

  /* Strip the Jac envelope's bookkeeping keys (_jac_type / _jac_id / ...). */
  function stripJac(v) {
    if (Array.isArray(v)) return v.map(stripJac);
    if (v && typeof v === "object") {
      var out = {};
      for (var k in v) {
        if (!Object.prototype.hasOwnProperty.call(v, k)) continue;
        if (k.indexOf("_jac") === 0) continue;
        out[k] = stripJac(v[k]);
      }
      return out;
    }
    return v;
  }

  /* Defensive normalisation — the UI must never blank out on a odd payload. */
  function normalize(s) {
    s = s && typeof s === "object" ? s : {};
    var g = s.grid && typeof s.grid === "object" ? s.grid : {};
    var cells = Array.isArray(g.cells) ? g.cells.slice(0, 64) : [];
    while (cells.length < 64) cells.push("free");
    var ex = s.executor && typeof s.executor === "object" ? s.executor : {};
    return {
      tick: typeof s.tick === "number" ? s.tick : 0,
      status: s.status || "running",
      mission: {
        goal: (s.mission && s.mission.goal) || "—",
        tasks: (s.mission && Array.isArray(s.mission.tasks)) ? s.mission.tasks : []
      },
      grid: { w: g.w || W, h: g.h || H, cells: cells },
      executor: {
        pos: Array.isArray(ex.pos) ? ex.pos : [0, 0],
        energy: typeof ex.energy === "number" ? ex.energy : 0,
        path: Array.isArray(ex.path) ? ex.path : [],
        visited: Array.isArray(ex.visited) ? ex.visited : []
      },
      proposals: s.proposals || null,
      verdict: s.verdict || null,
      handoffs: Array.isArray(s.handoffs) ? s.handoffs : [],
      log: Array.isArray(s.log) ? s.log : [],
      stats: s.stats || { moves_committed: 0, vetoes: 0, llm_calls: 0, replans: 0, llm_mode: "mock" }
    };
  }

  /* ========================================================================
     A*  (used only by the offline fixture simulator)
     ===================================================================== */
  function findPath(cells, start, goal) {
    var sk = start[0] + "," + start[1], gk = goal[0] + "," + goal[1];
    if (!inBounds(start[0], start[1]) || !inBounds(goal[0], goal[1])) return [];
    var h = function (x, y) { return Math.abs(x - goal[0]) + Math.abs(y - goal[1]); };
    var open = [{ x: start[0], y: start[1], g: 0, f: h(start[0], start[1]) }];
    var came = {}, best = {}; best[sk] = 0;
    while (open.length) {
      var bi = 0;
      for (var i = 1; i < open.length; i++) if (open[i].f < open[bi].f) bi = i;
      var cur = open.splice(bi, 1)[0];
      var ck = cur.x + "," + cur.y;
      if (ck === gk) {
        var out = [[cur.x, cur.y]], k = ck;
        while (came[k]) { out.unshift(came[k].slice()); k = came[k][0] + "," + came[k][1]; }
        return out;
      }
      for (var d in DELTA) {
        var nx = cur.x + DELTA[d][0], ny = cur.y + DELTA[d][1];
        if (!inBounds(nx, ny)) continue;
        if (cells[idx(nx, ny)] === "hazard") continue;
        var nk = nx + "," + ny, ng = cur.g + 1;
        if (best[nk] !== undefined && best[nk] <= ng) continue;
        best[nk] = ng; came[nk] = [cur.x, cur.y];
        open.push({ x: nx, y: ny, g: ng, f: ng + h(nx, ny) });
      }
    }
    return [];
  }
  function moveBetween(a, b) {
    var dx = b[0] - a[0], dy = b[1] - a[1];
    if (dx === 1) return "EAST";
    if (dx === -1) return "WEST";
    if (dy === 1) return "SOUTH";
    if (dy === -1) return "NORTH";
    return null;
  }

  /* ========================================================================
     FIXTURE SIMULATOR — keeps the demo alive with no backend
     ===================================================================== */
  function seedState() {
    var cells = [];
    for (var i = 0; i < 64; i++) cells.push("free");
    var hz = [[3,0],[3,1],[3,2],[3,3],[3,5],[3,6],[3,7],[5,2],[6,5],[1,6],[5,7]];
    hz.forEach(function (c) { cells[idx(c[0], c[1])] = "hazard"; });
    cells[idx(START[0], START[1])] = "start";
    cells[idx(TARGET[0], TARGET[1])] = "target";
    var path = findPath(cells, START, TARGET);
    return normalize({
      tick: 0, status: "running",
      mission: {
        goal: "Locate and reach the survivor at (7,6). Avoid all hazards.",
        tasks: [
          { id: "t1", desc: "Plot safe route to survivor", owner: "Planner", state: "done" },
          { id: "t2", desc: "Advance one cell along route", owner: "Executor", state: "active" }
        ]
      },
      grid: { w: W, h: H, cells: cells },
      executor: { pos: START.slice(), energy: ENERGY_START, path: path, visited: [START.slice()] },
      proposals: null, verdict: null,
      handoffs: [],
      log: [
        { tick: 0, agent: "Commander", level: "info", text: "Mission received. Decomposed into 2 tasks." },
        { tick: 0, agent: "Planner", level: "info", text: "Route plotted: " + path.length + " cells via gap at (3,4)" }
      ],
      stats: { moves_committed: 0, vetoes: 0, llm_calls: 1, replans: 0,
               llm_mode: (state && state.stats && state.stats.llm_mode) || "mock" }
    });
  }

  function hs(s, from, to, task, payload) {
    s.handoffs.push({ tick: s.tick, from: from, to: to, task_id: task, payload: payload || {} });
    if (s.handoffs.length > 60) s.handoffs.splice(0, s.handoffs.length - 60);
  }
  function lg(s, agent, level, text) {
    s.log.push({ tick: s.tick, agent: agent, level: level, text: text });
    if (s.log.length > 300) s.log.splice(0, s.log.length - 300);
  }

  function simTick() {
    var s = clone(state);
    if (s.status !== "running") return s;
    s.tick += 1;
    s.handoffs = s.handoffs.filter(function (h) { return h.tick >= s.tick - 3; });

    var cells = s.grid.cells, pos = s.executor.pos;
    hs(s, "Commander", "Executor", "t2", { order: "advance" });

    var path = findPath(cells, pos, TARGET);
    var astar = path.length > 1 ? moveBetween(path[0], path[1]) : null;

    /* llm / net proposals: usually agree with A*, occasionally diverge so the
       three-way comparison is worth looking at. */
    var alt = null, opts = ["NORTH", "SOUTH", "EAST", "WEST"];
    for (var i = 0; i < opts.length; i++) {
      var d = DELTA[opts[i]], nx = pos[0] + d[0], ny = pos[1] + d[1];
      if (opts[i] !== astar && inBounds(nx, ny) && cells[idx(nx, ny)] !== "hazard") { alt = opts[i]; break; }
    }
    var diverge = (s.tick % 5 === 0) && alt;
    var llm = diverge ? alt : astar;
    var net = astar;
    var conf = diverge ? 0.52 + Math.random() * 0.16 : 0.78 + Math.random() * 0.2;
    var chosen = llm || net || astar, source = llm ? "llm" : (net ? "net" : "astar");
    var vals = [llm, net, astar].filter(function (v) { return !!v; });
    var agreed = vals.length > 0 && vals.every(function (v) { return v === vals[0]; });
    s.proposals = { llm: llm, net: net, astar: astar, chosen: chosen,
                    source: source, agreed: agreed, net_confidence: Math.round(conf * 100) / 100 };
    s.stats.llm_calls += 1;
    lg(s, "Executor", "info", chosen
      ? "Proposing " + chosen + (agreed ? " (llm+net+astar agree" : " (SPLIT vote") + ", conf " + s.proposals.net_confidence + ")"
      : "No viable move found");
    hs(s, "Executor", "Verifier", "t2", { move: chosen });

    /* -------- Verifier: pure rules, never an LLM call -------- */
    var d2 = DELTA[chosen] || [0, 0];
    var tx = pos[0] + d2[0], ty = pos[1] + d2[1];
    var rule = "ok", reason = "", vetoed = false;
    var prev = s.executor.visited.length > 1 ? s.executor.visited[s.executor.visited.length - 2] : null;
    if (!chosen) { vetoed = true; rule = "in_bounds"; reason = "No move proposed"; }
    else if (!inBounds(tx, ty)) { vetoed = true; rule = "in_bounds"; reason = "Move " + chosen + " leaves the grid"; }
    else if (cells[idx(tx, ty)] === "hazard") { vetoed = true; rule = "no_hazard"; reason = "Move " + chosen + " enters hazard at (" + tx + "," + ty + ")"; }
    else if (s.executor.energy <= 0) { vetoed = true; rule = "energy_budget"; reason = "Energy exhausted"; }
    else if (prev && eq(prev, [tx, ty])) { vetoed = true; rule = "no_thrash"; reason = "Move " + chosen + " oscillates back to (" + tx + "," + ty + ")"; }
    s.verdict = { vetoed: vetoed, rule: rule, reason: reason || "Move " + chosen + " approved",
                  target_cell: chosen ? [tx, ty] : null };

    if (vetoed) {
      s.stats.vetoes += 1;
      lg(s, "Verifier", "veto", "VETO: " + reason.toLowerCase());
      hs(s, "Verifier", "Commander", "t2", { rule: rule, blocked_move: chosen });
      lg(s, "Commander", "info", "Veto received. Re-delegating route planning.");
      hs(s, "Commander", "Planner", "t3", { replan: true });
      var np = findPath(cells, pos, TARGET);
      s.executor.path = np;
      s.stats.replans += 1;
      if (np.length) {
        lg(s, "Planner", "info", "Re-routed: " + np.length + " cells avoiding hazard");
        hs(s, "Planner", "Executor", "t2", { path_len: np.length });
      } else {
        lg(s, "Planner", "warn", "No safe route remains from (" + pos[0] + "," + pos[1] + ")");
        s.status = "failed";
      }
      s.mission.tasks = [
        { id: "t1", desc: "Plot safe route to survivor", owner: "Planner", state: "done" },
        { id: "t2", desc: "Advance one cell along route", owner: "Executor", state: "blocked" },
        { id: "t3", desc: "Re-route around new hazard", owner: "Commander", state: "active" }
      ];
    } else {
      hs(s, "Verifier", "Executor", "t2", { approved: chosen });
      s.executor.pos = [tx, ty];
      s.executor.energy -= 1;
      s.executor.visited.push([tx, ty]);
      s.executor.path = findPath(cells, [tx, ty], TARGET);
      s.stats.moves_committed += 1;
      lg(s, "Verifier", "success", "Move " + chosen + " approved (rule ok)");
      lg(s, "Executor", "info", "Committed " + chosen + " → (" + tx + "," + ty + "), energy " + s.executor.energy);
      s.mission.tasks = [
        { id: "t1", desc: "Plot safe route to survivor", owner: "Planner", state: "done" },
        { id: "t2", desc: "Advance one cell along route", owner: "Executor", state: "active" }
      ];
      if (eq(s.executor.pos, TARGET)) {
        s.status = "complete";
        lg(s, "Commander", "success", "SURVIVOR REACHED at (7,6). Mission complete.");
        hs(s, "Executor", "Commander", "t2", { arrived: true });
      }
      if (s.executor.energy <= 0 && s.status === "running") {
        s.status = "failed";
        lg(s, "Commander", "veto", "Energy exhausted. Mission failed.");
      }
    }
    return s;
  }

  function simInject(x, y) {
    var s = clone(state);
    s.grid.cells[idx(x, y)] = "hazard";
    s.handoffs = s.handoffs.filter(function (h) { return h.tick >= s.tick - 3; });
    lg(s, "Executor", "warn", "SENSOR: new hazard detected at (" + x + "," + y + ")");
    hs(s, "Executor", "Commander", "t2", { alert: "hazard", cell: [x, y] });
    lg(s, "Commander", "info", "Disruption acknowledged. Tasking Planner with a re-route.");
    hs(s, "Commander", "Planner", "t3", { replan: true });
    var np = findPath(s.grid.cells, s.executor.pos, TARGET);
    s.executor.path = np;
    s.stats.replans += 1;
    if (np.length) {
      lg(s, "Planner", "success", "Re-route accepted: " + np.length + " cells, hazard bypassed");
      hs(s, "Planner", "Executor", "t2", { path_len: np.length });
    } else {
      lg(s, "Planner", "veto", "NO SAFE ROUTE — survivor unreachable");
      s.status = "failed";
    }
    return s;
  }

  /* ========================================================================
     LIVE API
     ===================================================================== */
  function api(path, body, timeoutMs) {
    var ctl = ("AbortController" in window) ? new AbortController() : null;
    var to = setTimeout(function () { if (ctl) ctl.abort(); }, timeoutMs || 4000);
    return fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
      signal: ctl ? ctl.signal : undefined
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      var st = j && j.data && j.data.result;
      if (!st || typeof st !== "object") throw new Error("missing data.result");
      return normalize(stripJac(st));
    })["finally"](function () { clearTimeout(to); });
  }

  function callBackend(path, body) {
    // REPLAY: step through a recording captured from the real backend. No
    // network, no Jac, no LLM -- the demo of last resort.
    if (mode === "replay") {
      if (path === "/function/tick") {
        if (replayAt < replayFrames.length - 1) replayAt++;
      } else if (path === "/function/reset_mission") {
        replayAt = 0;
      } else if (path === "/function/inject_hazard_ahead" ||
                 path === "/function/inject_hazard") {
        // Jump to the recorded disruption so the veto is always reachable.
        for (var i = replayAt; i < replayFrames.length; i++) {
          var v = replayFrames[i].verdict;
          if (v && v.vetoed) { replayAt = Math.max(replayAt, i - 1); break; }
        }
      }
      return Promise.resolve(normalize(replayFrames[replayAt]));
    }
    if (mode === "fixture") {
      var s;
      if (path === "/function/tick") s = simTick();
      else if (path === "/function/reset_mission") s = seedState();
      else if (path === "/function/inject_hazard") s = simInject(body.x, body.y);
      else if (path === "/function/inject_hazard_ahead") s = simInject(DEMO_HAZARD[0], DEMO_HAZARD[1]);
      else s = state;
      return Promise.resolve(s);
    }
    return api(path, body).then(function (s) {
      liveFailures = 0;
      return s;
    })["catch"](function (e) {
      liveFailures++;
      toast("live backend error: " + e.message + " (" + liveFailures + "/3)");
      if (liveFailures >= 3) { setMode("fixture"); toast("backend unreachable — switched to FIXTURE"); }
      return state;                       // never blank the screen
    });
  }

  /* ========================================================================
     RENDER — board
     ===================================================================== */
  function buildBoard() {
    boardEl.innerHTML = "";
    cellEls = [];
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var c = document.createElement("div");
        c.className = "cell free";
        c.setAttribute("data-c", x + "," + y);
        boardEl.appendChild(c);
        cellEls.push(c);
      }
    }
  }

  function renderBoard(s) {
    var cells = s.grid.cells;
    var pathSet = {}, visSet = {};
    s.executor.path.forEach(function (p) { pathSet[p[0] + "," + p[1]] = 1; });
    s.executor.visited.forEach(function (p) { visSet[p[0] + "," + p[1]] = 1; });
    var vt = (s.verdict && s.verdict.vetoed && s.verdict.target_cell) ? s.verdict.target_cell : null;

    var newHazards = [];
    for (var i = 0; i < 64; i++) {
      var x = i % W, y = Math.floor(i / W), k = cells[i], key = x + "," + y;
      var cl = "cell " + (["free", "hazard", "target", "start"].indexOf(k) >= 0 ? k : "free");
      if (pathSet[key]) cl += " path";
      if (visSet[key]) cl += " visited";
      if (vt && vt[0] === x && vt[1] === y) cl += " vetoed";
      var el = cellEls[i];
      if (el.dataset.base !== cl) { el.className = cl; el.dataset.base = cl; }
      if (prevCells && prevCells[i] !== "hazard" && k === "hazard") newHazards.push([i, x, y]);
    }
    prevCells = cells.slice();

    if (newHazards.length) {
      newHazards.forEach(function (n) {
        var el = cellEls[n[0]];
        el.classList.remove("flash-new");
        void el.offsetWidth;                     // restart the animation
        el.classList.add("flash-new");
        setTimeout(function () { el.classList.remove("flash-new"); }, 2400);
      });
      var c0 = newHazards[0];
      banner("⚠  DISRUPTION — HAZARD INJECTED AT (" + c0[1] + "," + c0[2] + ")  ⚠  RE-PLANNING");
    }

    pathLine.setAttribute("points", s.executor.path.map(function (p) {
      return (p[0] + 0.5) + "," + (p[1] + 0.5);
    }).join(" "));
    visitedLine.setAttribute("points", s.executor.visited.map(function (p) {
      return (p[0] + 0.5) + "," + (p[1] + 0.5);
    }).join(" "));

    var pos = s.executor.pos;
    agentDot.hidden = false;
    agentDot.style.left = ((pos[0] + 0.5) / W * 100) + "%";
    agentDot.style.top = ((pos[1] + 0.5) / H * 100) + "%";
  }

  var bannerTimer = null;
  function banner(text) {
    var b = $("disruption");
    b.textContent = text;
    b.classList.add("show");
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(function () { b.classList.remove("show"); }, 5000);
  }
  var toastTimer = null;
  function toast(text) {
    var t = $("toast");
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove("show"); }, 3200);
  }

  /* ========================================================================
     RENDER — org panel + handoff animation (the money shot)
     ===================================================================== */
  var wires = {};
  function wireKey(a, b) { return [a, b].sort().join("|"); }

  function buildWires() {
    var ns = "http://www.w3.org/2000/svg";
    wiresEl.innerHTML = "";
    wires = {};
    for (var i = 0; i < AGENTS.length; i++) {
      for (var j = i + 1; j < AGENTS.length; j++) {
        var a = ANCHOR[AGENTS[i]], b = ANCHOR[AGENTS[j]];
        var ln = document.createElementNS(ns, "line");
        ln.setAttribute("x1", a[0]); ln.setAttribute("y1", a[1]);
        ln.setAttribute("x2", b[0]); ln.setAttribute("y2", b[1]);
        ln.setAttribute("class", "wire");
        wiresEl.appendChild(ln);
        wires[wireKey(AGENTS[i], AGENTS[j])] = ln;
      }
    }
  }

  function payloadText(p) {
    if (!p || typeof p !== "object") return "";
    var bits = [];
    for (var k in p) {
      if (!Object.prototype.hasOwnProperty.call(p, k)) continue;
      var v = p[k];
      if (Array.isArray(v)) v = "(" + v.join(",") + ")";
      bits.push(k + "=" + v);
    }
    return bits.slice(0, 2).join(" ");
  }

  /* Fire one handoff: a directional beam along the wire, a travelling pulse
     orb with a payload label, and send/receive flares on both agent panels. */
  function fireHandoff(h) {
    var from = h.from, to = h.to;
    if (!ANCHOR[from] || !ANCHOR[to] || from === to) return;
    var a = ANCHOR[from], b = ANCHOR[to];
    var color = AGENT_COLOR[from] || "#22d3ee";
    var DUR = 950;
    var ns = "http://www.w3.org/2000/svg";

    /* directional beam */
    var beam = document.createElementNS(ns, "line");
    beam.setAttribute("x1", a[0]); beam.setAttribute("y1", a[1]);
    beam.setAttribute("x2", b[0]); beam.setAttribute("y2", b[1]);
    beam.setAttribute("class", "beam");
    beam.style.stroke = color;
    beam.style.filter = "drop-shadow(0 0 6px " + color + ")";
    wiresEl.appendChild(beam);

    /* base wire glow */
    var wire = wires[wireKey(from, to)];
    if (wire) { wire.style.stroke = color; wire.style.strokeWidth = "2.5"; wire.style.opacity = "0.85"; }

    /* travelling orb + label */
    var orb = document.createElement("div");
    orb.className = "pulse";
    orb.style.color = color;
    orgEl.appendChild(orb);

    var label = document.createElement("div");
    label.className = "pulse-label";
    label.style.color = color;
    label.textContent = (h.task_id ? h.task_id + " · " : "") + (payloadText(h.payload) || from + "→" + to);
    orgEl.appendChild(label);

    /* node flares */
    var nf = $("node-" + from), nt = $("node-" + to);
    if (nf) { nf.classList.add("sending", "hot-" + from); nf.style.color = color; }

    /* push the payload label to the side of the wire so it never sits on top
       of an agent panel's own text */
    var dx = b[0] - a[0], dy = b[1] - a[1], len = Math.sqrt(dx * dx + dy * dy) || 1;
    var ox = (-dy / len) * 9, oy = (dx / len) * 9;

    /* Cleanup is idempotent and also runs off a plain timer, so nothing can be
       orphaned if requestAnimationFrame is throttled (e.g. background tab). */
    var cleaned = false;
    function cleanup() {
      if (cleaned) return;
      cleaned = true;
      orb.remove(); label.remove(); beam.remove();
      if (wire) { wire.style.stroke = ""; wire.style.strokeWidth = ""; wire.style.opacity = ""; }
      if (nf) nf.classList.remove("sending", "hot-" + from);
      if (nt) nt.classList.remove("receiving", "hot-" + to);
    }
    setTimeout(cleanup, DUR + 1400);

    var t0 = performance.now();
    function step(now) {
      var t = Math.min(1, (now - t0) / DUR);
      var e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;   // easeInOutCubic
      var px = a[0] + dx * e, py = a[1] + dy * e;
      orb.style.left = px + "%"; orb.style.top = py + "%";
      label.style.left = (px + ox) + "%"; label.style.top = (py + oy) + "%";
      if (t > 0.8) {                       // hand off to the node flare
        var f = (1 - t) / 0.2;
        orb.style.opacity = f; label.style.opacity = f;
      }
      if (t < 1) requestAnimationFrame(step);
      else if (!cleaned) {
        if (nt) { nt.classList.add("receiving", "hot-" + to); nt.style.color = AGENT_COLOR[to]; }
        orb.remove(); label.remove();
        setTimeout(cleanup, 620);
      }
    }
    requestAnimationFrame(step);
  }

  function renderHandoffs(s) {
    var cur = (s.handoffs || []).filter(function (h) { return h.tick === s.tick; });
    var list = $("handoffList");
    list.innerHTML = "";
    if (!cur.length) {
      list.innerHTML = '<span class="muted">no handoffs this tick</span>';
    } else {
      cur.forEach(function (h) {
        var c = document.createElement("span");
        c.className = "hchip fresh";
        c.style.borderColor = AGENT_COLOR[h.from] || "#1e2f45";
        c.style.color = AGENT_COLOR[h.from] || "#7b90a8";
        c.textContent = h.from + " → " + h.to + (h.task_id ? " · " + h.task_id : "");
        list.appendChild(c);
      });
    }

    /* Light the edges. Fires immediately whenever this tick's handoff set
       changes, and otherwise replays on a heartbeat so a paused screen still
       shows the agents talking. */
    if (!cur.length) return;
    var sig = cur.map(function (h) { return h.tick + h.from + h.to + h.task_id; }).join(";");
    var now = performance.now();
    if (sig === lastHandoffSig && now - lastHandoffFire < 1600) return;
    lastHandoffSig = sig;
    lastHandoffFire = now;
    cur.forEach(function (h, i) {
      handoffTimeouts.push(setTimeout(function () { fireHandoff(h); }, i * 220));
    });
    if (handoffTimeouts.length > 40) handoffTimeouts.splice(0, handoffTimeouts.length - 40);
  }

  function renderAgentTags(s) {
    var last = {};
    (s.log || []).forEach(function (e) { if (AGENT_COLOR[e.agent]) last[e.agent] = e; });
    AGENTS.forEach(function (a) {
      var n = $("node-" + a); if (!n) return;
      var tag = n.querySelector('[data-role="tag"]');
      var e = last[a];
      tag.textContent = e ? ("t" + e.tick + " · " + e.level) : "idle";
      tag.style.color = e && e.level === "veto" ? "#ff9fb1" : "";
      tag.style.borderColor = e && e.level === "veto" ? "#ff4d6d" : "";
    });
  }

  /* ========================================================================
     RENDER — proposals, verdict, stats, mission, header
     ===================================================================== */
  function renderProposals(s) {
    var p = s.proposals;
    var chip = $("agreeChip");
    ["llm", "net", "astar"].forEach(function (k) {
      var el = $("prop-" + k), b = el.querySelector("b");
      var v = p ? p[k] : null;
      b.textContent = v ? (ARROW[v] || "") + " " + v : "—";
      el.classList.remove("winner", "dissent");
      if (!p) return;
      if (p.source === k) el.classList.add("winner");
      else if (v && p.chosen && v !== p.chosen) el.classList.add("dissent");
    });
    if (!p) {
      chip.textContent = "—"; chip.className = "chip";
      $("confBar").style.width = "0%"; $("confVal").textContent = "—";
    } else {
      chip.textContent = p.agreed ? "CONSENSUS" : "SPLIT VOTE — " + (p.source || "?").toUpperCase() + " OVERRODE";
      chip.className = "chip " + (p.agreed ? "agreed" : "split");
      var c = typeof p.net_confidence === "number" ? p.net_confidence : 0;
      $("confBar").style.width = Math.round(c * 100) + "%";
      $("confVal").textContent = c.toFixed(2);
    }

    var v = s.verdict, ve = $("verdict");
    if (!v) { ve.className = "verdict"; ve.innerHTML = '<span class="muted">no verdict yet</span>'; return; }
    ve.className = "verdict " + (v.vetoed ? "veto" : "ok");
    ve.innerHTML = (v.vetoed ? "⛔ <b>VETO · " + v.rule + "</b><br>" : "✅ <b>APPROVED · " + v.rule + "</b><br>") +
      escapeHtml(v.reason || "") +
      (v.target_cell ? ' <span class="muted">→ cell (' + v.target_cell[0] + "," + v.target_cell[1] + ")</span>" : "");
  }

  function escapeHtml(t) {
    return String(t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function renderHeader(s) {
    $("r-tick").textContent = s.tick;
    var st = $("r-status");
    st.textContent = s.status;
    st.className = "pill " + s.status;
    $("r-energy").textContent = s.executor.energy;
    var bar = $("r-energy-bar");
    var pct = Math.max(0, Math.min(100, (s.executor.energy / ENERGY_START) * 100));
    bar.style.width = pct + "%";
    bar.className = pct < 30 ? "low" : "";

    var lm = (s.stats && s.stats.llm_mode) === "live";
    var bl = $("badge-llm");
    bl.textContent = lm ? "◉ LIVE LLM" : "◍ MOCK LLM";
    bl.className = "badge " + (lm ? "live" : "mock");

    var bm = $("badge-mode");
    bm.textContent = mode === "live"   ? "SRC: LIVE API :8800"
                   : mode === "replay" ? "SRC: REPLAY " + (replayAt + 1) + "/" + replayFrames.length
                                       : "SRC: FIXTURE (offline)";
    bm.className = "badge " + (mode === "live" ? "src-live" : "src-fixture");

    $("s-moves").textContent = s.stats.moves_committed || 0;
    $("s-vetoes").textContent = s.stats.vetoes || 0;
    $("s-llm").textContent = s.stats.llm_calls || 0;
    $("s-replans").textContent = s.stats.replans || 0;

    $("goal").textContent = s.mission.goal;
    var ul = $("tasks");
    ul.innerHTML = "";
    (s.mission.tasks || []).forEach(function (t) {
      var li = document.createElement("li");
      var who = document.createElement("span");
      who.className = "who"; who.textContent = t.owner || "?";
      who.style.color = AGENT_COLOR[t.owner] || "#7b90a8";
      var d = document.createElement("span"); d.textContent = t.desc || "";
      var stt = document.createElement("span");
      stt.className = "st " + (t.state || "pending"); stt.textContent = t.state || "pending";
      li.appendChild(who); li.appendChild(d); li.appendChild(stt);
      ul.appendChild(li);
    });
  }

  /* ========================================================================
     RENDER — reasoning log
     ===================================================================== */
  function logSig(e) { return e.tick + "|" + e.agent + "|" + e.level + "|" + e.text; }

  function renderLog(s) {
    var sigs = (s.log || []).map(logSig);
    var isAppend = sigs.length >= renderedLog.length &&
      renderedLog.every(function (v, i) { return sigs[i] === v; });
    var startAt = 0;
    if (isAppend) startAt = renderedLog.length;
    else { logEl.innerHTML = ""; startAt = 0; }

    var nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 90;
    for (var i = startAt; i < s.log.length; i++) {
      var e = s.log[i];
      var div = document.createElement("div");
      var agent = AGENT_COLOR[e.agent] ? e.agent : "Commander";
      div.className = "entry a-" + agent + " lv-" + (e.level || "info") + (isAppend && startAt > 0 ? " new" : "");
      var t = document.createElement("span"); t.className = "t"; t.textContent = "t" + e.tick;
      var ag = document.createElement("span"); ag.className = "ag"; ag.textContent = e.agent || "?";
      var tx = document.createElement("span"); tx.className = "tx"; tx.textContent = e.text || "";
      div.appendChild(t); div.appendChild(ag); div.appendChild(tx);
      logEl.appendChild(div);
    }
    renderedLog = sigs;
    if (nearBottom || !isAppend) {
      logEl.scrollTop = logEl.scrollHeight;
      requestAnimationFrame(function () { logEl.scrollTop = logEl.scrollHeight; });
    }
  }

  /* ========================================================================
     MASTER RENDER
     ===================================================================== */
  function render(s) {
    if (!s) return;
    state = s;
    try {
      renderHeader(s);
      renderBoard(s);
      renderProposals(s);
      renderAgentTags(s);
      renderHandoffs(s);
      renderLog(s);
      if (s.status !== "running" && running) setRunning(false);
    } catch (err) {
      console.error("render error", err);
    }
  }

  /* ========================================================================
     CONTROL LOOP
     ===================================================================== */
  function setMode(m) {
    mode = m;
    if (m === "fixture") liveFailures = 0;
    if (state) renderHeader(state);
  }

  function setRunning(on) {
    running = on;
    var b = $("btn-run");
    b.textContent = on ? "❚❚ PAUSE" : "▶ START";
    b.classList.toggle("on", on);
    if (timer) { clearInterval(timer); timer = null; }
    if (on) { timer = setInterval(loop, TICK_MS); return; }
    /* Paused: live mode keeps polling get_state ~2×/sec; fixture mode replays
       the current tick's handoff animation so the org panel stays alive. */
    timer = setInterval(function () {
      if (mode === "live") pollOnly();
      else if (state) renderHandoffs(state);
    }, TICK_MS);
  }

  function loop() {
    if (busy) return;
    busy = true;
    callBackend("/function/tick", {}).then(function (s) { render(s); })
      ["finally"](function () { busy = false; });
  }

  function pollOnly() {
    if (busy || mode !== "live") return;
    busy = true;
    callBackend("/function/get_state", {}).then(function (s) { render(s); })
      ["finally"](function () { busy = false; });
  }

  function doStep() {
    if (busy) return;
    busy = true;
    callBackend("/function/tick", {}).then(function (s) { render(s); })
      ["finally"](function () { busy = false; });
  }

  function doReset() {
    setRunning(false);
    busy = true;
    callBackend("/function/reset_mission", {}).then(function (s) {
      prevCells = null; renderedLog = []; logEl.innerHTML = "";
      render(s); toast("mission reset");
    })["finally"](function () { busy = false; if (mode === "live") setRunning(false); });
  }

  function doHazard() {
    busy = true;
    // Target the cell the Executor is ACTUALLY about to enter rather than a
    // fixed coordinate: the three proposers don't always agree on the route,
    // so a hardcoded cell can miss and the veto never fires. The backend also
    // refuses any cell that would strand the mission.
    callBackend("/function/inject_hazard_ahead", {})
      .then(function (s) { render(s); })
      ["finally"](function () { busy = false; });
  }

  function wireControls() {
    $("btn-run").addEventListener("click", function () { setRunning(!running); });
    $("btn-step").addEventListener("click", doStep);
    $("btn-reset").addEventListener("click", doReset);
    $("btn-hazard").addEventListener("click", doHazard);
    $("badge-mode").addEventListener("click", function () {
      if (mode === "live") { setMode("fixture"); toast("forced FIXTURE mode"); return; }
      toast("probing live backend…");
      api("/function/get_state", {}, 1500).then(function (s) {
        setMode("live"); render(s); toast("connected to live backend");
      })["catch"](function () { toast("backend still unreachable — staying on fixture"); });
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      var tn = (ev.target && ev.target.tagName) || "";
      if (tn === "INPUT" || tn === "TEXTAREA") return;
      var k = String(ev.key || "").toLowerCase();
      var isSpace = k === " " || k === "space" || k === "spacebar" || ev.code === "Space" || ev.keyCode === 32;
      if (k === "h") { ev.preventDefault(); doHazard(); }
      else if (isSpace) { ev.preventDefault(); setRunning(!running); }
      else if (k === "n") { ev.preventDefault(); doStep(); }
      else if (k === "r") { ev.preventDefault(); doReset(); }
      if (document.activeElement && document.activeElement.tagName === "BUTTON") document.activeElement.blur();
    });
  }

  /* ========================================================================
     BOOT
     ===================================================================== */
  function loadFixture() {
    return fetch("sample_state.json", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) { return normalize(stripJac(j.data && j.data.result ? j.data.result : j)); })
      ["catch"](function () { return seedState(); });   // works even from file://
  }

  function boot() {
    buildBoard();
    buildWires();
    wireControls();

    if (FORCED === "replay") {
      return fetch("replay.json", { cache: "no-store" })
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
        .then(function (j) {
          replayFrames = (j.frames || []).map(function (f) { return normalize(stripJac(f)); });
          if (!replayFrames.length) throw new Error("empty recording");
          replayAt = 0;
          setMode("replay");
          render(replayFrames[0]);
          toast("REPLAY: " + replayFrames.length + " recorded frames, no backend needed");
        })["catch"](function () {
          toast("replay.json unavailable — falling back to fixture");
          return loadFixture().then(function (fx) { setMode("fixture"); render(fx); });
        })["finally"](function () { setRunning(false); });
    }

    loadFixture().then(function (fx) {
      setMode("fixture");
      render(fx);                                   // paint immediately — never blank
      if (FORCED === "fixture") { toast("fixture mode (forced)"); return; }
      return api("/function/get_state", {}, 1500).then(function (s) {
        setMode("live"); render(s); toast("live backend detected on :8800");
      })["catch"](function () {
        if (FORCED === "live") toast("live mode requested but :8800 unreachable — using fixture");
        else toast("no backend on :8800 — running on fixture");
      });
    })["finally"](function () { setRunning(false); });   // start the paused heartbeat
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
