/* =============================================================================
   Mission Control — SAR operations simulator, frontend.
   Vanilla JS. No build step, no npm, no CDN. Works offline.
   Wire format: see CONTRACT.md. State object lives at json.data.result.
   ========================================================================== */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- config */
  var PARAMS   = new URLSearchParams(location.search);
  var API_BASE = PARAMS.get("api") || "http://localhost:8800";
  var FORCED   = PARAMS.get("mode");            // "fixture" | "live" | "replay" | null
  var W = 8, H = 8;
  var TICK_MS = 667;                            // ~1.5 ticks/sec
  var ICS_ROLES = ["Incident Commander", "Operations", "Rescue Team", "Safety Officer"];
  var ROLE_DESC = {
    "Incident Commander": "Decompose the objective, re-task the team on a HALT.",
    "Operations": "Plan and re-plan the search/egress route.",
    "Rescue Team": "Advance one cell along the approved route.",
    "Safety Officer": "Clear each advance against doctrine against live ground truth."
  };
  var ARROW = { NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←" };
  var DELTA = { NORTH: [0, -1], SOUTH: [0, 1], EAST: [1, 0], WEST: [-1, 0] };
  var GLYPH = { hazard: "▲", staging: "⌂", "victim-remaining": "●", "victim-rescued": "✓" };

  /* ---------------------------------------------------- built-in fallback --
     Static domain data, used only when list_profiles is unreachable. Shape
     matches list_profiles verbatim (mirrors the live backend's own data). */
  var FALLBACK_PROFILES = {
    profiles: [
      { id: "urban", name: "Urban Search & Rescue", code: "US&R", agency: "FEMA US&R Task Force",
        summary: "Locate and reach victims trapped in the void spaces of a collapsed structure. Unstable rubble is lethal; monitor air and keep an egress path.",
        resource_label: "SCBA air", resource_unit: "min", hazard_noun: "collapse zone",
        safe_noun: "cleared void", objective_noun: "trapped victim", staging_noun: "staging area",
        disruption_label: "SECONDARY COLLAPSE",
        cell_labels: { open: "Cleared void", hazard: "Collapse zone", staging: "Staging area", victim: "Trapped victim" } },
      { id: "wildland", name: "Wildland Fire Rescue", code: "WILDLAND", agency: "Wildland Fire Crew",
        summary: "Reach a trapped party ahead of an advancing fire. Active fire is lethal. LCES doctrine: never let the fire sever your escape route to the safety zone.",
        resource_label: "Egress window", resource_unit: "min", hazard_noun: "active fire",
        safe_noun: "black / burned", objective_noun: "trapped party", staging_noun: "safety zone",
        disruption_label: "FIRE SPREAD",
        cell_labels: { open: "Black / safe", hazard: "Active fire", staging: "Safety zone", victim: "Trapped party" } },
      { id: "swiftwater", name: "Swiftwater Rescue", code: "SWIFTWATER", agency: "Swiftwater Rescue Team",
        summary: "Reach victims in moving water. Hydraulics and strainers are lethal. Manage cold-water exposure and keep a route back to the bank.",
        resource_label: "Exposure budget", resource_unit: "min", hazard_noun: "hydraulic / strainer",
        safe_noun: "slack water", objective_noun: "victim in water", staging_noun: "bank staging",
        disruption_label: "RISING WATER",
        cell_labels: { open: "Slack water", hazard: "Hydraulic / strainer", staging: "Bank staging", victim: "Victim in water" } }
    ],
    tunables: {
      team_size: { min: 1, max: 6, default: 3, label: "Team size", help: "Responders on the operation. More crew extends the resource window (rotation)." },
      resource_budget: { min: 16, max: 72, default: 44, label: "Resource budget", help: "Operational window (reach victims AND egress to staging)." },
      hazard_density: { min: 0, max: 6, default: 3, label: "Hazard density", help: "How much of the area is impassable. Higher = harder routing." },
      victim_count: { min: 1, max: 3, default: 2, label: "Victims", help: "People to locate and reach." },
      risk_tolerance: { values: ["conservative", "standard", "aggressive"], default: "standard", label: "Risk tolerance", help: "Safety Officer's air-reserve doctrine (rule of thirds)." }
    },
    default_profile: "urban"
  };

  /* ------------------------------------------------------------- dom cache */
  var $ = function (id) { return document.getElementById(id); };

  /* --------------------------------------------------------- runtime state */
  var screen = "setup";
  var SETUP = { profiles: [], tunables: {}, defaultProfile: "urban", selected: "urban", values: {} };

  var state = null;          // last good STATE object (dashboard)
  var prevCells = null;      // for new-hazard detection
  var boardBuiltForProfile = null;
  var mode = "fixture";      // "live" | "fixture" | "replay"
  var replayFrames = [];
  var replayAt = 0;
  var running = false;
  var timer = null;
  var busy = false;
  var liveFailures = 0;
  var renderedLog = [];
  var cellEls = [];

  /* ========================================================================
     UTILITIES
     ===================================================================== */
  function idx(x, y) { return y * W + x; }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function eqCell(a, b) { return !!a && !!b && a[0] === b[0] && a[1] === b[1]; }
  function inList(list, c) {
    return Array.isArray(list) && list.some(function (p) { return eqCell(p, c); });
  }
  function escapeHtml(t) {
    return String(t).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

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

  /* Defensive normalisation of a STATE object — the UI must never blank out
     on an odd payload. */
  function normalizeState(s) {
    s = s && typeof s === "object" ? s : {};
    var g = s.grid && typeof s.grid === "object" ? s.grid : {};
    var cells = Array.isArray(g.cells) ? g.cells.slice(0, 64) : [];
    while (cells.length < 64) cells.push("open");
    var r = s.responder && typeof s.responder === "object" ? s.responder : {};
    var v = s.victims && typeof s.victims === "object" ? s.victims : {};
    var p = s.profile && typeof s.profile === "object" ? s.profile : {};
    var m = s.mission && typeof s.mission === "object" ? s.mission : {};
    var st = s.stats && typeof s.stats === "object" ? s.stats : {};
    return {
      tick: typeof s.tick === "number" ? s.tick : 0,
      clock: s.clock || "T+00:00",
      status: s.status || "running",
      profile: {
        id: p.id || "urban", name: p.name || "Operation", code: p.code || "—",
        agency: p.agency || "—", summary: p.summary || "",
        resource_label: p.resource_label || "Resource", resource_unit: p.resource_unit || "",
        hazard_noun: p.hazard_noun || "hazard", safe_noun: p.safe_noun || "open",
        objective_noun: p.objective_noun || "victim", staging_noun: p.staging_noun || "staging",
        disruption_label: p.disruption_label || "DISRUPTION",
        cell_labels: p.cell_labels || { open: "Open", hazard: "Hazard", staging: "Staging", victim: "Victim" }
      },
      config: s.config || {},
      mission: {
        objective: m.objective || "—",
        phase: m.phase || "search",
        tasks: Array.isArray(m.tasks) ? m.tasks : []
      },
      grid: { w: g.w || W, h: g.h || H, cells: cells, staging: Array.isArray(g.staging) ? g.staging : null },
      responder: {
        pos: Array.isArray(r.pos) ? r.pos : [0, 0],
        resource: typeof r.resource === "number" ? r.resource : 0,
        resource_max: typeof r.resource_max === "number" ? r.resource_max : 1,
        objective: Array.isArray(r.objective) ? r.objective : null,
        path: Array.isArray(r.path) ? r.path : [],
        visited: Array.isArray(r.visited) ? r.visited : []
      },
      victims: {
        total: typeof v.total === "number" ? v.total : 0,
        reached: typeof v.reached === "number" ? v.reached : 0,
        remaining: Array.isArray(v.remaining) ? v.remaining : [],
        rescued: Array.isArray(v.rescued) ? v.rescued : []
      },
      proposals: s.proposals || null,
      verdict: s.verdict || null,
      handoffs: Array.isArray(s.handoffs) ? s.handoffs : [],
      log: Array.isArray(s.log) ? s.log : [],
      stats: {
        moves_committed: st.moves_committed || 0, vetoes: st.vetoes || 0,
        llm_calls: st.llm_calls || 0, replans: st.replans || 0,
        rescued: st.rescued || 0, llm_mode: st.llm_mode || "mock"
      }
    };
  }

  /* ========================================================================
     LIVE API
     ===================================================================== */
  function apiRaw(path, body, timeoutMs) {
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
      var res = j && j.data && j.data.result;
      if (res === undefined || res === null) throw new Error("missing data.result");
      return stripJac(res);
    })["finally"](function () { clearTimeout(to); });
  }

  function apiState(path, body, timeoutMs) {
    return apiRaw(path, body, timeoutMs).then(normalizeState);
  }

  /* ========================================================================
     SCREEN 1 — SETUP
     ===================================================================== */
  function initSetupValues() {
    SETUP.selected = SETUP.defaultProfile;
    SETUP.values = {};
    for (var k in SETUP.tunables) {
      if (!Object.prototype.hasOwnProperty.call(SETUP.tunables, k)) continue;
      SETUP.values[k] = SETUP.tunables[k].default;
    }
  }

  function renderProfileCards() {
    var wrap = $("profileCards");
    wrap.innerHTML = "";
    SETUP.profiles.forEach(function (p) {
      var card = document.createElement("div");
      card.className = "profile-card" + (p.id === SETUP.selected ? " selected" : "");
      card.setAttribute("tabindex", "0");
      card.setAttribute("role", "button");
      card.innerHTML =
        '<div class="profile-card-top"><h3>' + escapeHtml(p.name) + '</h3>' +
        '<span class="profile-code">' + escapeHtml(p.code) + '</span></div>' +
        '<div class="profile-agency">' + escapeHtml(p.agency) + '</div>' +
        '<p class="profile-summary">' + escapeHtml(p.summary) + '</p>' +
        '<span class="profile-select-mark">SELECTED</span>';
      card.addEventListener("click", function () {
        SETUP.selected = p.id;
        renderProfileCards();
      });
      wrap.appendChild(card);
    });
  }

  function renderTunables() {
    var wrap = $("tunablesForm");
    wrap.innerHTML = "";
    for (var key in SETUP.tunables) {
      if (!Object.prototype.hasOwnProperty.call(SETUP.tunables, key)) continue;
      var spec = SETUP.tunables[key];
      var box = document.createElement("div");
      box.className = "tunable";
      if (Array.isArray(spec.values)) {
        box.innerHTML =
          '<div class="tunable-head"><label>' + escapeHtml(spec.label || key) + '</label></div>' +
          '<div class="segmented" data-key="' + key + '"></div>' +
          '<p class="tunable-help">' + escapeHtml(spec.help || "") + '</p>';
        wrap.appendChild(box);
        var seg = box.querySelector(".segmented");
        spec.values.forEach(function (v) {
          var b = document.createElement("button");
          b.type = "button";
          b.textContent = v;
          b.className = v === SETUP.values[key] ? "active" : "";
          b.addEventListener("click", function () {
            SETUP.values[key] = v;
            seg.querySelectorAll("button").forEach(function (btn) { btn.classList.remove("active"); });
            b.classList.add("active");
          });
          seg.appendChild(b);
        });
      } else {
        var min = spec.min, max = spec.max;
        box.innerHTML =
          '<div class="tunable-head"><label>' + escapeHtml(spec.label || key) + '</label>' +
          '<span class="tval">' + SETUP.values[key] + '</span></div>' +
          '<input type="range" min="' + min + '" max="' + max + '" step="1" value="' + SETUP.values[key] + '">' +
          '<p class="tunable-help">' + escapeHtml(spec.help || "") + '</p>';
        wrap.appendChild(box);
        var input = box.querySelector("input");
        var vEl = box.querySelector(".tval");
        input.addEventListener("input", function () {
          SETUP.values[key] = parseInt(input.value, 10);
          vEl.textContent = input.value;
        });
      }
    }
  }

  function setupSourceNote(text, cls) {
    var el = $("setupSourceNote");
    el.textContent = text;
    el.className = "source-note" + (cls ? " " + cls : "");
  }

  function loadSetup() {
    if (FORCED === "fixture" || FORCED === "replay") {
      SETUP.profiles = FALLBACK_PROFILES.profiles;
      SETUP.tunables = FALLBACK_PROFILES.tunables;
      SETUP.defaultProfile = FALLBACK_PROFILES.default_profile;
      initSetupValues();
      renderProfileCards();
      renderTunables();
      setupSourceNote(FORCED === "replay" ? "mode=replay — recorded run, launch replays it" : "mode=fixture — static snapshot on launch", "offline");
      return;
    }
    setupSourceNote("checking backend…");
    apiRaw("/function/list_profiles", {}, 2500).then(function (res) {
      SETUP.profiles = res.profiles || FALLBACK_PROFILES.profiles;
      SETUP.tunables = res.tunables || FALLBACK_PROFILES.tunables;
      SETUP.defaultProfile = res.default_profile || FALLBACK_PROFILES.default_profile;
      initSetupValues();
      renderProfileCards();
      renderTunables();
      setupSourceNote("live backend detected on :8800", "live");
    })["catch"](function () {
      SETUP.profiles = FALLBACK_PROFILES.profiles;
      SETUP.tunables = FALLBACK_PROFILES.tunables;
      SETUP.defaultProfile = FALLBACK_PROFILES.default_profile;
      initSetupValues();
      renderProfileCards();
      renderTunables();
      setupSourceNote("no backend on :8800 — using built-in profile data; mission will launch into a fixture snapshot", "offline");
    });
  }

  function doLaunch() {
    var btn = $("btn-launch") || $("btnLaunch");
    btn.disabled = true;
    var origText = btn.textContent;
    btn.textContent = "Launching…";
    var seed = parseInt($("seedInput").value, 10) || 7;

    function finishLaunch(s, launchedMode) {
      setMode(launchedMode);
      prevCells = null; boardBuiltForProfile = null; renderedLog = [];
      $("log").innerHTML = "";
      showDashboard();
      render(s);
      btn.disabled = false;
      btn.textContent = origText;
    }

    if (FORCED === "replay") {
      fetch("replay.json", { cache: "no-store" })
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
        .then(function (j) {
          replayFrames = (j.frames || []).map(function (f) { return normalizeState(stripJac(f)); });
          if (!replayFrames.length) throw new Error("empty recording");
          replayAt = 0;
          finishLaunch(replayFrames[0], "replay");
          toast("REPLAY: " + replayFrames.length + " recorded frames, no backend needed");
        })["catch"](function () {
          loadFixture().then(function (fx) { finishLaunch(fx, "fixture"); toast("replay.json unavailable — using fixture"); });
        });
      return;
    }

    if (FORCED === "fixture") {
      loadFixture().then(function (fx) { finishLaunch(fx, "fixture"); toast("fixture mode — static snapshot"); });
      return;
    }

    var tunables = clone(SETUP.values);
    apiState("/function/configure_mission", { profile: SETUP.selected, tunables: tunables, seed: seed }, 15000)
      .then(function (s) {
        finishLaunch(s, "live");
        toast("mission configured on live backend");
      })["catch"](function () {
        loadFixture().then(function (fx) {
          finishLaunch(fx, "fixture");
          toast("live backend unreachable — launched into fixture snapshot");
        });
      });
  }

  /* ========================================================================
     SCREEN SWITCHING
     ===================================================================== */
  function showSetup() {
    screen = "setup";
    setRunning(false);
    $("screen-dashboard").classList.add("hidden");
    $("screen-setup").classList.remove("hidden");
  }
  function showDashboard() {
    screen = "dashboard";
    $("screen-setup").classList.add("hidden");
    $("screen-dashboard").classList.remove("hidden");
  }

  /* ========================================================================
     FIXTURE / REPLAY DATA SOURCES
     ===================================================================== */
  function loadFixture() {
    return fetch("sample_state.json", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) { return normalizeState(stripJac(j && j.data && j.data.result ? j.data.result : j)); });
  }

  function callBackend(path, body) {
    if (mode === "replay") {
      if (path === "/function/tick") {
        if (replayAt < replayFrames.length - 1) replayAt++;
      } else if (path === "/function/reset_mission") {
        replayAt = 0;
      } else if (path === "/function/inject_disruption") {
        for (var i = replayAt; i < replayFrames.length; i++) {
          var v = replayFrames[i].verdict;
          if (v && v.vetoed) { replayAt = Math.max(replayAt, i); break; }
        }
      }
      return Promise.resolve(replayFrames[replayAt]);
    }
    if (mode === "fixture") {
      if (path === "/function/reset_mission") {
        return loadFixture().then(function (fx) { toast("fixture reset — static snapshot"); return fx; });
      }
      if (path === "/function/inject_disruption") {
        toast("fixture mode: static snapshot — disruption requires a live backend");
        return Promise.resolve(state);
      }
      // tick: no-op in fixture mode
      return Promise.resolve(state);
    }
    var callTimeout = (path === "/function/reset_mission") ? 15000 : 4000;
    return apiState(path, body, callTimeout).then(function (s) {
      liveFailures = 0;
      return s;
    })["catch"](function (e) {
      liveFailures++;
      toast("live backend error: " + e.message + " (" + liveFailures + "/3)");
      if (liveFailures >= 3) { setMode("fixture"); toast("backend unreachable — switched to FIXTURE"); }
      return state;
    });
  }

  /* ========================================================================
     RENDER — header / status strip
     ===================================================================== */
  var STATUS_LABEL = { running: "In progress", complete: "Complete", aborted: "Return to base", failed: "Failed" };

  function renderHeader(s) {
    $("d-profile-name").textContent = s.profile.name;
    $("d-profile-code").textContent = s.profile.code;
    $("d-profile-agency").textContent = s.profile.agency;

    $("d-clock").textContent = s.clock;
    var pill = $("d-status");
    pill.textContent = STATUS_LABEL[s.status] || s.status;
    pill.className = "status-pill " + s.status;

    $("d-victims").textContent = s.victims.reached + " / " + s.victims.total;

    $("d-resource-label").textContent = s.profile.resource_label;
    var pct = Math.max(0, Math.min(100, (s.responder.resource / (s.responder.resource_max || 1)) * 100));
    var bar = $("d-resource-bar");
    bar.style.width = pct + "%";
    bar.className = pct < 30 ? "low" : "";
    $("d-resource-val").textContent = s.responder.resource + " / " + s.responder.resource_max +
      (s.profile.resource_unit ? " " + s.profile.resource_unit : "");

    var lm = s.stats.llm_mode === "live";
    var bl = $("badge-llm");
    bl.textContent = lm ? "LLM LIVE" : "LLM MOCK";
    bl.className = "badge " + (lm ? "live" : "mock");

    var bm = $("badge-mode");
    bm.textContent = mode === "live" ? "SRC LIVE :8800"
      : mode === "replay" ? "SRC REPLAY " + (replayAt + 1) + "/" + replayFrames.length
      : "SRC FIXTURE";
    bm.className = "badge " + (mode === "live" ? "src-live" : mode === "replay" ? "src-replay" : "src-fixture");

    $("d-objective").textContent = s.mission.objective;

    $("btn-hazard").textContent = "Inject disruption: " + s.profile.disruption_label;
  }

  /* ========================================================================
     RENDER — board
     ===================================================================== */
  function buildBoard() {
    var boardEl = $("board");
    boardEl.innerHTML = "";
    cellEls = [];
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var c = document.createElement("div");
        c.className = "cell open";
        c.setAttribute("data-c", x + "," + y);
        var g = document.createElement("span");
        g.className = "glyph";
        c.appendChild(g);
        boardEl.appendChild(c);
        cellEls.push(c);
      }
    }
  }

  function buildLegend(profile) {
    var cl = profile.cell_labels || {};
    var el = $("legend");
    el.innerHTML =
      '<span><i class="lg-open"></i>' + escapeHtml(cl.open || "Open") + '</span>' +
      '<span><i class="lg-hazard"></i>' + escapeHtml(cl.hazard || "Hazard") + '</span>' +
      '<span><i class="lg-staging"></i>' + escapeHtml(cl.staging || "Staging") + '</span>' +
      '<span><i class="lg-victim"></i>' + escapeHtml(cl.victim || "Victim") + ' (remaining)</span>' +
      '<span><i class="lg-rescued"></i>' + escapeHtml(cl.victim || "Victim") + ' (rescued)</span>' +
      '<span><i class="lg-responder"></i>responder</span>';
  }

  function renderBoard(s) {
    if (boardBuiltForProfile !== s.profile.id) {
      buildLegend(s.profile);
      boardBuiltForProfile = s.profile.id;
    }
    var cells = s.grid.cells;
    var pathSet = {}, visSet = {};
    s.responder.path.forEach(function (p) { pathSet[p[0] + "," + p[1]] = 1; });
    s.responder.visited.forEach(function (p) { visSet[p[0] + "," + p[1]] = 1; });
    var vt = (s.verdict && s.verdict.vetoed && s.verdict.target_cell) ? s.verdict.target_cell : null;

    var newHazards = [];
    for (var i = 0; i < 64; i++) {
      var x = i % W, y = Math.floor(i / W), k = cells[i], key = x + "," + y;
      var kind = k;
      if (k === "victim") kind = inList(s.victims.rescued, [x, y]) ? "victim-rescued" : "victim-remaining";
      else if (inList(s.victims.rescued, [x, y])) kind = "victim-rescued";
      else if (["open", "hazard", "staging"].indexOf(k) < 0) kind = "open";
      var cl = "cell " + kind;
      if (pathSet[key]) cl += " path";
      if (visSet[key]) cl += " visited";
      if (vt && vt[0] === x && vt[1] === y) cl += " vetoed";
      var el = cellEls[i];
      if (el.dataset.base !== cl) { el.className = cl; el.dataset.base = cl; }
      var glyph = el.querySelector(".glyph");
      var gText = GLYPH[kind] || "";
      if (glyph.textContent !== gText) glyph.textContent = gText;
      if (prevCells && prevCells[i] !== "hazard" && k === "hazard") newHazards.push(i);
    }
    prevCells = cells.slice();

    if (newHazards.length) {
      newHazards.forEach(function (i) {
        var el = cellEls[i];
        el.classList.remove("hazard-new");
        void el.offsetWidth;
        el.classList.add("hazard-new");
      });
    }

    var pl = $("pathLine"), vl = $("visitedLine");
    pl.setAttribute("points", s.responder.path.map(function (p) { return (p[0] + 0.5) + "," + (p[1] + 0.5); }).join(" "));
    vl.setAttribute("points", s.responder.visited.map(function (p) { return (p[0] + 0.5) + "," + (p[1] + 0.5); }).join(" "));

    var pos = s.responder.pos;
    var token = $("responderToken");
    token.hidden = false;
    token.style.left = ((pos[0] + 0.5) / W * 100) + "%";
    token.style.top = ((pos[1] + 0.5) / H * 100) + "%";
  }

  /* ========================================================================
     RENDER — ICS roster
     ===================================================================== */
  function renderRoster(s) {
    var wrap = $("roster");
    if (!wrap.children.length) {
      ICS_ROLES.forEach(function (role) {
        var tile = document.createElement("div");
        tile.className = "role-tile";
        tile.setAttribute("data-role", role);
        tile.innerHTML =
          '<div class="role-name"><span>' + role + '</span><span class="handoff-mark">→</span></div>' +
          '<div class="role-task"></div>' +
          '<span class="role-state"></span>';
        wrap.appendChild(tile);
      });
    }

    var handoffsNow = (s.handoffs || []).filter(function (h) { return h.tick === s.tick; });
    var involved = {};
    handoffsNow.forEach(function (h) { involved[h.from] = 1; involved[h.to] = 1; });

    ICS_ROLES.forEach(function (role) {
      var tile = wrap.querySelector('[data-role="' + role + '"]');
      var task = (s.mission.tasks || []).filter(function (t) { return t.owner === role; })[0];
      var taskEl = tile.querySelector(".role-task");
      var stateEl = tile.querySelector(".role-state");

      var desc = task ? task.desc : ROLE_DESC[role];
      if (role === "Safety Officer" && s.verdict) {
        desc = s.verdict.vetoed
          ? "HALT (" + s.verdict.rule + "): " + s.verdict.reason
          : "Cleared: " + (s.verdict.reason || "advance approved");
      }
      taskEl.textContent = desc || "—";

      var stt = task ? task.state : (role === "Safety Officer" ? (s.verdict ? (s.verdict.vetoed ? "blocked" : "active") : "pending") : "pending");
      stateEl.textContent = stt;
      stateEl.className = "role-state " + stt;

      tile.classList.toggle("handoff-active", !!involved[role]);
      tile.classList.toggle("halt", role === "Safety Officer" && !!(s.verdict && s.verdict.vetoed));
    });
  }

  /* ========================================================================
     RENDER — stats + decision panel
     ===================================================================== */
  function renderStats(s) {
    var wrap = $("statsRow");
    var items = [
      ["MOVES", s.stats.moves_committed],
      ["VETOES", s.stats.vetoes],
      ["REPLANS", s.stats.replans],
      ["RESCUED", s.stats.rescued]
    ];
    wrap.innerHTML = items.map(function (it) {
      return '<div class="stat"><b class="mono">' + it[1] + '</b><label>' + it[0] + '</label></div>';
    }).join("");
  }

  function renderDecision(s) {
    var p = s.proposals;
    var row = $("proposalRow");
    var keys = ["llm", "net", "astar"];
    row.innerHTML = keys.map(function (k) {
      var v = p ? p[k] : null;
      var cls = "proposal";
      if (p) {
        if (p.source === k) cls += " winner";
        else if (v && p.chosen && v !== p.chosen) cls += " dissent";
      }
      return '<div class="' + cls + '"><label>' + k.toUpperCase() + '</label><b>' +
        (v ? (ARROW[v] || "") + " " + v : "—") + '</b></div>';
    }).join("");

    var chip = $("decisionAgree");
    if (!p) {
      chip.textContent = "—"; chip.className = "agree-chip";
    } else {
      chip.textContent = p.agreed ? "consensus" : "split — " + (p.source || "?") + " chosen";
      chip.className = "agree-chip " + (p.agreed ? "agreed" : "split");
    }

    var v = s.verdict, ve = $("verdictLine");
    if (!v) {
      ve.className = "verdict-line";
      ve.innerHTML = '<span class="muted">no verdict yet</span>';
    } else {
      ve.className = "verdict-line " + (v.vetoed ? "veto" : "ok");
      ve.innerHTML = (v.vetoed ? "HALT · " + escapeHtml(v.rule) : "APPROVED · " + escapeHtml(v.rule)) +
        '<br>' + escapeHtml(v.reason || "");
    }

    var body = $("reasoningBody");
    var lines = [];
    if (p) {
      lines.push("llm=" + p.llm + "  net=" + p.net + "  astar=" + p.astar);
      lines.push("chosen=" + p.chosen + "  source=" + p.source + "  agreed=" + p.agreed);
      if (typeof p.net_confidence === "number") lines.push("net_confidence=" + p.net_confidence.toFixed(3));
    }
    if (v) {
      lines.push("");
      lines.push("rule=" + v.rule + "  vetoed=" + v.vetoed);
      if (v.target_cell) lines.push("target_cell=(" + v.target_cell[0] + "," + v.target_cell[1] + ")");
      lines.push("reason: " + (v.reason || ""));
    }
    body.textContent = lines.join("\n") || "no data yet";
  }

  /* ========================================================================
     RENDER — incident log
     ===================================================================== */
  function logSig(e) { return e.tick + "|" + e.agent + "|" + e.level + "|" + e.text; }

  function renderLog(s) {
    var logEl = $("log");
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
      div.className = "entry lv-" + (e.level || "info");
      var t = document.createElement("span"); t.className = "t"; t.textContent = e.clock || ("t" + e.tick);
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
      renderRoster(s);
      renderStats(s);
      renderDecision(s);
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
    if (b) {
      b.textContent = on ? "❚❚ Pause" : "▶ Start";
      b.classList.toggle("on", on);
    }
    if (timer) { clearInterval(timer); timer = null; }
    if (on) { timer = setInterval(loop, TICK_MS); }
  }

  function loop() {
    if (busy) return;
    busy = true;
    callBackend("/function/tick", {}).then(function (s) { render(s); })
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
      prevCells = null; renderedLog = []; $("log").innerHTML = "";
      render(s); toast("mission reset");
    })["finally"](function () { busy = false; });
  }

  function doHazard() {
    if (busy) return;
    busy = true;
    callBackend("/function/inject_disruption", {})
      .then(function (s) { render(s); })
      ["finally"](function () { busy = false; });
  }

  var toastTimer = null;
  function toast(text) {
    var t = $("toast");
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove("show"); }, 3200);
  }

  function wireControls() {
    $("btn-run").addEventListener("click", function () { setRunning(!running); });
    $("btn-step").addEventListener("click", doStep);
    $("btn-reset").addEventListener("click", doReset);
    $("btn-hazard").addEventListener("click", doHazard);
    $("btn-new").addEventListener("click", function () { setRunning(false); showSetup(); });
    $("badge-mode").addEventListener("click", function () {
      if (mode === "replay") return;
      if (mode === "live") { setMode("fixture"); toast("forced FIXTURE mode"); return; }
      toast("probing live backend…");
      apiState("/function/get_state", {}, 1500).then(function (s) {
        setMode("live"); render(s); toast("connected to live backend");
      })["catch"](function () { toast("backend still unreachable — staying on fixture"); });
    });
    document.addEventListener("keydown", function (ev) {
      if (screen !== "dashboard") return;
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
    $("btnLaunch").addEventListener("click", doLaunch);
  }

  /* ========================================================================
     BOOT
     ===================================================================== */
  function boot() {
    buildBoard();
    wireControls();
    showSetup();
    loadSetup();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
