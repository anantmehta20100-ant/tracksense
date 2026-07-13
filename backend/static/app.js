/* TrackSense dashboard — plain vanilla JS (no framework, no CDN).
   Polls the existing Flask JSON API and renders the single-page dashboard. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var CLASS_ORDER = ["nut_butter_jar", "cutlery", "bread", "plate"];
  var POLL_MS = 500;
  var pollTimer = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function score(x) { return (x == null) ? "—" : Number(x).toFixed(3); }

  function api(path, opts) {
    opts = opts || {};
    return fetch(path, opts).then(function (r) {
      if (!r.ok && !opts.allowError) throw new Error(path + " → " + r.status);
      return r.json();
    });
  }
  function postJSON(path, body) {
    return api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
  }

  function showError(msg) {
    var e = $("controls-error");
    e.textContent = msg;
    e.hidden = false;
  }
  function clearError() { var e = $("controls-error"); e.hidden = true; e.textContent = ""; }

  // -- track_id -> class_name map (for resolving risk chains) --------------
  function classMap(snap) {
    var m = {};
    (snap.objects || []).forEach(function (o) { m[o.track_id] = o.class_name; });
    (snap.tracked_objects || []).forEach(function (o) {
      if (m[o.track_id] === undefined) m[o.track_id] = o.class_name;
    });
    return m;
  }
  function objectsByClass(snap) {
    var m = {};
    (snap.objects || []).forEach(function (o) { m[o.class_name] = o; });
    return m;
  }

  // ---------------------------------------------------------------- status
  function renderStatus(snap) {
    var pill = $("status-pill");
    var st = snap.status || "idle";
    pill.textContent = st;
    pill.className = "pill pill-" + st;

    var isLive = (snap.mode === "live") || (snap.source_kind === "yolo") || (snap.source === "yolo");
    var srcEl = $("source-label");
    if (srcEl) {
      srcEl.textContent = "Source: " + (isLive ? "yolo/live" : "mock");
      srcEl.className = "pill " + (isLive ? "pill-running" : "pill-idle");
    }

    var pct = Math.round((snap.progress || 0) * 100);
    $("progress-bar").style.width = pct + "%";
    $("progress-text").textContent = (snap.cursor || 0) + " / " + (snap.total_frames || 0);

    $("verify-source").textContent =
      "source: " + esc(snap.source || snap.source_kind || "—") +
      "  ·  model: " + esc(snap.model || "—") +
      "  ·  physical_verification: " + String(!!snap.physical_verification);

    // hero chain badges reflect live risk
    var byClass = objectsByClass(snap);
    document.querySelectorAll("#hero-chain .chain-node").forEach(function (node) {
      var cls = node.getAttribute("data-class");
      var badge = node.querySelector("[data-badge]");
      var o = byClass[cls];
      var rc = o ? o.risk_class : "LOW";
      badge.textContent = o ? rc : "—";
      badge.className = "node-badge badge " + rc;
    });
  }

  // ------------------------------------------------------------ risk cards
  function renderRiskCards(snap) {
    var host = $("risk-cards");
    var byClass = objectsByClass(snap);
    var cmap = classMap(snap);
    var html = CLASS_ORDER.map(function (cls) {
      var o = byClass[cls];
      if (!o) {
        return '<div class="card LOW">' +
          '<div class="card-head"><span class="card-name">' + esc(cls) + '</span>' +
          '<span class="badge LOW">—</span></div>' +
          '<div class="card-meta">not yet tracked</div></div>';
      }
      var chain = (o.risk_chain || []).map(function (tid) {
        return esc(cmap[tid] || ("#" + tid));
      }).join(' <span class="sep">→</span> ');
      var pct = Math.round((o.risk_score || 0) * 100);
      return '<div class="card ' + o.risk_class + '">' +
        '<div class="card-head">' +
          '<span class="card-name">' + esc(o.class_name) + '</span>' +
          '<span class="badge ' + o.risk_class + '">' + esc(o.risk_class) + '</span>' +
        '</div>' +
        '<div class="card-meta">track ' + esc(o.track_id) +
          '  ·  depth ' + esc(o.propagation_depth || 0) +
          '  ·  contacts ' + esc(o.contact_count || 0) + '</div>' +
        '<div class="score-row">' +
          '<span class="score-num">' + score(o.risk_score) + '</span>' +
          '<span class="bar"><i class="' + o.risk_class + '" style="width:' + pct + '%"></i></span>' +
        '</div>' +
        '<div class="card-chain">chain: ' + (chain || esc(o.class_name)) + '</div>' +
      '</div>';
    }).join("");
    host.innerHTML = html;
  }

  // -------------------------------------------------------------- timeline
  function renderTimeline(snap) {
    var host = $("contact-timeline");
    var contacts = (snap.timeline || []).filter(function (e) { return e.type === "contact"; });
    if (contacts.length === 0) {
      host.innerHTML = '<p class="muted empty">No contact events yet.</p>';
      return;
    }
    host.innerHTML = contacts.map(function (e) {
      return '<div class="ev">' +
        '<span class="ev-t mono">f' + esc(e.frame_index) + '</span>' +
        '<div class="ev-main">' +
          '<div class="ev-pair">' + esc(e.source_class) + ' ↔ ' + esc(e.target_class) + '</div>' +
          '<div class="ev-sub">t=' + score(e.timestamp) + 's  ·  dur=' + score(e.duration) +
            's  ·  overlap=' + score(e.overlap_ratio) + '</div>' +
        '</div>' +
        '<span class="badge ' + e.risk_class + '">' + esc(e.risk_class) + ' ' + score(e.risk_score) + '</span>' +
      '</div>';
    }).join("");
  }

  // ----------------------------------------------------------- explanation
  function renderExplanation(snap) {
    var host = $("explanation-panel");
    var exps = snap.explanations || [];
    var plate = exps.filter(function (e) { return e.object === "plate"; })[0] || null;
    if (!plate) {
      host.innerHTML = '<p class="muted empty">Run the demo to trace the propagation chain to the plate.</p>';
      return;
    }
    var nodes = (plate.chain || []).map(function (item, i) {
      var root = (i === 0) ? " root" : "";
      var arrow = (i > 0) ? '<span class="exp-arrow">→</span>' : "";
      return arrow + '<span class="exp-node' + root + '">' + esc(item.class) + '</span>';
    }).join("");
    host.innerHTML =
      '<div class="exp-chain">' + nodes + '</div>' +
      '<p class="exp-text">The plate is elevated because risk propagated indirectly through ' +
        'cutlery and bread, even though the plate never directly touched the allergen source.</p>' +
      '<p class="muted small">' + esc(plate.note || "") + '</p>';
  }

  // ---------------------------------------------------------------- alerts
  function renderAlerts(snap) {
    var host = $("alerts-panel");
    var alerts = snap.alerts || [];
    if (alerts.length === 0) {
      host.innerHTML = '<p class="muted empty">No high-risk alerts yet.</p>';
      return;
    }
    host.innerHTML = alerts.map(function (a) {
      return '<div class="alert">' +
        '<div class="alert-head">' +
          '<span class="alert-obj">' + esc(a.object) + '</span>' +
          '<span class="badge ' + a.risk_class + '">' + esc(a.risk_class) + ' ' + score(a.risk_score) + '</span>' +
        '</div>' +
        '<div class="alert-chain">' + esc(a.chain_text) + '</div>' +
      '</div>';
    }).join("");
  }

  function renderAll(snap) {
    if (!snap) return;
    renderStatus(snap);
    renderRiskCards(snap);
    renderTimeline(snap);
    renderExplanation(snap);
    renderAlerts(snap);
  }

  // --------------------------------------------------------------- polling
  function startPolling() {
    stopPolling();
    pollTimer = setInterval(function () {
      api("/api/snapshot").then(function (snap) {
        renderAll(snap);
        if (snap.status === "finished") stopPolling();
      }).catch(function (e) { showError(e.message); stopPolling(); });
    }, POLL_MS);
  }
  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

  // ------------------------------------------------------------ YOLO smoke
  function loadSmoke() {
    var sumEl = $("smoke-summary");
    var gal = $("smoke-gallery");
    api("/api/yolo-smoke-summary").then(function (data) {
      if (data.summary_exists && data.summary_text) {
        sumEl.innerHTML = "";
        var pre = document.createElement("pre");
        pre.textContent = data.summary_text;
        sumEl.appendChild(pre);
      } else {
        sumEl.innerHTML = '<p class="muted empty">No YOLO smoke-test summary found yet.</p>';
      }
      gal.innerHTML = "";
      if (!data.images || data.images.length === 0) {
        gal.innerHTML = '<p class="muted empty">No YOLO smoke-test images found yet.</p>';
        return;
      }
      data.images.forEach(function (img) {
        var shot = document.createElement("div"); shot.className = "shot";
        var a = document.createElement("a"); a.href = img.url; a.target = "_blank"; a.rel = "noopener";
        var im = document.createElement("img"); im.loading = "lazy"; im.src = img.url; im.alt = img.filename;
        a.appendChild(im); shot.appendChild(a);
        var cap = document.createElement("div"); cap.className = "cap"; cap.textContent = img.filename;
        shot.appendChild(cap); gal.appendChild(shot);
      });
    }).catch(function () {
      sumEl.innerHTML = '<p class="muted empty">Could not load smoke-test summary.</p>';
      gal.innerHTML = '<p class="muted empty">No YOLO smoke-test images found yet.</p>';
    });
  }

  // ----------------------------------------------------------- live status
  function setLiveButtons(running) {
    var s = $("live-start"), t = $("live-stop"), w = $("live-warning");
    if (s) s.disabled = !!running;
    if (t) t.disabled = !running;
    if (w) w.hidden = !running;
  }

  function refreshLiveStatus() {
    return api("/api/live/status").then(function (st) {
      setLiveButtons(!!st.running);
      var srcEl = $("source-label");
      if (srcEl && !st.running) {
        srcEl.textContent = "Source: mock";
        srcEl.className = "pill pill-idle";
      }
      if (st.error) showError("live: " + st.error);
      return st;
    }).catch(function () { /* status is best-effort */ });
  }

  // --------------------------------------------------------------- wire up
  function init() {
    $("demo-start").onclick = function () {
      clearError();
      postJSON("/api/demo/start", { scenario: "flagship_chain" })
        .then(function (snap) { setLiveButtons(false); renderAll(snap); startPolling(); refreshLiveStatus(); })
        .catch(function (e) { showError(e.message); });
    };
    $("demo-reset").onclick = function () {
      clearError();
      stopPolling();
      postJSON("/api/demo/reset", { scenario: "flagship_chain" })
        .then(function (snap) { setLiveButtons(false); renderAll(snap); })
        .catch(function (e) { showError(e.message); });
    };
    $("live-start").onclick = function () {
      clearError();
      postJSON("/api/live/start", {}).then(function (res) {
        if (res.running) {
          setLiveButtons(true);
          startPolling();
          refreshLiveStatus();
        } else {
          showError("Could not start live YOLO: " + (res.error || "unknown error"));
          refreshLiveStatus();
        }
      }).catch(function (e) { showError(e.message); });
    };
    $("live-stop").onclick = function () {
      clearError();
      postJSON("/api/live/stop", {}).then(function () {
        stopPolling();
        setLiveButtons(false);
        refreshLiveStatus();
      }).catch(function (e) { showError(e.message); });
    };
    $("demo-refresh").onclick = function () {
      clearError();
      api("/api/snapshot").then(renderAll).catch(function (e) { showError(e.message); });
    };

    api("/api/snapshot").then(renderAll).catch(function () { /* first load: idle is fine */ });
    refreshLiveStatus();
    loadSmoke();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
