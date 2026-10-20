/* What are the odds? The calculator and the charts of the findings. Plain JavaScript, no build step: it reads
   the small JSON tables in ../data/ (made by `whiff-lab export` from Retrosheet's play-by-play). */
(function () {
  "use strict";
  const DATA = document.documentElement.dataset.data || "../data/";
  const MAX_DIFF = 10;
  const cache = {};
  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const get = (path) => cache[path] || (cache[path] = fetch(DATA + path).then((r) => {
    if (!r.ok) throw new Error(path + ": " + r.status);
    return r.json();
  }));
  const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
  const num = (x) => x.toLocaleString("en-US");
  const MONTHS = ["Jan.", "Feb.", "March", "April", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."];
  const ordinal = (i) => i + (i % 100 >= 11 && i % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[i % 10] || "th"));
  const oneIn = (p) => {
    const n = 1 / p;
    const r = n >= 10000 ? Math.round(n / 1000) * 1000 : n >= 1000 ? Math.round(n / 100) * 100 : Math.round(n / 10) * 10;
    return "1 in " + num(r);
  };

  /* ---------- teams and games ---------- */
  let TEAMS = {};
  const team = (code, year) => {
    const span = (TEAMS[code] || []).find((s) => s[0] <= year && year <= s[1]);
    return span ? span[2] : code;
  };
  const gameDate = (gid) => {
    const y = +gid.slice(3, 7), m = +gid.slice(7, 9), d = +gid.slice(9, 11);
    return MONTHS[m - 1] + " " + d + ", " + y;
  };
  const boxScore = (gid) => {
    const y = gid.slice(3, 7);
    return "https://www.retrosheet.org/boxesetc/" + y + "/B" + gid.slice(7, 11) + gid.slice(11, 12) + gid.slice(0, 3) + y + ".htm";
  };

  /* ---------- the calculator ---------- */
  const calc = $("#odds-calc");
  if (calc) setupCalculator();

  function setupCalculator() {
    const S = { inning: 9, bottom: 1, outs: 0, bases: 0, away: 3, home: 2, era: "runner" };
    const fromHash = new URLSearchParams(location.hash.slice(1));
    const int = (k, lo, hi) => { const v = parseInt(fromHash.get(k), 10); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; };
    for (const [k, lo, hi] of [["inning", 1, 10], ["bottom", 0, 1], ["outs", 0, 2], ["bases", 0, 7], ["away", 0, 30], ["home", 0, 30]]) {
      const v = int(k, lo, hi); if (v !== null) S[k] = v;
    }
    let ERAS = [], LAST = 0, touched = false; // the address keeps the spot only once the reader picks one

    // buttons: each group writes one field of S
    calc.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => {
      const [k, v] = b.dataset.set.split("=");
      S[k] = +v; touched = true; update();
    }));
    calc.querySelectorAll("[data-base]").forEach((b) => b.addEventListener("click", () => {
      S.bases ^= +b.dataset.base; touched = true; update();
    }));
    calc.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => {
      const [k, d] = b.dataset.step.split(":");
      S[k] = Math.max(0, Math.min(30, S[k] + +d)); touched = true; update();
    }));
    const eraSel = $("#odds-era");
    eraSel.addEventListener("change", () => { S.era = eraSel.value; touched = true; update(); });

    Promise.all([get("findings.json"), get("teams.json")]).then(([f, teams]) => {
      TEAMS = teams; ERAS = f.eras; LAST = f.summary.last_season;
      eraSel.replaceChildren(...ERAS.map((e) => el("option", { value: e.key, text: e.name + ", " + e.first + "–" + Math.min(e.last, f.summary.last_season) })));
      if (fromHash.get("era") && ERAS.some((e) => e.key === fromHash.get("era"))) S.era = fromHash.get("era");
      update();
    }).catch(fail);

    function fail(err) {
      $("#odds-out").replaceChildren(el("p", { class: "odds-msg", text: "The tables did not load (" + err.message + "). Reload the page to try again." }));
    }

    let seq = 0;
    function update() {
      // the controls first, so a tap shows at once
      calc.querySelectorAll("[data-set]").forEach((b) => {
        const [k, v] = b.dataset.set.split("=");
        b.setAttribute("aria-pressed", String(S[k] === +v));
      });
      calc.querySelectorAll("[data-base]").forEach((b) => b.setAttribute("aria-pressed", String(!!(S.bases & +b.dataset.base))));
      $("#odds-away").textContent = S.away; $("#odds-home").textContent = S.home;
      eraSel.value = S.era;
      $("#odds-sit").textContent = situation(S);
      if (touched) history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(S).map(([k, v]) => [k, String(v)])).toString());
      const mine = ++seq;
      Promise.all([get("odds/" + S.inning + ".json"), get("games/" + S.era + "-" + S.inning + ".json")]).then(([odds, games]) => {
        if (mine === seq) render(odds, games);
      }).catch(fail);
    }

    function render(odds, games) {
      const diff = S.home - S.away;
      const d = Math.max(-MAX_DIFF, Math.min(MAX_DIFF, diff));
      const cell = ((S.bottom * 3 + S.outs) * 8 + S.bases) * (2 * MAX_DIFF + 1) + d + MAX_DIFF;
      const E = odds.eras[S.era];
      const era = ERAS.find((e) => e.key === S.era);
      const p = E.p[cell] / 1000, lo = E.lo[cell] / 1000, hi = E.hi[cell] / 1000, n = E.n[cell], w = E.w[cell];
      const out = $("#odds-out");
      $("#odds-home-p").textContent = pct(p);
      $("#odds-mini-p").textContent = pct(p);
      $("#odds-away-p").textContent = pct(1 - p);
      $("#odds-bar").style.setProperty("--p", (p * 100).toFixed(1) + "%");
      $("#odds-range").textContent = pct(lo) + " to " + pct(hi);
      $("#odds-n").textContent = num(n);
      $("#odds-w").textContent = n ? num(w) + " (" + pct(w / n) + ")" : "–";
      const notes = [];
      if (n === 0) notes.push("No game of this era reached this spot: the number comes from the model and the other eras alone.");
      else if (n < 30) notes.push("Few games of this era reached this spot, so the number leans on the model and the other eras.");
      if (Math.abs(diff) > MAX_DIFF) notes.push("Leads of more than 10 runs count as 10.");
      if (S.inning === 10 && S.era === "runner" && S.outs === 0 && !(S.bases & 2) && S.bases === 0) notes.push("Since 2020 every extra half-inning starts with a runner on second; set one to see the usual start.");
      $("#odds-notes").replaceChildren(...notes.map((t) => el("p", { text: t })));
      const list = $("#odds-games");
      list.replaceChildren();
      const ref = games[String(cell)] || {};
      for (const [side, label] of [["won", "The latest home win from here"], ["lost", "The latest home loss from here"]]) {
        const r = ref[side];
        if (!r) continue;
        const [gid, away, ar, hr] = r, year = +gid.slice(3, 7), num2 = gid.slice(11, 12);
        list.append(el("li", {},
          el("span", { class: "odds-games__k", text: label }),
          el("span", {}, gameDate(gid) + (num2 !== "0" ? " (game " + num2 + ")" : "") + ": " + team(away, year) + " " + ar + " at " + team(gid.slice(0, 3), year) + " " + hr + " · "),
          el("a", { href: boxScore(gid), rel: "noopener", text: "box score" })));
      }
      $("#odds-games-wrap").hidden = !list.children.length;
      // the same spot in every era
      const strip = $("#odds-strip");
      const rows = ERAS.map((e) => ({ e, p: odds.eras[e.key].p[cell] / 1000, lo: odds.eras[e.key].lo[cell] / 1000, hi: odds.eras[e.key].hi[cell] / 1000, n: odds.eras[e.key].n[cell] }));
      let a = Math.min(...rows.map((r) => r.lo)), b = Math.max(...rows.map((r) => r.hi));
      const span = Math.max(0.1, b - a);
      const mid = (a + b) / 2;
      a = Math.max(0, mid - span * 0.6); b = Math.min(1, mid + span * 0.6);
      const x = (v) => ((v - a) / (b - a) * 100).toFixed(2) + "%";
      strip.replaceChildren(...rows.map((r) => el("li", { class: r.e.key === S.era ? "is-now" : "" },
        el("span", { class: "odds-strip__era" }, el("strong", { text: r.e.name }), " " + r.e.first + "–" + Math.min(r.e.last, LAST)),
        el("span", { class: "odds-strip__track", "aria-hidden": "true" },
          el("span", { class: "odds-strip__range", style: "left:" + x(r.lo) + ";right:calc(100% - " + x(r.hi) + ")" }),
          el("span", { class: "odds-strip__dot", style: "left:" + x(r.p) })),
        el("span", { class: "odds-strip__v num" }, pct(r.p), el("small", { text: " " + (r.n ? num(r.n) + " games" : "no games") })))));
      $("#odds-strip-axis").textContent = "Scale: " + pct(a, 0) + " to " + pct(b, 0) + ". Lines: 90% intervals.";
      out.classList.toggle("is-thin", n < 30);
      $("#odds-era-note").textContent = era.name + ": " + era.note + ".";
    }
  }

  function situation(S) {
    const half = S.bottom ? "Bottom" : "Top";
    const inn = S.inning === 10 ? "10th or later" : ordinal(S.inning);
    const outs = ["no outs", "one out", "two outs"][S.outs];
    const on = [[1, "first"], [2, "second"], [4, "third"]].filter(([b]) => S.bases & b).map(([, n]) => n);
    const runners = !on.length ? "bases empty" : on.length === 3 ? "bases loaded" : on.length === 1 ? "runner on " + on[0] : "runners on " + on[0] + " and " + on[1];
    const d = S.home - S.away;
    const score = d === 0 ? "tied " + S.away + "–" + S.home : (d > 0 ? "home team leads " : "home team trails ") + Math.max(S.home, S.away) + "–" + Math.min(S.home, S.away);
    return half + " of the " + inn + ", " + outs + ", " + runners + ", " + score;
  }

  /* ---------- charts of the findings ---------- */
  const charts = document.querySelectorAll("[data-chart]");
  if (charts.length) {
    get("findings.json").then((f) => {
      const draw = () => charts.forEach((c) => CHARTS[c.dataset.chart] && CHARTS[c.dataset.chart](c, f));
      draw();
      let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 150); });
      fillTables(f);
    }).catch(() => charts.forEach((c) => c.replaceChildren(el("p", { class: "lab-note", text: "The chart did not load." }))));
  }

  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  function frame(box, height, xs, ys, fmtY) {
    const W = Math.max(280, box.clientWidth), H = height, m = { l: 40, r: 12, t: 12, b: 26 };
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    const X = (v) => m.l + (v - xs[0]) / (xs[1] - xs[0]) * (W - m.l - m.r);
    const Y = (v) => H - m.b - (v - ys[0]) / (ys[1] - ys[0]) * (H - m.t - m.b);
    for (const t of ys[2]) {
      svg.append(s("line", { class: "ch-grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      svg.append(s("text", { class: "ch-tick", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, fmtY(t)));
    }
    const step = W < 500 ? 25 : 10;
    for (let yr = Math.ceil(xs[0] / step) * step; yr <= xs[1]; yr += step) {
      svg.append(s("text", { class: "ch-tick", x: X(yr), y: H - 8, "text-anchor": "middle" }, String(yr)));
    }
    svg.append(s("line", { class: "ch-axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    return { svg, X, Y, W, H, m };
  }
  const path = (pts) => pts.map(([x, y], i) => (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1)).join("");
  const smooth = (vals, k) => vals.map((_, i) => {
    const w = vals.slice(Math.max(0, i - k), i + k + 1).filter((v) => v != null);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });

  const CHARTS = {
    // the home team's share of wins, every season since 1901
    homefield(box, f) {
      const S = f.seasons;
      const yrs = S.map((r) => r.year), hw = S.map((r) => r.home_win);
      const { svg, X, Y } = frame(box, 240, [yrs[0], yrs[yrs.length - 1]], [0.48, 0.6, [0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]], (t) => Math.round(t * 100) + "%");
      S.forEach((r) => svg.append(s("circle", { class: "ch-dot" + (r.year === 2020 ? " is-now" : ""), cx: X(r.year), cy: Y(r.home_win), r: r.year === 2020 ? 4 : 2.4 })));
      svg.append(s("path", { class: "ch-line is-accent", d: path(smooth(hw, 5).map((v, i) => [X(yrs[i]), Y(v)])) }));
      const y20 = S.find((r) => r.year === 2020);
      if (y20) svg.append(s("text", { class: "ch-label", x: X(2020) - 6, y: Y(y20.home_win) - 8, "text-anchor": "end" }, "2020, empty stands: " + pct(y20.home_win)));
      box.replaceChildren(svg);
    },
    // the share of games won by a team that was below 10% at some point, and the runs per game, by season
    comebacks(box, f) {
      const S = f.seasons.filter((r) => r.under10 != null);
      const yrs = S.map((r) => r.year);
      const top = frame(box, 200, [yrs[0], yrs[yrs.length - 1]], [0.04, 0.12, [0.04, 0.06, 0.08, 0.1, 0.12]], (t) => Math.round(t * 100) + "%");
      top.svg.append(s("path", { class: "ch-line is-accent", d: path(S.map((r) => [top.X(r.year), top.Y(r.under10)])) }));
      top.svg.append(s("text", { class: "ch-label", x: top.m.l + 6, y: top.m.t + 10 }, "Games won from below 10%"));
      const bot = frame(box, 170, [yrs[0], yrs[yrs.length - 1]], [3, 6, [3, 4, 5, 6]], (t) => String(t));
      bot.svg.append(s("path", { class: "ch-line", d: path(S.map((r) => [bot.X(r.year), bot.Y(r.rpg)])) }));
      bot.svg.append(s("text", { class: "ch-label", x: bot.m.l + 6, y: bot.m.t + 10 }, "Runs per team per game"));
      box.replaceChildren(top.svg, bot.svg);
    },
    // a tie game at the start of each half-inning: the home team's chance
    ties(box, f) {
      const before = f.tie_by_inning.find((t) => t.rule === "before"), runner = f.tie_by_inning.find((t) => t.rule === "runner");
      const W = Math.max(280, box.clientWidth), H = 220, m = { l: 40, r: 12, t: 14, b: 26 };
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
      const ys = [0.45, 0.9], Y = (v) => H - m.b - (v - ys[0]) / (ys[1] - ys[0]) * (H - m.t - m.b);
      for (const t of [0.5, 0.6, 0.7, 0.8, 0.9]) {
        svg.append(s("line", { class: t === 0.5 ? "ch-axis" : "ch-grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
        svg.append(s("text", { class: "ch-tick", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, Math.round(t * 100) + "%"));
      }
      const slot = (W - m.l - m.r) / 11, bw = Math.min(14, slot / 3.2);
      for (let i = 0; i < 10; i++) {
        const cx = m.l + slot * (i + 0.8);
        const bars = [[before.top[i], -1, ""], [before.bottom[i], 1, " is-now"]];
        for (const [v, side, cls] of bars) {
          svg.append(s("rect", { class: "ch-bar" + cls, x: cx + (side < 0 ? -bw : 0), y: Y(v), width: bw - 1, height: Y(0.45) - Y(v), style: cls ? "fill:var(--leather);fill-opacity:.85" : "fill-opacity:.35" }));
        }
        svg.append(s("text", { class: "ch-tick", x: cx, y: H - m.b + 14, "text-anchor": "middle" }, i === 9 ? "10+" : String(i + 1)));
      }
      // the runner on second: the 10th since 2020
      const cx = m.l + slot * 10.6;
      svg.append(s("rect", { x: cx - bw, y: Y(runner.top[9]), width: bw - 1, height: Y(0.45) - Y(runner.top[9]), style: "fill:var(--graphite);fill-opacity:.35;stroke:var(--graphite);stroke-dasharray:2 2" }));
      svg.append(s("rect", { x: cx, y: Y(runner.bottom[9]), width: bw - 1, height: Y(0.45) - Y(runner.bottom[9]), style: "fill:var(--leather);fill-opacity:.45;stroke:var(--leather-text);stroke-dasharray:2 2" }));
      svg.append(s("text", { class: "ch-tick", x: cx, y: H - m.b + 14, "text-anchor": "middle" }, "10+*"));
      box.replaceChildren(svg, el("figcaption", { text: "A tie game at the start of each half-inning, by inning. Grey: top (visitors bat). Orange: bottom (home team bats). 10+*: extra innings since 2020, which start with a runner on second." }));
    },
  };

  /* ---------- tables of the findings ---------- */
  function fillTables(f) {
    const cb = $("#comebacks-body");
    if (cb) {
      cb.replaceChildren(...f.comebacks.slice(0, 15).map((c, i) => el("tr", {},
        el("td", { class: "num", text: String(i + 1) }),
        el("td", { class: "nw", text: gameDate(c.game_id) }),
        el("td", {}, el("strong", { text: c.winner }), " beat " + c.loser + " " + Math.max(c.away_runs, c.home_runs) + "–" + Math.min(c.away_runs, c.home_runs), el("br"), el("small", { text: "from " + c.moment })),
        el("td", { class: "num", text: oneIn(c.low) }),
        el("td", {}, el("a", { href: boxScore(c.game_id), rel: "noopener", text: "box score" })))));
    }
    const sp = $("#spots-body");
    if (sp) {
      const head = $("#spots-head");
      head.replaceChildren(el("th", { text: "Spot" }), ...f.eras.map((e) => el("th", { class: "num", title: e.name, text: e.first + "–" + (e.last > 2050 ? "" : String(e.last).slice(2)) })));
      sp.replaceChildren(...f.spots.map((r) => el("tr", {}, el("th", { text: r.name }),
        ...f.eras.map((e) => { const v = r.eras[e.key]; return el("td", { class: "num", title: "90%: " + pct(v.lo) + " to " + pct(v.hi) + ", " + num(v.n) + " games", text: pct(v.p) }); }))));
    }
    document.querySelectorAll("[data-fact]").forEach((n) => {
      const v = n.dataset.fact.split(".").reduce((o, k) => (o == null ? o : o[k]), f);
      if (v != null) n.textContent = n.dataset.fmt === "pct" ? pct(v) : n.dataset.fmt === "pct0" ? pct(v, 0) : typeof v === "number" ? num(v) : v;
    });
  }
})();
