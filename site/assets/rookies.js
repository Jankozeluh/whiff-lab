/* Rookie Report: the latest rookie classes, each rookie's chance to become a star, the most similar rookies
   since 1932 and how much of his season was luck. Reads ../data/rookies/ (made by `whiff-lab rookies`). */
(function () {
  "use strict";
  const DATA = (document.documentElement.dataset.data || "../data/") + "rookies/";
  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null) continue;
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null && k !== false) n.append(k);
    return n;
  };
  const cache = {};
  const get = (name) => cache[name] || (cache[name] = fetch(DATA + name).then((r) => {
    if (!r.ok) throw new Error(name + ": " + r.status);
    return r.json();
  }));
  const pct = (p) => (p < 0.01 ? "<1%" : Math.round(p * 100) + "%");
  const avg = (x) => x.toFixed(3).replace(/^0/, "");
  const bref = (id) => id ? "https://www.baseball-reference.com/players/" + id[0] + "/" + id + ".shtml" : null;
  const S = { year: null, kind: "hitter", find: "" };

  get("index.json").then((ix) => {
    S.year = ix.classes[0];
    ages(ix.age_curve);
    const seg = $("#rk-class");
    seg.replaceChildren(...ix.classes.map((y) => el("button", { class: "odds-btn", type: "button", "data-year": y, text: y + " rookies" })));
    seg.querySelectorAll("[data-year]").forEach((b) => b.addEventListener("click", () => { S.year = b.dataset.year; render(); }));
    document.querySelectorAll("[data-kind]").forEach((b) => b.addEventListener("click", () => { S.kind = b.dataset.kind; render(); }));
    $("#rk-find").addEventListener("input", (e) => { S.find = e.target.value.trim().toLowerCase(); render(); });
    render();
  }).catch((e) => { $("#rk-list").replaceChildren(el("li", { class: "odds-msg", text: "The report did not load (" + e.message + ")." })); });

  // the finding: the share of rookies who became stars, by age
  function ages(curve) {
    const box = $("#rk-ages");
    const max = Math.max(...curve.hitter.map((r) => r.rate), ...curve.pitcher.map((r) => r.rate));
    box.replaceChildren(...[["hitter", "Hitters"], ["pitcher", "Pitchers"]].map(([k, label]) =>
      el("div", { class: "rk-ages__col" }, el("p", { class: "rk-ages__h", text: label }),
        el("ol", { class: "rk-ages__bars" }, ...curve[k].map((r) => el("li", {},
          el("span", { class: "rk-ages__age", text: r.age === 29 ? "29+" : String(r.age) }),
          el("span", { class: "rk-ages__track" }, el("span", { class: "rk-ages__bar", style: "width:" + (r.rate / max * 100).toFixed(1) + "%" })),
          el("span", { class: "rk-ages__v" }, Math.round(r.rate * 100) + "%", el("small", { text: " of " + r.rookies.toLocaleString("en-US") }))))))));
  }

  let seq = 0;
  function render() {
    document.querySelectorAll("[data-year]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.year === S.year)));
    document.querySelectorAll("[data-kind]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.kind === S.kind)));
    const mine = ++seq;
    get(S.year + ".json").then((c) => {
      if (mine !== seq) return;
      const all = c[S.kind] || [];
      const rows = all.map((r, i) => [r, i + 1]).filter(([r]) => !S.find || (r.name + " " + (r.team || "")).toLowerCase().includes(S.find));
      $("#rk-list").replaceChildren(...rows.slice(0, S.find ? 200 : 40).map(([r, rank]) => item(r, rank)));
      if (!rows.length) $("#rk-list").append(el("li", { class: "odds-msg", text: "No rookie of " + S.year + " matches." }));
      const shown = Math.min(rows.length, S.find ? 200 : 40);
      $("#rk-count").textContent = (S.find ? shown + " of " + all.length : "The top " + shown + " of " + all.length) + " " + (S.kind === "hitter" ? "hitters" : "pitchers") + " who were rookies in " + S.year + (S.find ? "" : ". Search to find the others.");
    });
  }

  function line(r) {
    const l = r.line;
    if (S.kind === "hitter") return [avg(l.avg) + " / " + avg(l.obp) + " / " + avg(l.slg) + ", " + l.hr + " HR", l.pa + " PA · OPS+ " + l.ops_plus];
    return [l.era.toFixed(2) + " ERA, " + l.so + " SO", Math.round(l.ip) + " IP · " + (l.gs ? l.gs + " starts" : l.sv + " saves") + " · ERA− " + l.era_minus];
  }

  function luck(r) {
    const a = r.luck.actual, x = r.luck.expected;
    if (S.kind === "hitter") {
      const d = a - x;
      const verdict = Math.abs(d) < 0.008 ? "about what his skills earned" : d > 0 ? "luckier than his skills" : "unluckier than his skills";
      return "Luck: he hit " + avg(a) + "; with his strikeouts, power and an ordinary share of hits on balls in play, about " + avg(x) + " (" + verdict + ").";
    }
    const d = a - x;
    const verdict = Math.abs(d) < 0.25 ? "about what his pitching earned" : d < 0 ? "luckier than his pitching" : "unluckier than his pitching";
    return "Luck: his ERA was " + a.toFixed(2) + "; his strikeouts, walks and home runs with ordinary luck point to about " + x.toFixed(2) + " (" + verdict + ").";
  }

  function comp(c) {
    const l = c.line;
    const stat = S.kind === "hitter" ? avg(l.avg) + ", " + l.hr + " HR, OPS+ " + l.ops_plus + " in " + l.pa + " PA"
      : l.era.toFixed(2) + " ERA, " + l.so + " SO in " + l.ip + " IP";
    const after = c.allstar_next >= 2 ? "became a star" : c.allstar_next === 1 ? "one All-Star team" : "no All-Star team";
    const name = c.bbref ? el("a", { href: bref(c.bbref), rel: "noopener", text: c.name }) : c.name;
    return el("li", { class: c.allstar_next >= 2 ? "is-star" : "" },
      el("span", {}, name, " " + c.year + ", " + c.age + (c.team ? ", " + c.team : "") + ": " + stat),
      el("b", { text: after + (c.allstar_next ? " (" + c.allstar_next + " in 9 seasons" + (c.allstar_career > c.allstar_next ? ", " + c.allstar_career + " in all" : "") + ")" : "") }));
  }

  function item(r, rank) {
    const [l1, l2] = line(r);
    const stars = r.comparables.filter((c) => c.allstar_next >= 2).length;
    return el("li", { class: "rk-item" }, el("details", {},
      el("summary", { class: "rk-sum" },
        el("span", { class: "rk-rank", text: String(rank) }),
        el("span", { class: "rk-who" }, el("strong", { text: r.name }), el("small", { text: [r.team, r.position, "age " + r.age].filter(Boolean).join(" · ") })),
        el("span", { class: "rk-line" }, l1, el("small", { text: l2 })),
        el("span", { class: "rk-chance", "aria-label": "Chance to become a star: " + pct(r.p) },
          el("b", { text: pct(r.p) }), el("i", { style: "--p:" + Math.max(1, r.p * 100).toFixed(1) + "%" }))),
      el("div", { class: "rk-more" },
        r.abroad ? el("p", { class: "rk-flag", text: "Came to MLB from a top league in " + r.born + ", where he already played as a pro: the model's age penalty is unfair to him, so read his chance as too low." }) : null,
        el("p", { text: luck(r) }),
        el("p", { text: "Rookie of the Year voting: " + (r.roy_share ? Math.round(r.roy_share * 100) + "% of the points" : "no votes") + ". All-Star teams since: " + r.allstar_so_far + "." }),
        el("h3", { text: "The five rookies most like him (" + stars + " of 5 became stars)" }),
        el("ol", { class: "rk-comps" }, ...r.comparables.map(comp)),
        r.bbref ? el("p", {}, el("a", { class: "lab-more", href: bref(r.bbref), rel: "noopener", text: "His page on Baseball-Reference" })) : null)));
  }
})();
