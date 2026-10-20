"""Build the site's pages and the lab notes from their sources, with every number taken from the results.

Sources (edit these):  pages/*.html      the site's pages, without the header and the footer
                       notes/*.md        the lab notes
Built (never by hand): site/**/*.html    every page in the Daily Whiff look
                       lab-notes/*.md    the notes with their numbers filled in, as GitHub shows them
                       lab-notes/figures/*.svg, site/notes/figures/*.svg

In a source, {{ h.table.ll.all | f4 }} is a number from the results (see `context()`): the notes say what the
latest rebuild found, every year, without anyone editing them.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import markdown

from whiff_lab.data import ROOT

PAGES, NOTES_SRC = ROOT / "pages", ROOT / "notes"
SITE, NOTES_OUT = ROOT / "site", ROOT / "lab-notes"
RESULTS = ROOT / "results"
REPO = "https://github.com/Jankozeluh/whiff-lab"
RETROSHEET_NOTICE = (
    "The information used here was obtained free of charge from and is copyrighted by Retrosheet.  "
    'Interested parties may contact Retrosheet at "www.retrosheet.org".'
)
SHORT = {
    "Home share only": "home",
    "Lookup table": "table",
    "Shrunk table (empirical Bayes)": "shrunk",
    "Logistic regression": "logistic",
    "Gradient boosting": "boosted",
}
GROUPS = {
    "all": "all",
    "rare states (seen < 100 times in training)": "rare",
    "ninth inning": "ninth",
    "extra innings, 2017-2019": "x17",
    "extra innings since 2020 (runner on second)": "x20",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def context() -> dict:
    """Everything a page or a note may quote."""
    f = load("findings")
    hold, rule, eras = load("holdout"), load("rule_change"), load("era_holdout")
    h = {
        "re": {int(k): v for k, v in hold["run_expectancy"].items()},
        "train": hold["train"],
        "test": hold["test"],
        "games": hold["test_games"],
        "pas": hold["test_plate_appearances"],
        "share": {GROUPS[k]: v for k, v in hold["groups"].items()},
    }
    for m in hold["models"]:
        key = SHORT[m["name"]]
        boot = m["bootstrap"]
        h[key] = {
            "name": m["name"],
            "ll": {GROUPS[k]: v for k, v in m["log_loss"].items()},
            "brier": {GROUPS[k]: v for k, v in m["brier"].items()},
            "ci": boot["log_loss_95"],
            "vs": next(v for k, v in boot.items() if k.startswith("vs_")),
            "k": m.get("prior_strengths", {}),
            "coef": m.get("coefficients", {}),
        }
    for key in ("shrunk", "logistic", "boosted"):  # how much better than the table, as a share of its loss
        for g in ("all", "rare", "x20"):
            h[key].setdefault("gain", {})[g] = 1 - h[key]["ll"][g] / h["table"]["ll"][g]
    h["table"]["gain_over_home"] = 1 - h["table"]["ll"]["all"] / h["home"]["ll"]["all"]
    r = {"games": rule["games"], "pas": rule["plate_appearances"]}
    for m in rule["models"]:
        r[SHORT[m["name"]]] = {"ll": m["log_loss"], "brier": m["brier"]}
    r["boosted"]["gain"] = 1 - r["boosted"]["ll"] / r["table"]["ll"]
    e = {row["era"]: row for row in eras["eras"]}
    x = {row["rule"]: row for row in f["extra_innings"]}
    tie = {t["rule"]: t for t in f["tie_by_inning"]}
    seasons = {row["year"]: row for row in f["seasons"]}
    first = f["summary"]["first_season"]
    played = [row for row in f["seasons"] if row["year"] >= first]
    cov = {
        "used": sum(row["games_pbp"] or 0 for row in played),
        "all": sum(row["games_all"] for row in played),
    }
    cov["share"] = cov["used"] / cov["all"]
    ks = [era["k"] for era in f["eras"]]
    return {
        "nseasons": f["summary"]["last_season"] - f["summary"]["first_season"] + 1,
        "ek": {"min": min(ks), "max": max(ks)},
        "cov": cov,
        "s": f["summary"],
        "f": f,
        "h": h,
        "r": r,
        "e": e,
        "x": x,
        "tie": tie,
        "cb": f["comebacks"],
        "seasons": seasons,
        "eras": {era["key"]: era for era in f["eras"]},
        "notice": RETROSHEET_NOTICE,
    }


MONTHS = ("Jan.", "Feb.", "March", "April", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.")


def fmt(value, how: str) -> str:
    if how == "year":
        return str(int(value))
    if how == "date":  # 1952-06-29 -> June 29, 1952
        y, m, d = (int(x) for x in str(value).split("-"))
        return f"{MONTHS[m - 1]} {d}, {y}"
    if how == "int":
        return f"{int(round(value)):,}"
    if how.startswith("pct"):
        return f"{value * 100:.{int(how[3:] or 1)}f}%"
    if how.startswith("f"):
        return f"{value:.{int(how[1:])}f}"
    if how == "onein":
        n = 1 / value
        step = 1000 if n >= 10000 else 100 if n >= 1000 else 10
        return f"1 in {int(round(n / step) * step):,}"
    if how == "abs_pct1":
        return f"{abs(value) * 100:.1f}%"
    if how == "pp":
        return f"{value * 100:.1f}"
    return str(value)


def fill(text: str, ctx: dict) -> str:
    def one(m: re.Match) -> str:
        path, _, how = (p.strip() for p in m.group(1).partition("|"))
        v = ctx
        for part in path.split("."):
            # lists by position, seasons by the year, everything else by name
            v = v[int(part)] if isinstance(v, list) or part.isdigit() and int(part) in v else v[part]
        return fmt(v, how or "raw")

    return re.sub(r"\{\{(.+?)\}\}", one, text)


NAV = (
    ("odds/", "What are the odds?"),
    ("notes/", "Lab notes"),
    ("https://thedailywhiff.com/", "The Daily Whiff"),
)
MENU = (
    '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    '<path d="M4 7.5h16M4 12h16M4 16.5h16" stroke="currentColor" stroke-width="1.8" fill="none"/></svg>'
)
CLOSE = (
    '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    '<path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="1.8" fill="none"/></svg>'
)


def asset(root: str, name: str) -> str:
    """An asset's address with a short hash of its content, so a browser never keeps an old copy."""
    digest = hashlib.sha256((SITE / "assets" / name).read_bytes()).hexdigest()[:10]
    return f"{root}assets/{name}?v={digest}"


def layout(
    body: str, *, title: str, description: str, root: str, current: str = "", css: tuple = (), js: tuple = ()
) -> str:
    nav = "".join(
        f'<li><a href="{href if href.startswith("http") else root + href}"'
        f"{' aria-current="page"' if href == current else ''}>{label}</a></li>"
        for href, label in NAV
    )
    links = "".join(f'<link rel="stylesheet" href="{asset(root, c)}">' for c in css)
    scripts = "".join(f'<script src="{asset(root, j)}" defer></script>' for j in js)
    return f"""<!doctype html>
<html lang="en" data-data="{root}data/">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="icon" href="{root}assets/whiff-site-icon.png">
<link rel="preload" href="{root}assets/fonts/archivo-latin.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="{asset(root, "lab.css")}">{links}
<script>document.documentElement.classList.add("is-js")</script>
</head>
<body>
<header class="lab-header"><div class="wrap lab-header__row">
<a class="lab-brand" href="{root or "./"}"><img src="{root}assets/whiff-logo-chalk.png" alt="" width="70" height="30"><span>Whiff Lab</span></a>
<nav class="lab-nav" aria-label="Main">
<button class="lab-nav__open" type="button" aria-expanded="false" aria-controls="lab-menu" aria-label="Open menu">{MENU}</button>
<div class="lab-nav__panel" id="lab-menu">
<button class="lab-nav__close" type="button" aria-label="Close menu">{CLOSE}</button>
<ul>{nav}</ul>
</div>
</nav>
</div></header>
<div class="lab-seam"></div>
<main class="wrap">
{body}
</main>
<footer class="lab-footer"><div class="wrap">
<div class="lab-footer__cols">
<div><img src="{root}assets/whiff-logo-chalk.png" alt="Whiff Lab" width="60" height="26">
<p class="lab-footer__tagline">Open baseball models from free play-by-play, with the data, the steps and the tests in the open.</p></div>
<div><p class="lab-footer__label">Lab</p><ul class="lab-footer__links">
<li><a href="{root}odds/">What are the odds?</a></li><li><a href="{root}notes/">Lab notes</a></li>
<li><a href="{REPO}">Code and data on GitHub</a></li></ul></div>
<div><p class="lab-footer__label">Read</p><ul class="lab-footer__links">
<li><a href="https://thedailywhiff.com/">The Daily Whiff</a></li>
<li><a href="https://thedailywhiff.com/the-whiff/">The Whiff, a daily puzzle</a></li></ul></div>
</div>
<div class="lab-legal">
<p>Whiff Lab, an independent fan project, not affiliated with Major League Baseball or its clubs · Data: Retrosheet's
official releases (through the Chadwick Baseball Bureau's mirror) · {html.escape(RETROSHEET_NOTICE)} · Typeface:
Archivo, SIL Open Font License.</p>
</div>
</div></footer>
<script src="{asset(root, "lab.js")}" defer></script>{scripts}
</body>
</html>
"""


PAGE_META = {
    "home.html": (
        "index.html",
        "Whiff Lab: Open Baseball Models From Free Data",
        "Baseball models built from free, openly licensed play-by-play, with the data, every step and "
        "an honest test in the open. First: win probability for any game state since 1908.",
        "",
        (),
    ),
    "odds.html": (
        "odds/index.html",
        "What Are the Odds? Win Probability for Any Baseball Game State Since 1908",
        "The home team's chance to win from any inning, out, runner and score, in every era since 1908: "
        "learned from 15 million plate appearances, with the real games behind each number.",
        "odds/",
        ("odds.css",),
    ),
}


def note_sources() -> list[Path]:
    return sorted(NOTES_SRC.glob("[0-9][0-9]-*.md"))


def title_of(md: str) -> str:
    return md.splitlines()[0].lstrip("# ").strip()


def lede_of(md: str) -> str:
    para = next(p for p in md.split("\n\n")[1:] if p.strip() and not p.startswith(("#", "!", "<")))
    return re.sub(r"[*_`]|\[([^\]]+)\]\([^)]+\)", r"\1", " ".join(para.split()))


def build_pages(ctx: dict) -> list[Path]:
    out = []
    for src, (dest, title, desc, current, css) in PAGE_META.items():
        body = fill((PAGES / src).read_text(), ctx)
        depth = dest.count("/")
        path = SITE / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        js = ("odds.js",) if src in ("odds.html",) else ()
        path.write_text(
            layout(body, title=title, description=desc, root="../" * depth, current=current, css=css, js=js)
        )
        out.append(path)
    return out


def build_notes(ctx: dict | None = None) -> list[Path]:
    from whiff_lab.figures import draw_all

    ctx = ctx or context()
    figures = draw_all(NOTES_OUT / "figures")
    (SITE / "notes" / "figures").mkdir(parents=True, exist_ok=True)
    for fig in figures:
        shutil.copy(fig, SITE / "notes" / "figures" / fig.name)
    out = build_pages(ctx)
    sources = note_sources()
    items = []
    for i, src in enumerate(sources):
        md = fill(src.read_text(), ctx)
        slug = src.stem
        header = (
            f"<!-- Built by `whiff-lab notes` from notes/{src.name}: edit that file, not this one. -->\n\n"
        )
        # SITE/ in a note is the lab's site: from lab-notes/ on GitHub that is ../site/, from site/notes/ it is ../
        (NOTES_OUT / f"{slug}.md").write_text(header + md.replace("](SITE/", "](../site/"))
        md = md.replace("](SITE/", "](../")
        title = title_of(md)
        body_md = "\n".join(md.splitlines()[1:])
        # links between notes: GitHub reads 02-x.md, the site 02-x.html
        body_md = re.sub(r"\]\((\d\d-[a-z0-9-]+)\.md(#[^)]*)?\)", r"](\1.html\2)", body_md)
        body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "attr_list"])
        body = body.replace('<img alt="', '<img loading="lazy" alt="')
        prev = sources[i - 1] if i else None
        nxt = sources[i + 1] if i + 1 < len(sources) else None
        nav = '<nav class="lab-notes-nav" aria-label="Lab notes">'
        nav += (
            f'<a href="{prev.stem}.html">← {html.escape(title_of(prev.read_text()))}</a>'
            if prev
            else "<span></span>"
        )
        nav += (
            f'<a href="{nxt.stem}.html">{html.escape(title_of(nxt.read_text()))} →</a>'
            if nxt
            else "<span></span>"
        )
        nav += "</nav>"
        page = (
            f'<p class="lab-kicker">Lab note {i + 1} of {len(sources)} · What are the odds?</p>\n'
            f'<h1 class="lab-title">{html.escape(title)}</h1>\n<div class="lab-prose">\n{body}\n</div>\n{nav}'
        )
        path = SITE / "notes" / f"{slug}.html"
        path.write_text(
            layout(
                page,
                title=f"{title} · Whiff Lab notes",
                description=lede_of(md),
                root="../",
                current="notes/",
            )
        )
        out.append(path)
        items.append((slug, title, lede_of(md)))
    listing = "".join(
        f'<li><a href="{s}.html">{i + 1}. {html.escape(t)}</a><p>{html.escape(d)}</p></li>'
        for i, (s, t, d) in enumerate(items)
    )
    index = (
        '<h1 class="lab-title">Lab notes</h1>\n<p class="lab-intro">How "What are the odds?" was built, one '
        "step at a time, in plain words: the data, four models from the plainest to the strongest, the honest "
        "test that compares them, and what the eras change.</p>\n"
        f'<ol class="lab-notes-list">{listing}</ol>\n'
        f'<p class="lab-credit">The same notes are in the repository, with the code that makes every number: '
        f'<a href="{REPO}/tree/main/lab-notes">lab-notes/ on GitHub</a>.</p>'
    )
    path = SITE / "notes" / "index.html"
    path.write_text(
        layout(
            index,
            title="Lab notes · Whiff Lab",
            description="How the win probability models were built and tested, step by step.",
            root="../",
            current="notes/",
        )
    )
    out.append(path)
    readme = [
        "# Lab notes",
        "",
        "How *What are the odds?* was built, one step at a time. Each note is built from "
        "`notes/` with the numbers of the latest rebuild filled in.",
        "",
    ]
    readme += [f"{i + 1}. [{t}]({s}.md): {d}" for i, (s, t, d) in enumerate(items)]
    (NOTES_OUT / "README.md").write_text("\n".join(readme) + "\n")
    return out
