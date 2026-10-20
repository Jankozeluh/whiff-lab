"""Write the small tables the site reads (site/data/) and the numbers the lab notes quote (results/).

site/data/odds/<inning>.json   every state of one inning, every era: the odds, a 90% interval, the record
site/data/games/<era>-<inning>.json   for each of those states, the latest home win and home loss from it
site/data/teams.json           what to call each Retrosheet team code in each season
site/data/findings.json        the series and lists behind the findings
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from whiff_lab.data import RAW, ROOT, WORK
from whiff_lab.features import ERAS, MAX_DIFF, MAX_INNING, counts
from whiff_lab.lab import (
    comebacks,
    era_table,
    extra_innings,
    final_model,
    grid,
    grid_index,
    home_field,
    predictions_for_every_plate_appearance,
    summary,
    tail_calibration,
)

SITE_DATA = ROOT / "site" / "data"
RESULTS = ROOT / "results"
CELLS_PER_INNING = 2 * 3 * 8 * (2 * MAX_DIFF + 1)


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def era_case(column: str = "g.year") -> str:
    return "case " + " ".join(f"when {column} <= {e.last} then '{e.key}'" for e in ERAS) + " end"


def team_labels() -> dict[str, list]:
    """Retrosheet's team codes as places: 'Boston', or 'Chicago (AL)' when a city had two teams that year."""
    info = {}
    with open(RAW / "reference" / "teams.csv", newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            info[r["TEAM"]] = (r["CITY"].replace("St.Louis", "St. Louis"), r["LEAGUE"])
    seasons = duckdb.sql(f"""
        select distinct year, team from (
            select year, home as team from '{WORK / "games.parquet"}' union
            select year, away as team from '{WORK / "games.parquet"}')
    """).df()
    seasons["city"] = seasons.team.map(lambda t: info.get(t, (t, ""))[0])
    shared = seasons.groupby(["year", "city"]).team.transform("size") > 1
    seasons["label"] = np.where(
        shared, seasons.city + " (" + seasons.team.map(lambda t: info.get(t, ("", ""))[1]) + ")", seasons.city
    )
    out: dict[str, list] = {}
    for team, g in seasons.sort_values("year").groupby("team"):
        spans: list = []
        for year, label in zip(g.year, g.label, strict=True):
            if spans and spans[-1][2] == label and spans[-1][1] == year - 1:
                spans[-1][1] = year
            else:
                spans.append([int(year), int(year), label])
        out[team] = spans
    return out


def label_of(labels: dict[str, list], team: str, year: int) -> str:
    return next((s[2] for s in labels.get(team, []) if s[0] <= year <= s[1]), team)


def examples() -> pd.DataFrame:
    """For each era and state: the latest game the home team won from it, and the latest it lost."""
    idx = (
        f"((((least(p.inning, {MAX_INNING}) - 1) * 2 + p.bottom::int) * 3 + p.outs) * 8"
        f" + p.b1::int + 2 * p.b2::int + 4 * p.b3::int) * {2 * MAX_DIFF + 1}"
        f" + greatest(-{MAX_DIFF}, least({MAX_DIFF}, p.home_score - p.away_score)) + {MAX_DIFF}"
    )
    return duckdb.sql(f"""
        select {era_case()} as era, {idx} as idx, g.home_win,
               max_by(g.game_id, g.date || g.game_id) as game_id
        from '{WORK / "pa.parquet"}' p join '{WORK / "games.parquet"}' g using (game_id)
        where g.used
        group by all
    """).df()


def odds_files(table: pd.DataFrame, ex: pd.DataFrame, games: pd.DataFrame) -> None:
    """The calculator's data: one file per inning with every era, and the example games per era and inning."""
    g = grid()
    assert (grid_index(g.inning, g.bottom, g.outs, g.bases, g["diff"]) == np.arange(len(g))).all()
    by_era = {e: t.reset_index(drop=True) for e, t in table.groupby("era")}
    game_info = games.set_index("game_id")
    for inning in range(1, MAX_INNING + 1):
        cells = slice((inning - 1) * CELLS_PER_INNING, inning * CELLS_PER_INNING)
        out = {"inning": inning, "eras": {}}
        for era in ERAS:
            t = by_era[era.key].iloc[cells]
            out["eras"][era.key] = {
                "p": np.rint(t.p * 1000).astype(int).tolist(),
                "lo": np.rint(t.lo * 1000).astype(int).tolist(),
                "hi": np.rint(t.hi * 1000).astype(int).tolist(),
                "n": t.n.astype(int).tolist(),
                "w": t.w.astype(int).tolist(),
            }
            e = ex[(ex.era == era.key) & (ex.idx >= cells.start) & (ex.idx < cells.stop)]
            refs = {}
            for side, name in ((1, "won"), (0, "lost")):
                for i, gid in zip(e[e.home_win == side].idx, e[e.home_win == side].game_id, strict=True):
                    gi = game_info.loc[gid]
                    refs.setdefault(int(i) - cells.start, {})[name] = [
                        gid,
                        gi.away,
                        int(gi.away_runs),
                        int(gi.home_runs),
                    ]
            write(SITE_DATA / "games" / f"{era.key}-{inning}.json", refs)
        write(SITE_DATA / "odds" / f"{inning}.json", out)


def describe(row) -> str:
    """The winner's low point in words: 'down 13-3, top of the 7th, two outs, bases empty'."""
    home_won = row.home_win == 1
    own, other = (row.home_score, row.away_score) if home_won else (row.away_score, row.home_score)
    inning = int(row.real_inning)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(inning if inning < 20 else inning % 10, "th")
    half = "bottom" if row.bottom else "top"
    outs = ("no outs", "one out", "two outs")[int(row.outs)]
    on = [b for b, x in (("first", row.b1), ("second", row.b2), ("third", row.b3)) if x]
    runners = (
        "bases empty"
        if not on
        else "bases loaded"
        if len(on) == 3
        else f"runner on {on[0]}"
        if len(on) == 1
        else f"runners on {on[0]} and {on[1]}"
    )
    return f"down {other}–{own}, {half} of the {inning}{suffix}, {outs}, {runners}"


def findings(cb: pd.DataFrame, labels: dict, model) -> dict:
    hf = home_field()
    used = duckdb.sql(
        f"select year, count(*) as games from '{WORK / 'games.parquet'}' where used group by 1"
    ).df()
    rate = (
        cb.assign(under10=cb.win_p < 0.10, under1=cb.win_p < 0.01)
        .groupby("year")
        .agg(under10=("under10", "mean"), under1=("under1", "mean"))
        .reset_index()
    )
    seasons = hf.merge(rate, on="year", how="left").merge(
        used, on="year", how="left", suffixes=("_all", "_pbp")
    )
    top = []
    for r in cb.head(25).itertuples():
        winner, loser = (r.home, r.away) if r.home_win == 1 else (r.away, r.home)
        top.append(
            {
                "game_id": r.game_id,
                "date": r.date,
                "year": int(r.year),
                "away": label_of(labels, r.away, r.year),
                "home": label_of(labels, r.home, r.year),
                "away_runs": int(r.away_runs),
                "home_runs": int(r.home_runs),
                "winner": label_of(labels, winner, r.year),
                "loser": label_of(labels, loser, r.year),
                "winner_runs": int(max(r.away_runs, r.home_runs)),
                "loser_runs": int(min(r.away_runs, r.home_runs)),
                "low": float(r.win_p),
                "moment": describe(r),
            }
        )
    # a tie game at the start of each half-inning: what the home team's chance is (runner on second in the
    # 10th from 2020)
    g = grid()
    starts = g[(g.outs == 0) & (g["diff"] == 0) & (g.bases == 0)].copy()
    tie = []
    for runner in (0, 1):
        s = starts.copy()
        if runner:
            s.loc[s.inning == MAX_INNING, ["bases", "b2"]] = [2, 1]
        s["extra_runner"] = runner
        s["p"] = model.predict(s)
        tie.append(
            {
                "rule": "runner" if runner else "before",
                "top": s[s.bottom == 0].p.round(4).tolist(),
                "bottom": s[s.bottom == 1].p.round(4).tolist(),
            }
        )
    both = seasons.dropna(subset=["under10"])
    return {
        "comeback_rpg_r": float(np.corrcoef(both.under10, both.rpg)[0, 1]),
        "seasons": json.loads(seasons.round(4).to_json(orient="records")),
        "comebacks": top,
        "comeback_share": {
            "under10": float((cb.win_p < 0.10).mean()),
            "under1": float((cb.win_p < 0.01).mean()),
            "under_1_in_1000": int((cb.win_p < 0.001).sum()),
            "games": len(cb),
        },
        "extra_innings": json.loads(extra_innings().round(4).to_json(orient="records")),
        "tie_by_inning": tie,
        "eras": [
            {"key": e.key, "name": e.name, "first": e.first, "last": e.last, "note": e.note} for e in ERAS
        ],
    }


SPOTS = (
    ("First pitch", (1, 0, 0, 0, 0)),
    ("Home team down 1, bottom of the 9th, nobody on, nobody out", (9, 1, 0, 0, -1)),
    ("Home team up 1, top of the 9th, nobody on, nobody out", (9, 0, 0, 0, 1)),
    ("Tied, bottom of the 9th, nobody on, nobody out", (9, 1, 0, 0, 0)),
    ("Home team up 3, top of the 7th, nobody on, nobody out", (7, 0, 0, 0, 3)),
    ("Home team down 2, bottom of the 8th, bases loaded, one out", (8, 1, 1, 7, -2)),
)


def spots(table: pd.DataFrame) -> list[dict]:
    """A few famous spots in every era: the odds, the 90% interval and the number of games behind them."""
    out = []
    for name, (inning, bottom, outs, bases, diff) in SPOTS:
        i = int(grid_index(inning, bottom, outs, bases, diff))
        rows = table.groupby("era").nth(i).set_index("era")
        out.append(
            {
                "name": name,
                "cell": [inning, bottom, outs, bases, diff],
                "eras": {
                    e.key: {
                        "p": round(float(rows.p[e.key]), 4),
                        "lo": round(float(rows.lo[e.key]), 4),
                        "hi": round(float(rows.hi[e.key]), 4),
                        "n": int(rows.n[e.key]),
                    }
                    for e in ERAS
                },
            }
        )
    return out


def run() -> dict:
    c = counts()
    model = final_model(c)
    table = era_table(c, model)
    games = duckdb.sql(f"select * from '{WORK / 'games.parquet'}'").df()
    labels = team_labels()
    write(SITE_DATA / "teams.json", labels)
    odds_files(table, examples(), games)
    pa = predictions_for_every_plate_appearance(model)
    cb = comebacks(pa)
    del pa
    f = findings(cb, labels, model)
    era_k = table.groupby("era").k.first().to_dict()
    for e in f["eras"]:
        e["k"] = round(float(era_k[e["key"]]), 1)
        e["games"] = int(((games.year >= e["first"]) & (games.year <= e["last"]) & games.used).sum())
    f["spots"] = spots(table)
    f["tail"] = tail_calibration(model, c)
    f["summary"] = summary()
    write(SITE_DATA / "findings.json", f)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "findings.json").write_text(json.dumps(f, indent=1) + "\n")
    return f
