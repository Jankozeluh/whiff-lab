"""The findings of "What are the odds?", computed from the parsed data and the final model.

The final model is the gradient-boosting step fitted on every season (1908 to the latest), so it has
seen the extra-inning runner on second since 2020. Each era's own record is then combined with it by
empirical Bayes (the same idea as step 2): an era moves away from the model only as far as its own games
show it should.
"""

from __future__ import annotations

import csv

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import beta

from whiff_lab.data import RAW, WORK
from whiff_lab.features import ERAS, MAX_DIFF, MAX_INNING, STATE, counts, run_environment
from whiff_lab.models import Boosted, fit_prior_strength

LEVEL = 0.90  # every interval on the site is a 90% interval


def grid() -> pd.DataFrame:
    """Every state the calculator offers, in a fixed order (the order of the site's arrays)."""
    i, b, o, r, d = np.meshgrid(
        np.arange(1, MAX_INNING + 1),
        [0, 1],
        [0, 1, 2],
        np.arange(8),
        np.arange(-MAX_DIFF, MAX_DIFF + 1),
        indexing="ij",
    )
    g = pd.DataFrame(
        {"inning": i.ravel(), "bottom": b.ravel(), "outs": o.ravel(), "bases": r.ravel(), "diff": d.ravel()}
    )
    g["b1"], g["b2"], g["b3"] = g.bases % 2, g.bases // 2 % 2, g.bases // 4
    return g


def final_model(c: pd.DataFrame) -> Boosted:
    return Boosted().fit(c)


def era_table(c: pd.DataFrame, model: Boosted) -> pd.DataFrame:
    """For every era and state: the era's record (n, w), the model, and the two combined with an interval.

    The model is asked as of the era: for the runner-on-second era it knows extra innings start with a
    runner on second. k (fitted per era) says how many games' worth of weight the model gets against the
    era's own record.
    """
    g = grid()
    parts = []
    for era in ERAS:
        e = c[(c.year >= era.first) & (c.year <= era.last)]
        rec = e.groupby(STATE)[["n", "w"]].sum().reset_index()
        cells = g.merge(rec, on=STATE, how="left").fillna({"n": 0, "w": 0})
        cells["extra_runner"] = int(era.key == "runner")
        cells["model"] = model.predict(cells)
        seen = cells[cells.n > 0]
        k = fit_prior_strength(seen.n.to_numpy(), seen.w.to_numpy(), seen.model.to_numpy())
        a = cells.w + k * cells.model
        b = cells.n - cells.w + k * (1 - cells.model)
        cells["p"] = a / (a + b)
        cells["lo"], cells["hi"] = beta.ppf((1 - LEVEL) / 2, a, b), beta.ppf((1 + LEVEL) / 2, a, b)
        cells["era"], cells["k"] = era.key, k
        parts.append(cells)
    return pd.concat(parts, ignore_index=True)


def grid_index(inning, bottom, outs, bases, diff):
    """A state's place in `grid()` (works on numbers and on arrays alike)."""
    return ((((inning - 1) * 2 + bottom) * 3 + outs) * 8 + bases) * (2 * MAX_DIFF + 1) + diff + MAX_DIFF


def predictions_for_every_plate_appearance(model: Boosted) -> pd.DataFrame:
    """The model's home win chance before every plate appearance of every game used (by state lookup)."""
    g = grid()
    table = np.concatenate([model.predict(g.assign(extra_runner=r), floor=1e-5) for r in (0, 1)])
    pa = duckdb.sql(f"""
        select p.game_id, p.inning as real_inning, p.bottom::int as bottom, p.outs::int as outs,
               p.b1::int as b1, p.b2::int as b2, p.b3::int as b3, p.away_score, p.home_score,
               g.year, g.date, g.away, g.home, g.away_runs, g.home_runs, g.home_win,
               g.extra_runner::int as runner
        from '{WORK / "pa.parquet"}' p join '{WORK / "games.parquet"}' g using (game_id)
        where g.used
    """).df()
    idx = grid_index(
        np.minimum(pa.real_inning.to_numpy(), MAX_INNING),
        pa.bottom.to_numpy(),
        pa.outs.to_numpy(),
        pa.b1.to_numpy() + 2 * pa.b2.to_numpy() + 4 * pa.b3.to_numpy(),
        np.clip(pa.home_score.to_numpy() - pa.away_score.to_numpy(), -MAX_DIFF, MAX_DIFF),
    )
    pa["p"] = table[idx + len(g) * pa.runner.to_numpy()]
    return pa


def comebacks(pa: pd.DataFrame) -> pd.DataFrame:
    """Every game's low point for its winner: the smallest chance the winner had before a plate appearance."""
    pa = pa.assign(win_p=np.where(pa.home_win == 1, pa.p, 1 - pa.p))
    low = pa.loc[pa.groupby("game_id").win_p.idxmin()]
    return low.sort_values("win_p").reset_index(drop=True)


def home_field() -> pd.DataFrame:
    """The home team's share of wins in each AL/NL season since 1901, from Retrosheet's game logs (every
    game, with or without play-by-play). Ties are left out."""
    rows = []
    for path in sorted((RAW / "seasons").glob("*/gl*.txt", case_sensitive=False)):
        year = int(path.parent.name)
        if year < 1901:
            continue
        with open(path, newline="", encoding="latin-1") as f:
            for r in csv.reader(f):
                if r[4] in ("AL", "NL") and r[7] in ("AL", "NL") and r[9] != r[10]:
                    rows.append((year, int(r[10]) > int(r[9]), int(r[9]) + int(r[10])))
    d = pd.DataFrame(rows, columns=["year", "home_win", "runs"])
    return (
        d.groupby("year")
        .agg(games=("home_win", "size"), home_win=("home_win", "mean"), rpg=("runs", lambda s: s.mean() / 2))
        .reset_index()
    )


def extra_innings() -> pd.DataFrame:
    """Extra-inning games before and since the runner on second: how long they last, who wins."""
    return duckdb.sql(f"""
        select case when extra_runner then 'runner' else 'before' end as rule,
               count(*) as games, avg(home_win) as home_win,
               avg(case when last_inning = 10 then 1 else 0 end) as over_in_10th,
               avg(last_inning) as mean_last_inning, max(last_inning) as longest
        from '{WORK / "games.parquet"}'
        where used and last_inning >= 10 and year >= 2010
        group by 1 order by 1
    """).df()


def summary() -> dict:
    """The few numbers the lab notes quote, recomputed on every rebuild."""
    c = counts()
    rpg = run_environment()
    games = duckdb.sql(f"select * from '{WORK / 'games.parquet'}'").df()
    return {
        "first_season": int(c.year.min()),
        "last_season": int(c.year.max()),
        "games_parsed": len(games),
        "games_used": int(games.used.sum()),
        "score_match": float(games.score_ok.mean()),
        "plate_appearances": int(c.n.sum()),
        "states_seen": int(c.groupby(STATE).n.sum().gt(0).sum()),
        "rpg_low": rpg.loc[rpg.rpg.idxmin()].to_dict(),
        "rpg_high": rpg.loc[rpg.rpg.idxmax()].to_dict(),
    }
