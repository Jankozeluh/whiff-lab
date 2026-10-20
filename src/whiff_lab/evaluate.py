"""The honest test: fit on the past, score on seasons the models never saw.

Train: 1908-2016. Test: 2017-2025, nine seasons and about 21,000 games, which include the change of 2020
(extra innings start with a runner on second) that no model could have learned from its training years.
The settings of each model were chosen on 2013-2016 with the models fitted on 1908-2012, before the test
seasons were looked at.

Plate appearances from one game are not independent (they share an ending), so every interval here comes
from resampling whole games (a bootstrap by game), not single plate appearances.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from whiff_lab.data import ROOT, WORK
from whiff_lab.features import (
    ERAS,
    MAX_DIFF,
    MAX_INNING,
    STATE,
    base_code,
    counts,
    run_environment,
    runs_to_end_of_inning,
)
from whiff_lab.models import Boosted, HomeRate, Logistic, Shrunk, Table, _key, brier, clip, log_loss

TRAIN_LAST = 2016
TEST_FIRST = 2017
RESULTS = ROOT / "results"


def rows(first: int, last: int) -> pd.DataFrame:
    """The plate appearances of some seasons one by one (for the bootstrap by game)."""
    return duckdb.sql(f"""
        select p.game_id, g.year,
               least(p.inning, {MAX_INNING})::tinyint as inning, p.bottom::tinyint as bottom,
               p.outs::tinyint as outs, p.b1::tinyint as b1, p.b2::tinyint as b2, p.b3::tinyint as b3,
               greatest(-{MAX_DIFF}, least({MAX_DIFF}, p.home_score - p.away_score))::tinyint as diff,
               g.extra_runner::tinyint as extra_runner, g.home_win::int as y
        from '{WORK / "pa.parquet"}' p join '{WORK / "games.parquet"}' g using (game_id)
        where g.used and g.year between {first} and {last}
    """).df()


def run_expectancy(years: range) -> dict[int, float]:
    """Runs to the end of the inning from each (runners, outs), keyed base_code * 3 + outs."""
    d = runs_to_end_of_inning()
    d = d[d.year.isin(years)]
    d = d.assign(k=base_code(d) * 3 + d.outs, rn=d.runs * d.n).groupby("k")[["rn", "n"]].sum()
    return (d.rn / d.n).to_dict()


def ladder(train: pd.DataFrame, years: range) -> list:
    rpg = run_environment().set_index("year").rpg
    re = run_expectancy(years)
    return [m.fit(train) for m in (HomeRate(), Table(), Shrunk(), Logistic(re, rpg), Boosted())]


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 20) -> list[dict]:
    """Forecasts grouped in bins of 5 points: what each bin said, and what happened."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append(
                {"bin": b, "n": int(m.sum()), "said": float(p[m].mean()), "happened": float(y[m].mean())}
            )
    return out


def bootstrap(loss: dict[str, np.ndarray], games: np.ndarray, base: str, reps: int = 1000) -> dict:
    """95% intervals for each model's log loss and for its difference from `base`, resampling games."""
    codes, gi = np.unique(games, return_inverse=True)
    n_pa = np.bincount(gi)
    sums = {k: np.bincount(gi, weights=v) for k, v in loss.items()}
    rng = np.random.default_rng(0)
    draws = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=reps)
    pa = draws @ n_pa
    out = {}
    for k, s in sums.items():
        ll = draws @ s / pa
        diff = ll - draws @ sums[base] / pa
        out[k] = {
            "log_loss_95": np.percentile(ll, [2.5, 97.5]).tolist(),
            "vs_" + base.replace(" ", "_").lower() + "_95": np.percentile(diff, [2.5, 97.5]).tolist(),
        }
    return out


def holdout() -> dict:
    c = counts()
    train = c[c.year <= TRAIN_LAST]
    models = ladder(train, range(1908, TRAIN_LAST + 1))
    test = rows(TEST_FIRST, 2100)
    seen = _key(test, STATE).map(models[1].n).fillna(0).to_numpy()
    y = test.y.to_numpy()
    ones = np.ones(len(test))
    groups = {
        "all": np.ones(len(test), bool),
        "rare states (seen < 100 times in training)": seen < 100,
        "ninth inning": (test.inning == 9).to_numpy(),
        "extra innings, 2017-2019": ((test.inning == 10) & (test.year < 2020)).to_numpy(),
        "extra innings, 2020-2025 (runner on second)": ((test.inning == 10) & (test.year >= 2020)).to_numpy(),
    }
    result = {
        "train": f"1908-{TRAIN_LAST}",
        "test": f"{TEST_FIRST}-{int(test.year.max())}",
        "test_games": int(test.game_id.nunique()),
        "test_plate_appearances": len(test),
        "models": [],
        "groups": {g: float(m.mean()) for g, m in groups.items()},
    }
    losses = {}
    for m in models:
        p = clip(m.predict(test))
        losses[m.name] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        entry = {
            "name": m.name,
            "log_loss": {g: log_loss(p[k], ones[k], y[k]) for g, k in groups.items()},
            "brier": {g: brier(p[k], ones[k], y[k]) for g, k in groups.items()},
            "calibration": calibration(p, y),
        }
        if isinstance(m, Shrunk):
            entry["prior_strengths"] = m.strengths
        if isinstance(m, Logistic):
            entry["coefficients"] = m.coefficients
        result["models"].append(entry)
    boot = bootstrap(losses, test.game_id.to_numpy(), base=models[1].name)
    for entry in result["models"]:
        entry["bootstrap"] = boot[entry["name"]]
    return result


def save(result: dict, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    return path


def rule_change() -> dict:
    """Can a model learn a new rule from a few seasons? Fit on 1908-2022 (three seasons of the runner on
    second), score the extra innings of 2023 on."""
    c = counts()
    models = ladder(c[c.year <= 2022], range(1908, 2023))
    test = rows(2023, 2100)
    extra = (test.inning == MAX_INNING).to_numpy()
    y = test.y.to_numpy()
    out = {
        "train": "1908-2022",
        "test": f"2023-{int(test.year.max())}, extra innings",
        "plate_appearances": int(extra.sum()),
        "games": int(test[extra].game_id.nunique()),
        "models": [],
    }
    for m in models:
        p = clip(m.predict(test))
        out["models"].append(
            {
                "name": m.name,
                "log_loss": log_loss(p[extra], np.ones(extra.sum()), y[extra]),
                "brier": brier(p[extra], np.ones(extra.sum()), y[extra]),
            }
        )
    return out


def era_holdout() -> dict:
    """Does knowing the era help? Each era in turn is left out, the models learn from all the others, and
    are scored on it. The logistic regression is told the era's run environment (runs per game); the table
    and the trees are not."""
    c = counts()
    rpg = run_environment().set_index("year").rpg
    out = []
    for era in ERAS:
        inside = (c.year >= era.first) & (c.year <= era.last)
        train, test = c[~inside], c[inside]
        others = [y for y in range(1908, int(c.year.max()) + 1) if not era.first <= y <= era.last]
        models = [Table().fit(train), Logistic(run_expectancy(others), rpg).fit(train), Boosted().fit(train)]
        row = {
            "era": era.key,
            "name": era.name,
            "rpg": float(np.average(test.year.map(rpg), weights=test.n)),
            "plate_appearances": int(test.n.sum()),
        }
        for m in models:
            row[m.name] = log_loss(m.predict(test), test.n, test.w)
        row["own era (in-sample)"] = log_loss(Table().fit(test).predict(test), test.n, test.w)
        out.append(row)
    return {"eras": out}
