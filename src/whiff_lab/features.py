"""The game states the models learn from, and the eras they are compared across.

A state is what a scoreboard shows as a batter steps in: inning, top or bottom, outs, runners, score.
Every inning from the tenth on counts as "extra innings" (10): they all play the same way, sudden death
for the home team. Score differences beyond 10 runs are counted as 10.

Rows are counted, not listed: every row of `counts()` is one state in one season, with how many times
that state came up and how many of those games the home team went on to win. A model fitted to these
counts learns exactly what it would learn from the 15 million plate appearances one by one, in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from whiff_lab.data import WORK

MAX_INNING = 10
MAX_DIFF = 10


@dataclass(frozen=True)
class Era:
    key: str
    name: str
    first: int
    last: int
    note: str


ERAS = (
    Era("deadball", "Dead ball", 1908, 1919, "a soft, overused ball; bunts, steals and few home runs"),
    Era(
        "liveball",
        "Live ball",
        1920,
        1941,
        "the spitball banned, a cleaner ball, Ruth: runs and home runs boom",
    ),
    Era(
        "postwar",
        "War and integration",
        1942,
        1960,
        "wartime rosters, then Robinson and the color line broken",
    ),
    Era(
        "expansion",
        "Expansion and the pitchers' peak",
        1961,
        1968,
        "new teams, a bigger strike zone, 1968's 3.4 runs",
    ),
    Era(
        "moundlow",
        "Lower mound, the DH",
        1969,
        1992,
        "the mound lowered in 1969, the AL's designated hitter in 1973",
    ),
    Era(
        "offense",
        "High offense",
        1993,
        2005,
        "expansion, small parks and big hitters: the most runs since the 1930s",
    ),
    Era(
        "strikeout",
        "Strikeouts and relievers",
        2006,
        2019,
        "testing, harder throwers, deeper bullpens, more strikeouts",
    ),
    Era(
        "runner",
        "Runner on second",
        2020,
        2100,
        "extra innings start with a runner on second; the pitch clock from 2023",
    ),
)


def era_of(year: int) -> Era:
    return next(e for e in ERAS if e.first <= year <= e.last)


STATE = ["inning", "bottom", "outs", "b1", "b2", "b3", "diff"]


def counts(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """One row per (season, state): n plate appearances in it, w of them in games the home team won.

    Only the games `data.label` marks as used. `extra_runner`: the season's extra innings start with a
    runner on second (Retrosheet's `info,tiebreaker,2`).
    """
    con = con or duckdb.connect()
    return con.sql(f"""
        select g.year,
               least(p.inning, {MAX_INNING})::tinyint as inning,
               p.bottom::tinyint as bottom, p.outs::tinyint as outs,
               p.b1::tinyint as b1, p.b2::tinyint as b2, p.b3::tinyint as b3,
               greatest(-{MAX_DIFF}, least({MAX_DIFF}, p.home_score - p.away_score))::tinyint as diff,
               bool_or(g.extra_runner)::tinyint as extra_runner,
               count(*)::int as n, sum(g.home_win)::int as w
        from '{WORK / "pa.parquet"}' p join '{WORK / "games.parquet"}' g using (game_id)
        where g.used
        group by all
    """).df()


def run_environment(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Runs per team per game in each season, from the games the lab uses."""
    con = con or duckdb.connect()
    return con.sql(f"""
        select year, avg(home_runs + away_runs) / 2 as rpg, count(*) as games
        from '{WORK / "games.parquet"}' where used group by year order by year
    """).df()


def runs_to_end_of_inning(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Run expectancy: from each (runners, outs), the runs the batting team scored before the inning ended,
    on average, per season. Innings 1-8 only: from the ninth on, a walk-off can cut an inning short."""
    con = con or duckdb.connect()
    return con.sql(f"""
        with pa as (
            select p.*, g.year,
                   case when p.bottom then p.home_score else p.away_score end as bat_score,
                   case when p.bottom then g.home_runs else g.away_runs end as final_own
            from '{WORK / "pa.parquet"}' p join '{WORK / "games.parquet"}' g using (game_id)
            where g.used
        ),
        halves as (
            select game_id, inning, bottom, min(bat_score) as start_score, any_value(final_own) as final_own
            from pa group by all
        ),
        ends as (
            select game_id, inning, bottom,
                   coalesce(lead(start_score) over (partition by game_id, bottom order by inning), final_own)
                       as end_score
            from halves
        )
        select pa.year, pa.outs, pa.b1, pa.b2, pa.b3,
               avg(e.end_score - pa.bat_score) as runs, count(*) as n
        from pa join ends e using (game_id, inning, bottom)
        where pa.inning <= 8
        group by all
    """).df()


def base_code(df: pd.DataFrame) -> np.ndarray:
    """Runners as one number 0-7: first base 1, second 2, third 4 (so 7 is bases loaded)."""
    return (df.b1.astype(int) + 2 * df.b2.astype(int) + 4 * df.b3.astype(int)).to_numpy()
