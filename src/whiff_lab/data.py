"""Get Retrosheet's files and turn them into two tables: games and plate appearances.

The files come from Retrosheet's official releases as mirrored by the Chadwick Baseball Bureau on GitHub
(branch `official`). Only the files the lab reads are fetched (a sparse, shallow git checkout), they are
kept in data/raw/ and never fetched twice: a rebuild asks git for what changed and nothing else.
"""

from __future__ import annotations

import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from whiff_lab.retrosheet import PA_COLUMNS, read_games

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "retrosheet"
WORK = ROOT / "data" / "work"
MIRROR = "https://github.com/chadwickbureau/retrosheet.git"
BRANCH = "official"
# the names as the mirror spells them (2024's game log is gl2024.txt); git matches them by case on Linux
SPARSE = (
    "/seasons/*/*.EVA",
    "/seasons/*/*.EVN",
    "/seasons/*/GL*.TXT",
    "/seasons/*/gl*.txt",
    "/reference/teams.csv",
    "/README.md",
)
FIRST_SEASON = 1901


def git(*args: str, cwd: Path = RAW) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch() -> str:
    """Fetch the event files and game logs (or only what changed since the last fetch). Returns the commit."""
    if not (RAW / ".git").exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            MIRROR,
            str(RAW),
            cwd=RAW.parent,
        )
        git("sparse-checkout", "set", "--no-cone", *SPARSE)
        git("checkout", BRANCH)
    else:
        git("fetch", "--depth", "1", "origin", BRANCH)
        git("reset", "--hard", f"origin/{BRANCH}")
    return git("rev-parse", "HEAD")


def record_source(**sources: str) -> None:
    """Write down which releases the results come from (the rebuild compares them with the latest ones)."""
    path = ROOT / "results" / "source.json"
    path.parent.mkdir(exist_ok=True)
    known = json.loads(path.read_text()) if path.exists() else {"mirror": MIRROR, "branch": BRANCH}
    path.write_text(json.dumps({**known, **sources}, indent=1) + "\n")


def seasons() -> list[int]:
    return sorted(
        int(p.name) for p in (RAW / "seasons").iterdir() if p.name.isdigit() and any(p.glob("*.EV[AN]"))
    )


def game_logs(year: int) -> pd.DataFrame:
    """Retrosheet's game log of a season: one line per game, with the final score."""
    path = next((RAW / "seasons" / str(year)).glob("gl*.txt", case_sensitive=False), None)
    if path is None:
        return pd.DataFrame()
    rows = []
    with open(path, newline="", encoding="latin-1") as f:
        for r in csv.reader(f):
            rows.append(
                {
                    "game_id": f"{r[6]}{r[0]}{r[1]}",
                    "gl_away_runs": int(r[9]),
                    "gl_home_runs": int(r[10]),
                    "gl_outs": int(r[11]) if r[11] else None,
                    "gl_completion": r[13],
                    "gl_forfeit": r[14],
                }
            )
    return pd.DataFrame(rows)


def runs_per_game(year: int) -> float | None:
    """Runs per team per game in a season, from its game log (every game, with or without play-by-play)."""
    gl = game_logs(year)
    if gl.empty:
        return None
    return float((gl.gl_away_runs + gl.gl_home_runs).sum() / (2 * len(gl)))


def parse_season(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one season's regular-season event files (AL and NL)."""
    games, pas = [], []
    for path in sorted((RAW / "seasons" / str(year)).glob("*.EV[AN]")):
        with open(path, encoding="latin-1") as f:
            for g in read_games(f):
                info = g.info
                games.append(
                    {
                        "game_id": g.game_id,
                        "year": year,
                        "date": info.get("date", "").replace("/", "-"),
                        "away": info.get("visteam"),
                        "home": info.get("hometeam"),
                        "scheduled_innings": int(info.get("innings") or 9),
                        "extra_runner": info.get("tiebreaker") == "2",
                        "away_runs": g.runs[0],
                        "home_runs": g.runs[1],
                        "last_inning": g.last_inning,
                        "plate_appearances": len(g.rows),
                        "fixes": g.fixes,
                        "errors": len(g.errors),
                    }
                )
                pas.extend((g.game_id, *row) for row in g.rows)
    gdf = pd.DataFrame(games)
    pdf = pd.DataFrame(pas, columns=("game_id", *PA_COLUMNS))
    gl = game_logs(year)
    if not gl.empty:
        gdf = gdf.merge(gl, on="game_id", how="left")
    return gdf, pdf


def build(years: list[int] | None = None, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every season and write data/work/games.parquet and data/work/pa.parquet."""
    years = years or seasons()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(parse_season, years))
    games = pd.concat([g for g, _ in parts], ignore_index=True)
    pas = pd.concat([p for _, p in parts], ignore_index=True)
    games = label(games)
    WORK.mkdir(parents=True, exist_ok=True)
    games.to_parquet(WORK / "games.parquet", index=False)
    pas.to_parquet(WORK / "pa.parquet", index=False)
    return games, pas


def label(games: pd.DataFrame) -> pd.DataFrame:
    """Mark which games the models learn from, and who won.

    A game is used when our replay of its play-by-play ends with the game log's final score (so every state
    in it is right), it was scheduled for nine innings, and it went at least nine (no rain-shortened game:
    those end on the weather, not on the baseball). Ties are left out: nobody won.
    """
    g = games.copy()
    g["score_ok"] = (g.away_runs == g.gl_away_runs) & (g.home_runs == g.gl_home_runs)
    g["home_win"] = (g.home_runs > g.away_runs).astype("Int8")
    g.loc[g.home_runs == g.away_runs, "home_win"] = pd.NA
    full = (g.gl_outs.fillna(0) >= 51) & (g.last_inning >= 9)
    g["used"] = g.score_ok & (g.scheduled_innings == 9) & full & g.home_win.notna() & (g.errors == 0)
    return g
