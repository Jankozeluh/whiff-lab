"""Pitch by pitch: for every hitter and every pitcher in a season, how often a swing missed.

Since 1988 almost every Retrosheet plate appearance comes with its pitches, one letter each:

    B ball   C called strike   S swinging strike   F foul   T foul tip   X put in play   (and a few more)

The last play record of a plate appearance carries all of its pitches (marks such as `.` for a play between
pitches, `>` for a runner going, digits for pickoff throws are skipped). From them: swings, misses, fouls,
balls put in play, called strikes and balls, per player and season, for the batter and the pitcher he faced.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from whiff_lab.data import RAW, WORK, seasons
from whiff_lab.retrosheet import EventError, parse_event

FIRST_PITCH_SEASON = 1988
MISS = set("SMQ")  # swinging strike, missed bunt, swinging on a pitchout
CONTACT = set("FTXLORY")  # fouls, foul tips, in play (also bunted and on pitchouts)
BALL = set("BIPV")  # ball, intentional ball, pitchout, automatic ball
CALLED = set("CK")  # called strike (K: a strike of unknown kind, counted as called)
FIELDS = ("pa", "pa_pitched", "pitches", "swings", "misses", "contact", "in_play", "called", "balls")


def _count(seq: str) -> dict[str, int] | None:
    """One plate appearance's pitches, or None when they are unknown."""
    seq = "".join(ch for ch in seq if ch.isalpha())
    if not seq or "U" in seq or "?" in seq:
        return None
    misses = sum(ch in MISS for ch in seq)
    contact = sum(ch in CONTACT for ch in seq)
    return {
        "pitches": sum(ch != "N" for ch in seq),  # N: no pitch (a balk, an interference)
        "swings": misses + contact,
        "misses": misses,
        "contact": contact,
        "in_play": sum(ch in "XY" for ch in seq),
        "called": sum(ch in CALLED for ch in seq),
        "balls": sum(ch in BALL for ch in seq),
    }


def season_counts(year: int) -> pd.DataFrame:
    """Per player in a season: his plate appearances and pitches as a batter and as a pitcher."""
    tally: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    for path in sorted((RAW / "seasons" / str(year)).glob("*.EV[AN]")):
        pitcher = {}
        with open(path, encoding="latin-1") as f:
            for raw in f:
                line = raw.rstrip("\r\n")
                kind, _, rest = line.partition(",")
                if kind == "id":
                    pitcher = {}
                elif kind in ("start", "sub"):
                    pid = rest.split(",", 1)[0]
                    team, _, pos = rest.rsplit(",", 3)[-3:]
                    if pos.strip() == "1":  # the new pitcher
                        pitcher[team] = pid
                elif kind == "play":
                    _, team, batter, _, pitches, event = rest.split(",", 5)
                    if event.replace("#", "") == "NP":
                        continue
                    try:
                        if parse_event(event).batter is None:
                            continue  # not the end of a plate appearance
                    except EventError:
                        continue
                    fielding = "1" if team == "0" else "0"
                    c = _count(pitches)
                    for role, who in (("bat", batter), ("pit", pitcher.get(fielding))):
                        if who is None:
                            continue
                        t = tally[(role, who)]
                        t["pa"] += 1
                        if c:
                            t["pa_pitched"] += 1
                            for k, v in c.items():
                                t[k] += v
    rows = [{"year": year, "role": role, "retroID": who, **t} for (role, who), t in tally.items()]
    return pd.DataFrame(rows)


def build(workers: int | None = None) -> pd.DataFrame:
    years = [y for y in seasons() if y >= FIRST_PITCH_SEASON]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        df = pd.concat(pool.map(season_counts, years), ignore_index=True)
    df.to_parquet(WORK / "pitch_counts.parquet", index=False)
    return df
