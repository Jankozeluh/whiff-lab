"""Read Retrosheet event files: one row per plate appearance, with the game state before it.

An event file is a list of games. Each game is a list of records (one per line): `id`, `info`, `start`,
`sub`, `play`, `radj` and a few others. Only `play` records change the state; the event field of a play
record says what happened, in Retrosheet's notation (https://www.retrosheet.org/eventfile.htm):

    S8/G.2-H;1-3     single to centre, runner from second scores, runner from first to third
    64(1)3/GDP       ground ball double play, short to second to first
    K+SB2            strikeout, and the runner on first steals second
    W.1-2            walk, the runner on first is forced to second

The parser keeps the bases, the outs and the score, and writes down the state each time a new batter
steps in. It does not need to know who the runners are, only where they are.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

# The bases as the notation names them: B is the batter, H is home (a run).
BASE = {"B": 0, "1": 1, "2": 2, "3": 3, "H": 4}
OUT = -1

# Plays that end the batter's turn and put him somewhere (or out). Order matters: longest prefix first.
_HIT = (("HR", 4), ("DGR", 2), ("H", 4), ("S", 1), ("D", 2), ("T", 3))
_ON_FIRST = ("IW", "HP", "W", "I", "C")  # walks, hit by pitch, catcher's interference: forced advances
_RUNNER_ONLY = ("SB", "CS", "PO", "POCS", "WP", "PB", "BK", "DI", "OA", "FLE", "NP")

_ADVANCE = re.compile(r"^([B123])([-X])([123H])((?:\([^)]*\))*)$")
_GROUP = re.compile(r"(\d+)(?:\(([B123])\))?")
_PAREN = re.compile(r"\(([^)]*)\)")
_ERROR = re.compile(r"E\d")
_PUTOUT = re.compile(r"\d+(?:/[^E]*)?")


class EventError(ValueError):
    """An event the parser does not understand."""


@dataclass
class Outcome:
    """What one event does: where each runner (and the batter) ends up."""

    batter: int | None = None  # None: the batter's turn goes on; OUT, or the base he reaches (4 = scored)
    runners: dict[int, int] = field(default_factory=dict)  # from base -> to base (OUT, 1-4)
    forced: bool = False  # a walk-like play: runners not named are forced along
    clears: bool = False  # a home run: runners not named score


def split_top(text: str, sep: str) -> list[str]:
    """Split on `sep`, but not inside parentheses."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _has_error(parens: str) -> bool:
    """True when the runner the play tried to put out is safe on an error.

    `1X3(5E4)`: the throw to third was dropped, the runner is safe. But `2XH(96421)(E1/TH)` and
    `1XH(E2/TH)(42)` have both an error and a clean putout: the error let some runner move up, and the
    putout still stands, so the runner is out.
    """
    groups = _PAREN.findall(parens)
    error = any(_ERROR.search(p) for p in groups)
    putout = any(_PUTOUT.fullmatch(p) for p in groups)
    return error and not putout


def _runner_event(code: str, out: Outcome) -> None:
    """Steals, caught stealing, pickoffs and the plays that move runners without the batter."""
    if code.startswith("SB"):
        frm = {"2": 1, "3": 2, "H": 3}[code[2]]
        out.runners[frm] = frm + 1
    elif code.startswith("POCS"):
        to = code[4]
        frm = {"2": 1, "3": 2, "H": 3}[to]
        out.runners[frm] = BASE[to] if _has_error(code[5:]) else OUT
    elif code.startswith("CS"):
        to = code[2]
        frm = {"2": 1, "3": 2, "H": 3}[to]
        out.runners[frm] = BASE[to] if _has_error(code[3:]) else OUT
    elif code.startswith("PO"):
        frm = int(code[2])
        if not _has_error(code[3:]):
            out.runners[frm] = OUT
    elif code.startswith(("WP", "PB", "BK", "DI", "OA", "FLE", "NP")) or _ERROR.match(code):
        pass  # the runners' moves are in the advances
    else:
        raise EventError(f"unknown runner event {code!r}")


def parse_event(event: str) -> Outcome:
    """Turn an event string into an Outcome, before the base state is known."""
    event = event.strip().replace("#", "").replace("!", "").replace("?", "")
    head, _, adv = event.partition(".")
    basic = split_top(head, "/")[0]
    out = Outcome()

    first, _, extra = basic.partition("+")
    if first.startswith(_RUNNER_ONLY):
        for code in split_top(first, ";"):
            _runner_event(code, out)
    elif first.startswith("K"):
        out.batter = OUT
        for code in split_top(extra, ";") if extra else ():
            _runner_event(code, out)
    elif first.startswith(_ON_FIRST):
        out.batter, out.forced = 1, True
        for code in split_top(extra, ";") if extra else ():
            _runner_event(code, out)
    elif first.startswith("FC") or first.startswith("E"):
        out.batter = 1
    elif first[:1].isdigit():
        groups = _GROUP.findall(first)
        if "E" in first:  # a fielder's throw muffed by another (3E1): the batter is safe
            out.batter = 1
        else:
            batter_out = False
            for i, (_, who) in enumerate(groups):
                if who == "B" or (not who and i == len(groups) - 1):
                    batter_out = True
                elif who:
                    out.runners[int(who)] = OUT
            out.batter = OUT if batter_out else 1
    else:
        for code, base in _HIT:
            if first.startswith(code):
                out.batter = base
                out.clears = base == 4
                break
        else:
            raise EventError(f"unknown event {event!r}")

    for item in split_top(adv, ";") if adv else ():
        m = _ADVANCE.match(item)
        if not m:
            raise EventError(f"bad advance {item!r} in {event!r}")
        frm, kind, to, parens = m.groups()
        dest = OUT if kind == "X" and not _has_error(parens) else BASE[to]
        if frm == "B":
            out.batter = dest
        else:
            out.runners[int(frm)] = dest
    return out


@dataclass
class State:
    """The state of a half-inning: runners on 1st, 2nd, 3rd, and the outs."""

    bases: list[bool] = field(default_factory=lambda: [False, False, False, False])  # index 1-3 used
    outs: int = 0

    def clear(self) -> None:
        self.bases = [False, False, False, False]
        self.outs = 0


def apply(state: State, out: Outcome) -> tuple[int, int]:
    """Move the runners and the batter. Returns (runs scored, base collisions fixed)."""
    start = state.bases[:]
    moves = dict(out.runners)
    if out.forced and out.batter == 1 and start[1]:  # the batter takes first: each runner behind is forced
        moves.setdefault(1, 2)
        if start[2]:
            moves.setdefault(2, 3)
            if start[3]:
                moves.setdefault(3, 4)
    if out.clears:
        for b in (1, 2, 3):
            if start[b] and b not in moves:
                moves[b] = 4
    runs = 0
    bases = [False, False, False, False]
    staying = [b for b in (1, 2, 3) if start[b] and b not in moves]
    for b in staying:
        bases[b] = True
    movers = [(b, moves[b]) for b in (3, 2, 1) if b in moves and start[b]]
    fixes = sum(1 for b in moves if not start[b])  # a move from an empty base: an earlier play was off
    if out.batter is not None:
        movers.append((0, out.batter))
    for _, dest in movers:
        if dest == OUT:
            state.outs += 1
        elif dest == 4:
            runs += 1
        else:
            while dest < 4 and bases[dest]:  # a base already taken: the notation left a move out
                dest += 1
                fixes += 1
            if dest == 4:
                runs += 1
            else:
                bases[dest] = True
    state.bases = bases
    return runs, fixes


@dataclass
class Game:
    game_id: str
    info: dict[str, str]
    rows: list[tuple] = field(default_factory=list)  # one per plate appearance
    runs: list[int] = field(default_factory=lambda: [0, 0])  # visitors, home
    last_inning: int = 0
    last_half: int = 0
    outs_in_last_half: int = 0
    fixes: int = 0
    errors: list[str] = field(default_factory=list)


PA_COLUMNS = ("inning", "bottom", "outs", "b1", "b2", "b3", "away_score", "home_score")


def read_games(lines: Iterable[str]) -> Iterator[Game]:
    """Read the games of one event file."""
    game: Game | None = None
    state = State()
    half = None
    new_turn = True
    pending_radj: list[int] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        kind, _, rest = line.partition(",")
        if kind == "id":
            if game is not None:
                yield game
            game = Game(rest.strip(), {})
            state, half, new_turn, pending_radj = State(), None, True, []
        elif game is None:
            continue
        elif kind == "info":
            key, _, value = rest.partition(",")
            game.info[key] = value.strip('"')
        elif kind == "radj":
            pending_radj.append(int(rest.rsplit(",", 1)[1]))
        elif kind == "play":
            inning_s, team_s, _, _, _, event = rest.split(",", 5)
            event = event.strip()
            if event.replace("#", "") == "NP":
                continue
            this_half = (int(inning_s), int(team_s))
            if this_half != half:
                state.clear()
                half, new_turn = this_half, True
            for base in pending_radj:  # the extra-inning runner placed on second (2020 on)
                state.bases[base] = True
            pending_radj = []
            if state.outs >= 3:  # a play after the third out: skip it, it cannot change the state
                continue
            if new_turn:
                game.rows.append(
                    (
                        half[0],
                        half[1],
                        state.outs,
                        state.bases[1],
                        state.bases[2],
                        state.bases[3],
                        game.runs[0],
                        game.runs[1],
                    )
                )
                new_turn = False
            try:
                outcome = parse_event(event)
            except EventError as err:
                game.errors.append(str(err))
                continue
            runs, fixes = apply(state, outcome)
            game.runs[half[1]] += runs
            if state.outs < 3:  # a collision on the third out changes nothing
                game.fixes += fixes
            game.last_inning, game.last_half, game.outs_in_last_half = half[0], half[1], state.outs
            if outcome.batter is not None or state.outs >= 3:  # the batter's turn is over
                new_turn = True
    if game is not None:
        yield game
