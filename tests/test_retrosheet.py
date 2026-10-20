from pathlib import Path

import pytest

from whiff_lab.retrosheet import OUT, State, apply, parse_event, read_games

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "retrosheet" / "seasons"


def after(bases: tuple[int, ...], outs: int, event: str) -> tuple[tuple[int, ...], int, int]:
    """Play `event` with runners on `bases` (e.g. (1, 3)) and `outs` out: (bases after, outs after, runs)."""
    state = State()
    for b in bases:
        state.bases[b] = True
    state.outs = outs
    runs, _ = apply(state, parse_event(event))
    return tuple(b for b in (1, 2, 3) if state.bases[b]), state.outs, runs


@pytest.mark.parametrize(
    ("bases", "outs", "event", "expected"),
    [
        ((), 0, "S8/G", ((1,), 0, 0)),
        ((1, 2), 0, "S8/G.2-H;1-3", ((1, 3), 0, 1)),
        ((1,), 0, "D7/L", ((1, 2), 0, 0)),  # the runner not named stays (the notation always names them)
        ((1, 2, 3), 1, "HR/F7", ((), 1, 4)),  # a home run clears the bases, named or not
        ((), 0, "HR/F7.B-H", ((), 0, 1)),
        ((1,), 0, "64(1)3/GDP", ((), 2, 0)),
        ((1,), 1, "64(1)/FO/G", ((1,), 2, 0)),  # force at second, the batter is safe at first
        ((2,), 0, "8(B)84(2)/LDP", ((), 2, 0)),
        ((1, 2), 0, "1(B)16(2)63(1)/TP", ((), 3, 0)),
        ((), 0, "K", ((), 1, 0)),
        ((), 2, "K+WP.B-1", ((1,), 2, 0)),  # strike three gets away, the batter reaches
        ((1,), 0, "K+SB2", ((2,), 1, 0)),
        ((1,), 0, "K+CS2(24)/DP", ((), 2, 0)),
        ((1, 2), 0, "W", ((1, 2, 3), 0, 0)),  # forced runners move up without being named
        ((1, 2, 3), 0, "W", ((1, 2, 3), 0, 1)),
        ((2,), 0, "W", ((1, 2), 0, 0)),  # a runner not forced stays
        ((1, 3), 0, "HP", ((1, 2, 3), 0, 0)),
        ((1,), 0, "IW", ((1, 2), 0, 0)),
        ((3,), 1, "8/SF.3-H", ((), 2, 1)),
        ((), 0, "E6/G", ((1,), 0, 0)),
        ((), 0, "3E1/G34", ((1,), 0, 0)),
        ((1,), 0, "FC6/G.1X2(64)", ((1,), 1, 0)),
        ((1,), 0, "SB2", ((2,), 0, 0)),
        ((1, 3), 0, "SBH;SB2", ((2,), 0, 1)),
        ((1,), 0, "CS2(26)", ((), 1, 0)),
        ((1,), 0, "CS2(2E6).1-3", ((3,), 0, 0)),  # the tag was dropped: safe
        ((1,), 0, "PO1(13)", ((), 1, 0)),
        ((1,), 0, "PO1(E3).1-2", ((2,), 0, 0)),
        ((1,), 0, "POCS2(136)", ((), 1, 0)),
        ((3,), 0, "WP.3-H", ((), 0, 1)),
        ((1, 2), 0, "FLE5", ((1, 2), 0, 0)),
        ((1,), 0, "S9.1X3(5E4)", ((1, 3), 0, 0)),  # the error at third saves the runner
        ((2,), 1, "E5/P25.2XH(96421)(E1/TH);B-3", ((3,), 2, 0)),  # out at home; the error moves the batter
        ((1,), 1, "SB2.1XH(E2/TH)(42)", ((), 2, 0)),  # the throw got away, then he was out at home
        ((3,), 0, "FC6.3XH(6E2)(NR)(UR);B-1", ((1,), 0, 1)),
        ((1,), 0, "S7#.1-3!", ((1, 3), 0, 0)),  # uncertainty and exceptional-play marks are ignored
        ((), 0, "C/E2", ((1,), 0, 0)),
        ((), 0, "DGR/L9LS", ((2,), 0, 0)),
    ],
)
def test_events(bases, outs, event, expected):
    assert after(bases, outs, event) == expected


def test_outcome_parts():
    o = parse_event("64(1)3/GDP")
    assert o.batter == OUT and o.runners == {1: OUT}
    assert parse_event("NP").batter is None


GAME = """id,TST202007240
version,2
info,visteam,AWY
info,hometeam,TST
info,innings,9
info,tiebreaker,2
play,1,0,a1,00,,S8/G
play,1,0,a2,00,,64(1)3/GDP
play,1,0,a3,00,,K
play,1,1,h1,00,,HR/F7
play,1,1,h2,00,,SB2
play,1,1,h2,00,,8/F
play,1,1,h3,00,,K
play,1,1,h4,00,,63/G
radj,a9,2
play,10,0,a4,00,,K
play,10,0,a5,00,,S9.2-H
play,10,0,a6,00,,CS2(26)
radj,h9,2
play,10,1,h5,00,,HR/F7.2-H
"""


def test_read_game():
    (g,) = read_games(GAME.splitlines())
    assert g.game_id == "TST202007240" and g.info["tiebreaker"] == "2"
    assert g.runs == [1, 3]
    # (inning, bottom, outs, on 1st, on 2nd, on 3rd, visitors' runs, home runs) as each batter steps in
    assert g.rows == [
        (1, 0, 0, False, False, False, 0, 0),
        (1, 0, 0, True, False, False, 0, 0),
        (1, 0, 2, False, False, False, 0, 0),
        (1, 1, 0, False, False, False, 0, 0),
        (1, 1, 0, False, False, False, 0, 1),  # h2 steps in; his steal comes during his turn
        (1, 1, 1, False, False, False, 0, 1),
        (1, 1, 2, False, False, False, 0, 1),
        (10, 0, 0, False, True, False, 0, 1),  # the runner placed on second
        (10, 0, 1, False, True, False, 0, 1),
        (10, 0, 1, True, False, False, 1, 1),
        (10, 1, 0, False, True, False, 1, 1),
    ]


@pytest.mark.skipif(not (RAW / "2023").exists(), reason="needs the Retrosheet files (whiff-lab fetch)")
def test_a_whole_season_ends_on_the_game_logs_scores():
    from whiff_lab.data import label, parse_season

    games, pas = parse_season(2023)
    games = label(games)
    assert len(games) == 2430 and games.score_ok.all()
    assert games.fixes.sum() == 0
    assert len(pas) > 180_000
