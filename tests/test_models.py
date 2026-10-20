import numpy as np
import pandas as pd

from whiff_lab.lab import grid, grid_index
from whiff_lab.models import Shrunk, Table, brier, fit_prior_strength, log_loss
from whiff_lab.site import fill, fmt


def test_scores_of_a_coin_flip():
    n, w = np.array([10.0]), np.array([5.0])
    assert abs(log_loss(np.array([0.5]), n, w) - np.log(2)) < 1e-12
    assert brier(np.array([0.5]), n, w) == 0.25


def test_prior_strength_is_found_again():
    # states whose true rates scatter around 0.6 like a Beta with strength 50
    rng = np.random.default_rng(1)
    rates = rng.beta(0.6 * 50, 0.4 * 50, size=4000)
    n = rng.integers(5, 200, size=4000)
    w = rng.binomial(n, rates)
    k = fit_prior_strength(n.astype(float), w.astype(float), np.full(4000, 0.6))
    assert 35 < k < 70


def _counts(rows):
    cols = ["year", "inning", "bottom", "outs", "b1", "b2", "b3", "diff", "extra_runner", "n", "w"]
    return pd.DataFrame(rows, columns=cols)


def test_shrinking_pulls_a_rare_state_toward_its_parent():
    # a common state (bases empty) and a rare one (runner on third) with the same inning, outs and score
    c = _counts(
        [
            (2000, 5, 0, 0, 0, 0, 0, 1, 0, 1000, 600),
            (2000, 5, 0, 0, 0, 0, 1, 1, 0, 2, 0),
            (2000, 5, 1, 0, 0, 0, 0, -1, 0, 1000, 400),
            (2000, 3, 0, 1, 1, 0, 0, 0, 0, 1000, 500),
        ]
    )
    rare = c.iloc[[1]]
    assert Table().fit(c).predict(rare)[0] == 1e-3  # 0 of 2: the table says (almost) never
    assert Shrunk().fit(c).predict(rare)[0] > 0.3  # shrunk toward the parent's 60%


def test_the_grid_order():
    g = grid()
    assert len(g) == 10 * 2 * 3 * 8 * 21
    i = grid_index(
        g.inning.to_numpy(), g.bottom.to_numpy(), g.outs.to_numpy(), g.bases.to_numpy(), g["diff"].to_numpy()
    )
    assert (i == np.arange(len(g))).all()
    assert (g.b1 + 2 * g.b2 + 4 * g.b3 == g.bases).all()


def test_page_numbers():
    ctx = {"s": {"n": 15211482, "p": 0.4671, "year": 1908}, "seasons": {1930: {"x": 0.1084}}, "l": [0.5]}
    assert fill(
        "{{s.n|int}} · {{s.p|f3}} · {{s.year|year}} · {{seasons.1930.x|pct1}} · {{l.0|pct0}}", ctx
    ) == ("15,211,482 · 0.467 · 1908 · 10.8% · 50%")
    assert fmt("1952-06-29", "date") == "June 29, 1952"
    assert fmt(1 / 14396, "onein") == "1 in 14,000"


def test_a_rookie_season_is_the_one_that_passes_the_limits():
    from whiff_lab.rookies import rookies

    def bat(year, ab, so=20):
        return {
            "playerID": "kidxx01",
            "yearID": year,
            "stint": 1,
            "teamID": "BOS",
            "lgID": "AL",
            "G": 50,
            "AB": ab,
            "R": 10,
            "H": ab // 4,
            "X2B": 5,
            "X3B": 1,
            "HR": 4,
            "RBI": 20,
            "SB": 2,
            "CS": 1,
            "BB": 10,
            "SO": so,
            "IBB": 0,
            "HBP": 1,
            "SH": 0,
            "SF": 1,
            "GIDP": 2,
        }

    batting = pd.DataFrame([bat(2000, 60), bat(2001, 400), bat(2002, 500)])
    t = {
        "Batting": batting,
        "Pitching": pd.DataFrame(
            {
                c: pd.Series(dtype="object" if c in ("playerID", "teamID") else "int64")
                for c in (
                    "playerID",
                    "yearID",
                    "teamID",
                    "W",
                    "L",
                    "G",
                    "GS",
                    "SV",
                    "IPouts",
                    "H",
                    "ER",
                    "HR",
                    "BB",
                    "SO",
                    "BFP",
                    "HBP",
                )
            }
        ),
        "Appearances": pd.DataFrame(
            [
                {
                    "playerID": "kidxx01",
                    "yearID": y,
                    "G_all": 50,
                    "G_p": 0,
                    "G_c": 0,
                    "G_1b": 0,
                    "G_2b": 0,
                    "G_3b": 0,
                    "G_ss": 50,
                    "G_lf": 0,
                    "G_cf": 0,
                    "G_rf": 0,
                    "G_dh": 0,
                }
                for y in (2000, 2001, 2002)
            ]
        ),
        "People": pd.DataFrame(
            [
                {
                    "playerID": "kidxx01",
                    "birthYear": 1980,
                    "birthMonth": 8,
                    "birthCountry": "USA",
                    "nameFirst": "Kid",
                    "nameLast": "Rookie",
                    "retroID": "rookk001",
                    "bbrefID": "rookiki01",
                }
            ]
        ),
        "AllstarFull": pd.DataFrame([{"playerID": "kidxx01", "yearID": y} for y in (2001, 2003, 2004)]),
        "AwardsSharePlayers": pd.DataFrame(
            [
                {
                    "awardID": "Rookie of the Year",
                    "yearID": 2001,
                    "playerID": "kidxx01",
                    "pointsWon": 30,
                    "pointsMax": 120,
                }
            ]
        ),
        "Teams": pd.DataFrame(
            [{"yearID": y, "teamID": "BOS", "name": "Boston Red Sox"} for y in (2000, 2001, 2002)]
        ),
    }
    r = rookies(t)
    assert len(r) == 1
    row = r.iloc[0]
    assert (row.yearID, row.kind, row.age, row.position) == (
        2001,
        "hitter",
        20,
        "ss",
    )  # born in August: 20 on June 30
    assert row.allstar_next == 2 and row.star == 1  # 2003 and 2004 count, the rookie year's own does not
    assert row.roy_share == 0.25 and row.team == "Boston Red Sox"
