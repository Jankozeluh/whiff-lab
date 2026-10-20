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
