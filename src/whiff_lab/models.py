"""Four ways to answer "what are the odds?", from the plainest to the strongest.

Each model learns from the counts of `features.counts()` (a state in a season, how often it came up, how
often the home team went on to win) and gives the home team's chance of winning for any state.

1. Table: the share of games the home team won from exactly this state. Honest, and often too noisy:
   a state seen four times can say 0% or 100%.
2. Shrunk: the same table with every state pulled toward a coarser one (empirical Bayes). A state seen
   four times barely moves away from its parent; a state seen 100,000 times keeps its own record.
3. Logistic: no table at all. A few numbers built from baseball logic (the lead, the runs the batting team
   can expect, how much of the game is left, how many runs a season scores) and a logistic regression.
4. Boosted: gradient-boosted trees on the raw state: a smooth table that borrows strength between
   neighbouring states, and finds what the formula misses (the walk-off, the last outs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from whiff_lab.features import STATE, base_code

EPS = 1e-3  # no model may say 0% or 100%: one surprise would cost an infinite penalty


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)


def log_loss(p: np.ndarray, n: np.ndarray, w: np.ndarray) -> float:
    """Average surprise per plate appearance (lower is better; 0.693 is a coin flip)."""
    p = clip(p)
    return float(-(w * np.log(p) + (n - w) * np.log(1 - p)).sum() / n.sum())


def brier(p: np.ndarray, n: np.ndarray, w: np.ndarray) -> float:
    """Average squared distance between the forecast and what happened (lower is better; 0.25: coin flip)."""
    return float((w * (1 - p) ** 2 + (n - w) * p**2).sum() / n.sum())


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The state as one integer, so tables can be joined fast."""
    k = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        k = k * 32 + (df[c].to_numpy().astype(np.int64) + 16)
    return pd.Series(k, index=df.index)


class HomeRate:
    """Step 0: the home team's winning share, for every state alike."""

    name = "Home share only"

    def fit(self, c: pd.DataFrame) -> HomeRate:
        self.p = c.w.sum() / c.n.sum()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.p)


class Table:
    """Step 1: what happened from exactly this state."""

    name = "Lookup table"

    def fit(self, c: pd.DataFrame) -> Table:
        g = c.assign(k=_key(c, STATE)).groupby("k")[["n", "w"]].sum()
        self.rate = g.w / g.n
        self.n = g.n
        self.home = HomeRate().fit(c).p
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = _key(df, STATE).map(self.rate).to_numpy()
        return clip(np.where(np.isnan(p), self.home, p))


# From coarse to fine: each level is shrunk toward the one before it.
LEVELS = (
    ["bottom", "diff"],
    ["inning", "bottom", "diff"],
    ["inning", "bottom", "outs", "diff"],
    STATE,
)


def fit_prior_strength(n: np.ndarray, w: np.ndarray, m: np.ndarray) -> float:
    """Empirical Bayes: how many games' worth of trust the parent's rate deserves.

    Each state's true rate is drawn from a Beta distribution centred on its parent's rate m, with a
    strength k (like k games already seen at the parent's rate). k is the value under which the records we
    did see are most likely (the beta-binomial marginal likelihood).
    """
    m = np.clip(m, EPS, 1 - EPS)

    def nll(log_k: float) -> float:
        k = np.exp(log_k)
        a, b = k * m, k * (1 - m)
        return -float((betaln(w + a, n - w + b) - betaln(a, b)).sum())

    res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(1e6)), method="bounded")
    return float(np.exp(res.x))


class Shrunk:
    """Step 2: the table, with each state pulled toward its parent by empirical Bayes."""

    name = "Shrunk table (empirical Bayes)"

    def fit(self, c: pd.DataFrame) -> Shrunk:
        self.root = c.w.sum() / c.n.sum()
        self.levels: list[tuple[list[str], pd.Series, float]] = []
        parent_rate = None
        parent_cols: list[str] = []
        for cols in LEVELS:
            g = (
                c.assign(k=_key(c, cols))
                .groupby("k")
                .agg(n=("n", "sum"), w=("w", "sum"), **{p: (p, "first") for p in cols})
            )
            m = (
                np.full(len(g), self.root)
                if parent_rate is None
                else _key(g, parent_cols).map(parent_rate).to_numpy()
            )
            k = fit_prior_strength(g.n.to_numpy(), g.w.to_numpy(), m)
            rate = (g.w + k * m) / (g.n + k)
            self.levels.append((cols, rate, k))
            parent_rate, parent_cols = rate, cols
        return self

    @property
    def strengths(self) -> dict[str, float]:
        return {"+".join(cols): k for cols, _, k in self.levels}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = np.full(len(df), self.root)
        for cols, rate, _ in self.levels:
            # a state never seen gets its parent's rate (its own record is empty)
            own = _key(df, cols).map(rate).to_numpy()
            p = np.where(np.isnan(own), p, own)
        return clip(p)


def baseball_numbers(df: pd.DataFrame, re: dict[int, float], rpg: pd.Series, rpg_ref: float) -> pd.DataFrame:
    """Numbers a fan would reach for: the lead the home team can expect at the end, and how uncertain it is.

    re: runs expected from each (runners, outs) to the end of the inning, as base_code * 3 + outs.
    rpg: runs per team per game in each season (the run environment); rpg_ref: their average.
    """
    scale = df.year.map(rpg).to_numpy() / rpg_ref
    inning = df.inning.to_numpy()
    bottom = df.bottom.to_numpy().astype(bool)
    outs = df.outs.to_numpy()
    re_now = np.array([re[k] for k in range(24)])[base_code(df) * 3 + outs] * scale
    re0 = re[0] * scale
    # half-innings still to come after this one, for each team (the 9th and extra innings alike)
    last = np.minimum(inning, 9)
    away_left = 9 - last
    home_left = np.where(bottom, 9 - last, 10 - last)
    # from 2020, an extra inning starts with a runner on second: worth more than an empty inning
    runner = (df.extra_runner.to_numpy() == 1) & (inning >= 9)
    re_extra = re[2 * 3 + 0] * scale
    home_future = np.where(runner & (inning == 10) & ~bottom, re_extra, re0) * home_left
    away_future = re0 * away_left
    bat = np.where(bottom, 1, -1)
    mu = df["diff"].to_numpy() + bat * re_now + home_future - away_future
    halves = (3 - outs) / 3 + home_left + away_left
    z = mu / np.sqrt(halves * scale + 0.25)
    return pd.DataFrame({"mu": mu, "halves": halves, "z": z, "scale": scale}, index=df.index)


def _two_rows(c: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Counts to training rows: each state once as a home win (weight w), once as a loss (weight n - w)."""
    XX = pd.concat([X, X], ignore_index=True)
    y = np.r_[np.ones(len(X)), np.zeros(len(X))]
    sw = np.r_[c.w.to_numpy(), (c.n - c.w).to_numpy()].astype(float)
    keep = sw > 0
    return XX[keep], y[keep], sw[keep]


class Logistic:
    """Step 3: logistic regression on a handful of baseball numbers."""

    name = "Logistic regression"
    columns = ("z", "mu", "tie_bottom", "tie_late", "z_late")

    def __init__(self, re: dict[int, float], rpg: pd.Series):
        self.re, self.rpg = re, rpg

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        b = baseball_numbers(df, self.re, self.rpg, self.rpg_ref)
        tie = (df["diff"] == 0).to_numpy()
        late = (df.inning >= 9).to_numpy()
        return pd.DataFrame(
            {
                "z": b.z,
                "mu": b.mu,
                "tie_bottom": tie & (df.bottom == 1).to_numpy(),
                "tie_late": tie & late,
                "z_late": b.z * late,
            },
            index=df.index,
        ).astype(float)

    def fit(self, c: pd.DataFrame) -> Logistic:
        self.rpg_ref = float(np.average(c.year.map(self.rpg), weights=c.n))
        X, y, sw = _two_rows(c, self.design(c))
        self.lr = LogisticRegression(C=1e3, max_iter=500).fit(X, y, sample_weight=sw)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return clip(self.lr.predict_proba(self.design(df))[:, 1])

    @property
    def coefficients(self) -> dict[str, float]:
        return {
            "intercept": float(self.lr.intercept_[0]),
            **{c: float(v) for c, v in zip(self.columns, self.lr.coef_[0], strict=True)},
        }


class Boosted:
    """Step 4: gradient-boosted trees on the state itself (and, from 2020, the runner on second)."""

    name = "Gradient boosting"
    columns = ["inning", "bottom", "outs", "b1", "b2", "b3", "diff", "runner_now"]

    def __init__(self, max_iter: int = 400, learning_rate: float = 0.1, max_leaf_nodes: int = 31):
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes}

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.columns[:-1]].astype(float)
        X["runner_now"] = ((df.extra_runner == 1) & (df.inning >= 10)).astype(float)
        return X

    def fit(self, c: pd.DataFrame) -> Boosted:
        X, y, sw = _two_rows(c, self.design(c))
        mono = [
            1 if col == "diff" else 0 for col in self.columns
        ]  # a bigger home lead never hurts the home team
        self.gb = HistGradientBoostingClassifier(
            **self.params,
            min_samples_leaf=200,
            l2_regularization=1.0,
            monotonic_cst=mono,
            early_stopping=False,
            random_state=0,
        ).fit(X, y, sample_weight=sw)
        return self

    def predict(self, df: pd.DataFrame, floor: float = EPS) -> np.ndarray:
        """floor: how close to 0% or 100% a forecast may go (the comebacks list looks past 1 in 1,000)."""
        return np.clip(self.gb.predict_proba(self.design(df))[:, 1], floor, 1 - floor)
