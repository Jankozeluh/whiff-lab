"""Experiment 2, the tests before anything is built: does a model see future stars in a rookie season better
than the rookie's numbers do? And do his pitches (how often his swings miss) add what the card can't show?

A rookie season here is the season a player first goes past 130 at-bats (a hitter) or 50 innings (a pitcher),
the limits MLB uses for rookie status. The question for each rookie: will he be picked for at least two
All-Star teams in the nine seasons after it? That is the label: a "star".

Data: the Lahman Baseball Database (season statistics, birth dates, positions, All-Star teams, Rookie of the
Year votes), CC BY-SA 3.0, read from the R package Lahman on CRAN.

The test is by time, like experiment 1: everything is fitted on the rookies of 1932-1994 and scored on the
rookies of 1995-2016, whose nine following seasons are all in the data.
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from whiff_lab.data import ROOT, WORK
from whiff_lab.models import fit_prior_strength

RAW = ROOT / "data" / "raw" / "lahman"
TABLES = ("People", "Batting", "Pitching", "Appearances", "AllstarFull", "AwardsSharePlayers")
CRAN = "https://cran.r-project.org/src/contrib/"
FIRST, TRAIN_LAST, TEST_LAST = 1932, 1994, 2016
WINDOW, STAR = 9, 2  # a star: 2+ All-Star teams in the 9 seasons after the rookie season
AB_LIMIT, OUTS_LIMIT = 130, 150


def fetch() -> Path:
    """The current Lahman package from CRAN, kept in data/raw/lahman/ (fetched once per version)."""
    page = urllib.request.urlopen(
        "https://cran.r-project.org/web/packages/Lahman/index.html", timeout=60
    ).read()
    name = re.search(rb"Lahman_[\d.\-]+\.tar\.gz", page).group().decode()
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CRAN + name, path)
    return path


def tables(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """The Lahman tables the experiment reads, as data frames (cached as Parquet in data/work/lahman/)."""
    out = WORK / "lahman"
    if path is None and all((out / f"{t}.parquet").exists() for t in TABLES):
        return {t: pd.read_parquet(out / f"{t}.parquet") for t in TABLES}
    import rdata

    path = path or fetch()
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    with tarfile.open(path) as tar:
        for t in TABLES:
            raw = tar.extractfile(f"Lahman/data/{t}.RData").read()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # R's Date class: the dates we need are also in plain columns
                df = rdata.read_rda(io.BytesIO(raw))[t]
            df = df[
                [c for c in df.columns if df[c].dtype != object or isinstance(df[c].dropna().iloc[0], str)]
            ]
            df.to_parquet(out / f"{t}.parquet", index=False)
            result[t] = df
    return result


def _season_totals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["playerID", "yearID"], as_index=False)[cols].sum()


POSITIONS = ("c", "1b", "2b", "3b", "ss", "lf", "cf", "rf", "dh")


def rookies(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per rookie season, hitter or pitcher, with its numbers, its age and what came after."""
    bat = _season_totals(
        t["Batting"],
        ["G", "AB", "H", "X2B", "X3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF"],
    )
    pit = (
        _season_totals(
            t["Pitching"], ["W", "L", "G", "GS", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "BFP", "HBP"]
        )
        .add_prefix("p_")
        .rename(columns={"p_playerID": "playerID", "p_yearID": "yearID"})
    )
    s = bat.merge(pit, on=["playerID", "yearID"], how="outer").fillna(0).sort_values(["playerID", "yearID"])
    g = s.groupby("playerID")
    s["ab_before"] = g.AB.cumsum() - s.AB
    s["outs_before"] = g.p_IPouts.cumsum() - s.p_IPouts
    s["pa"] = s.AB + s.BB + s.HBP + s.SF + s.SH
    s["pa_before"] = g.pa.cumsum() - s.pa
    hitter = (s.ab_before <= AB_LIMIT) & (s.ab_before + s.AB > AB_LIMIT) & (s.outs_before <= OUTS_LIMIT)
    pitcher = (
        (s.outs_before <= OUTS_LIMIT) & (s.outs_before + s.p_IPouts > OUTS_LIMIT) & (s.ab_before <= AB_LIMIT)
    )
    s["kind"] = None
    s.loc[pitcher, "kind"] = "pitcher"
    s.loc[hitter & ~pitcher, "kind"] = "hitter"
    both = s[hitter & pitcher].assign(kind="hitter")  # a two-way rookie counts twice
    r = pd.concat([s[s.kind.notna()], both], ignore_index=True)

    # the league around him: rates of all non-pitchers (hitters), of all pitchers (pitchers)
    app = (
        t["Appearances"]
        .groupby(["playerID", "yearID"], as_index=False)[["G_all", "G_p", *[f"G_{p}" for p in POSITIONS]]]
        .sum()
    )
    lg_of = t["Batting"].groupby(["playerID", "yearID"]).lgID.first().rename("lgID").reset_index()
    league = s.merge(app, on=["playerID", "yearID"], how="left").merge(lg_of, on=["playerID", "yearID"])
    nonp = league[league.G_p < league.G_all / 2]
    lb = nonp.groupby("yearID")[["AB", "H", "X2B", "X3B", "HR", "BB", "SO", "HBP", "SF", "SH", "pa"]].sum()
    lp = s.groupby("yearID")[["p_IPouts", "p_ER", "p_SO", "p_BB", "p_HR", "p_BFP", "p_HBP", "p_H"]].sum()

    r = r.merge(app, on=["playerID", "yearID"], how="left")
    people = t["People"][
        ["playerID", "birthYear", "birthMonth", "nameFirst", "nameLast", "retroID", "bbrefID"]
    ]
    r = r.merge(people, on="playerID", how="left")
    r["age"] = r.yearID - r.birthYear - (r.birthMonth.fillna(1) >= 7)  # age on June 30, the usual convention

    # hitters' numbers, and the league's
    L = lb.reindex(r.yearID).reset_index(drop=True)
    pa, lpa = r.pa.clip(lower=1), L.pa
    single = r.H - r.X2B - r.X3B - r.HR
    r["avg"] = r.H / r.AB.clip(lower=1)
    r["obp"] = (r.H + r.BB + r.HBP) / (r.AB + r.BB + r.HBP + r.SF).clip(lower=1)
    r["slg"] = (single + 2 * r.X2B + 3 * r.X3B + 4 * r.HR) / r.AB.clip(lower=1)
    lg_obp = (L.H + L.BB + L.HBP) / (L.AB + L.BB + L.HBP + L.SF)
    lg_slg = (L.H - L.X2B - L.X3B - L.HR + 2 * L.X2B + 3 * L.X3B + 4 * L.HR) / L.AB
    r["ops_plus"] = 100 * (r.obp / lg_obp + r.slg / lg_slg - 1)
    r["k_rate"], r["lg_k"] = r.SO / pa, L.SO / lpa
    r["bb_rate"], r["lg_bb"] = r.BB / pa, L.BB / lpa
    r["hr_rate"], r["lg_hr"] = r.HR / pa, L.HR / lpa
    bip = (r.AB - r.SO - r.HR + r.SF).clip(lower=1)
    r["babip"], r["lg_babip"] = (r.H - r.HR) / bip, (L.H - L.HR) / (L.AB - L.SO - L.HR + L.SF)
    r["bip"] = bip
    r["iso"], r["lg_iso"] = r.slg - r.avg, lg_slg - L.H / L.AB
    r["sb_rate"] = (r.SB + r.CS) / pa
    pos = r[[f"G_{p}" for p in POSITIONS]].fillna(0)
    r["position"] = np.where(pos.sum(axis=1) > 0, pos.idxmax(axis=1).str[2:], "dh")

    # pitchers' numbers, and the league's
    P = lp.reindex(r.yearID).reset_index(drop=True)
    bf, lbf = r.p_BFP.clip(lower=1), P.p_BFP
    r["ip"] = r.p_IPouts / 3
    r["era"] = 27 * r.p_ER / r.p_IPouts.clip(lower=1)
    r["era_minus"] = 100 * r.era / (27 * P.p_ER / P.p_IPouts)
    r["p_k_rate"], r["p_lg_k"] = r.p_SO / bf, P.p_SO / lbf
    r["p_bb_rate"], r["p_lg_bb"] = r.p_BB / bf, P.p_BB / lbf
    r["p_hr_rate"], r["p_lg_hr"] = r.p_HR / bf, P.p_HR / lbf
    r["starter"] = r.p_GS / r.p_G.clip(lower=1)

    # what came after: All-Star teams in the next nine seasons (a season with two All-Star games counts once)
    stars = t["AllstarFull"].drop_duplicates(["playerID", "yearID"])[["playerID", "yearID"]]
    after = r[["playerID", "yearID", "kind"]].merge(
        stars.rename(columns={"yearID": "as_year"}), on="playerID"
    )
    after = after[(after.as_year > after.yearID) & (after.as_year <= after.yearID + WINDOW)]
    r = r.merge(
        after.groupby(["playerID", "yearID", "kind"]).size().rename("allstar_next").reset_index(),
        on=["playerID", "yearID", "kind"],
        how="left",
    )
    r["allstar_next"] = r.allstar_next.fillna(0).astype(int)
    r["star"] = (r.allstar_next >= STAR).astype(int)

    # what the Rookie of the Year voters said (their share of the points; no votes is 0)
    roy = t["AwardsSharePlayers"]
    roy = roy[roy.awardID == "Rookie of the Year"].assign(roy_share=lambda d: d.pointsWon / d.pointsMax)
    r = r.merge(
        roy[["playerID", "yearID", "roy_share"]].drop_duplicates(["playerID", "yearID"]),
        on=["playerID", "yearID"],
        how="left",
    )
    r["roy_share"] = r.roy_share.fillna(0)
    return r[r.yearID >= FIRST].reset_index(drop=True)


def shrink(
    hits: pd.Series, trials: pd.Series, league: pd.Series, train: pd.Series
) -> tuple[pd.Series, float]:
    """A rate pulled toward the league's by empirical Bayes (strength fitted on the training rookies)."""
    k = fit_prior_strength(
        trials[train].to_numpy(float), hits[train].to_numpy(float), league[train].to_numpy()
    )
    return (hits + k * league) / (trials + k), k


DEFENSE = {"c": 3, "ss": 3, "2b": 2, "cf": 2, "3b": 1, "rf": 0, "lf": 0, "1b": -1, "dh": -2}


def features(r: pd.DataFrame, kind: str, train: pd.Series) -> tuple[pd.DataFrame, dict]:
    """The model's view of a rookie: how young, how much he played, where, and his skills with the luck taken
    out (each rate shrunk toward the league's, so 150 plate appearances can't say more than they know)."""
    ks = {}
    if kind == "hitter":
        pa = r.pa
        out = {}
        for name, hits, trials, lg in (
            ("k", r.SO, pa, r.lg_k),
            ("bb", r.BB, pa, r.lg_bb),
            ("hr", r.HR, pa, r.lg_hr),
            ("babip", r.H - r.HR, r.bip, r.lg_babip),
        ):
            rate, ks[name] = shrink(hits, trials, lg, train)
            out[f"{name}_skill"] = rate - lg
        X = pd.DataFrame(
            {
                "age": r.age,
                "log_pa": np.log(pa),
                "log_pa_before": np.log1p(r.pa_before),
                "defense": r.position.map(DEFENSE).fillna(0),
                "catcher": (r.position == "c") * 1.0,
                "sb_rate": r.sb_rate,
                **out,
            }
        )
    else:
        bf = r.p_BFP.clip(lower=1)
        out = {}
        for name, hits, lg in (("k", r.p_SO, r.p_lg_k), ("bb", r.p_BB, r.p_lg_bb), ("hr", r.p_HR, r.p_lg_hr)):
            rate, ks[name] = shrink(hits, bf, lg, train)
            out[f"{name}_skill"] = rate - lg
        X = pd.DataFrame(
            {
                "age": r.age,
                "log_bf": np.log(bf),
                "starter": r.starter,
                "log_outs_before": np.log1p(r.outs_before),
                **out,
            }
        )
    return X.astype(float), ks


def back_of_the_card(r: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The numbers on the back of a baseball card: what anyone can rank by."""
    if kind == "hitter":
        cols = {"avg": r.avg, "hr": r.HR, "rbi": r.RBI, "sb": r.SB, "ops_plus": r.ops_plus}
    else:
        cols = {"w": r.p_W, "era_minus": r.era_minus, "so": r.p_SO, "sv": r.p_SV}
    return pd.DataFrame(cols).astype(float)


def _logit(X: pd.DataFrame, y: np.ndarray) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def top_k(df: pd.DataFrame, score: np.ndarray, k: int = 5) -> dict:
    """Each test season's top k rookies: how many became stars, and what share of all stars that is."""
    d = df.assign(score=score).sort_values(["yearID", "score"], ascending=[True, False])
    picked = d.groupby("yearID").head(k)
    return {
        "picked": len(picked),
        "stars_picked": int(picked.star.sum()),
        "stars_all": int(df.star.sum()),
        "hit_rate": float(picked.star.mean()),
        "share_of_stars": float(picked.star.sum() / df.star.sum()),
    }


def bootstrap_gap(y, p_model, p_base, reps: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of the model's log loss minus the baseline's, resampling rookies."""

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = ll(p_model) - ll(p_base)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5]).tolist()


def test(r: pd.DataFrame | None = None) -> dict:
    r = rookies(tables()) if r is None else r
    r = r[(r.yearID <= TEST_LAST) & r.age.notna()]
    out = {
        "window": WINDOW,
        "star": STAR,
        "train": f"{FIRST}-{TRAIN_LAST}",
        "test": f"{TRAIN_LAST + 1}-{TEST_LAST}",
    }
    for kind in ("hitter", "pitcher"):
        d = r[r.kind == kind].reset_index(drop=True)
        train = d.yearID <= TRAIN_LAST
        test_ = ~train
        y = d.star.to_numpy()
        X, ks = features(d, kind, train)
        card = back_of_the_card(d, kind)
        main = card[["ops_plus"]] if kind == "hitter" else card[["era_minus"]]
        roy = d[["roy_share"]].astype(float)
        both = pd.concat([X, roy], axis=1)
        fits = {
            "Back of the card: " + ("OPS+" if kind == "hitter" else "ERA-"): _logit(main[train], y[train]),
            "Back of the card: all of it": _logit(card[train], y[train]),
            "Rookie of the Year voters": _logit(roy[train], y[train]),
            "Model (logistic)": _logit(X[train], y[train]),
            "Model (gradient boosting)": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ).fit(X[train], y[train]),
            "Model + the voters": _logit(both[train], y[train]),
        }
        inputs = {
            "Back of the card: " + ("OPS+" if kind == "hitter" else "ERA-"): main,
            "Back of the card: all of it": card,
            "Rookie of the Year voters": roy,
            "Model (logistic)": X,
            "Model (gradient boosting)": X,
            "Model + the voters": both,
        }
        base_rate = y[train].mean()
        yt = y[test_]
        res = {
            "rookies_train": int(train.sum()),
            "rookies_test": int(test_.sum()),
            "stars_test": int(yt.sum()),
            "base_rate_train": float(base_rate),
            "shrink_strengths": ks,
            "methods": {},
        }
        preds = {}
        for name, m in fits.items():
            p = m.predict_proba(inputs[name][test_])[:, 1]
            preds[name] = p
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            res["methods"][name] = {
                "log_loss": float(-(yt * np.log(pc) + (1 - yt) * np.log(1 - pc)).mean()),
                "brier": float(((p - yt) ** 2).mean()),
                "auc": float(roc_auc_score(yt, p)),
                "top5": top_k(d[test_], p),
            }
        pb = np.full(len(yt), base_rate)
        res["methods"]["Nothing (the base rate)"] = {
            "log_loss": float(-(yt * np.log(pb) + (1 - yt) * np.log(1 - pb)).mean()),
            "brier": float(((pb - yt) ** 2).mean()),
            "auc": 0.5,
        }
        best_card = min(
            ("Back of the card: all of it", next(iter(fits))), key=lambda n: res["methods"][n]["log_loss"]
        )
        res["gap_vs_card_95"] = bootstrap_gap(yt, preds["Model (logistic)"], preds[best_card])
        res["gap_vs_voters_95"] = bootstrap_gap(
            yt, preds["Model (logistic)"], preds["Rookie of the Year voters"]
        )
        res["gap_card_used"] = best_card
        # what the edge is made of: age alone, age with the card's numbers, the model without shrinking
        raw = X.copy()
        rates = (("k", "SO", "lg_k"), ("bb", "BB", "lg_bb"), ("hr", "HR", "lg_hr"))
        if kind == "hitter":
            for name, col, lg in rates:
                raw[f"{name}_skill"] = d[col] / d.pa - d[lg]
            raw["babip_skill"] = d.babip - d.lg_babip
        else:
            for name, col, lg in rates:
                raw[f"{name}_skill"] = d[f"p_{col}"] / d.p_BFP.clip(lower=1) - d[f"p_{lg}"]
        res["parts"] = {}
        for name, Z in (
            ("Age only", d[["age"]].astype(float)),
            ("Age and the back of the card", pd.concat([d[["age"]].astype(float), card], axis=1)),
            ("Model without shrinking", raw),
            ("Model without age", X.drop(columns="age")),
        ):
            p = np.clip(_logit(Z[train], y[train]).predict_proba(Z[test_])[:, 1], 1e-6, 1 - 1e-6)
            res["parts"][name] = {
                "log_loss": float(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()),
                "auc": float(roc_auc_score(yt, p)),
                "top5": top_k(d[test_], p),
            }
            preds[name] = p
        res["gap_age_vs_voters_95"] = bootstrap_gap(yt, preds["Age only"], preds["Rookie of the Year voters"])
        res["gap_model_vs_age_and_card_95"] = bootstrap_gap(
            yt, preds["Model (logistic)"], preds["Age and the back of the card"]
        )
        lr = fits["Model (logistic)"][-1]
        res["coefficients"] = dict(zip(X.columns, map(float, lr.coef_[0]), strict=True))
        out[kind] = res
    return out


PITCH_FIRST, PITCH_TRAIN_LAST = 1988, 2003


def pitch_features(
    d: pd.DataFrame, counts: pd.DataFrame, kind: str, train: pd.Series
) -> tuple[pd.DataFrame, dict]:
    """What the card never shows: how often his swings missed (a hitter), how often he made them miss (a
    pitcher), how often he swung, how often he threw a ball. Each shrunk toward the league like the rest."""
    role = "bat" if kind == "hitter" else "pit"
    c = counts[counts.role == role].rename(columns={"year": "yearID"})
    league = counts[counts.role == "pit"].groupby("year")[["swings", "misses", "pitches", "balls"]].sum()
    m = d[["retroID", "yearID"]].merge(c, on=["retroID", "yearID"], how="left").fillna(0)
    L = league.reindex(d.yearID).reset_index(drop=True)
    ks = {}
    out = {}
    for name, hits, trials, lg in (
        ("miss", m.misses, m.swings, L.misses / L.swings),
        ("swing", m.swings, m.pitches, L.swings / L.pitches),
        ("ball", m.balls, m.pitches, L.balls / L.pitches),
    ):
        rate, ks[name] = shrink(hits, trials, lg, train)
        out[f"{name}_skill"] = rate - lg
    return pd.DataFrame(out).astype(float), ks


def test_pitches(r: pd.DataFrame, counts: pd.DataFrame) -> dict:
    """The rookies since 1988, with and without what their pitches say. Fit on 1988-2003, score 2004-2016."""
    r = r[(r.yearID >= PITCH_FIRST) & (r.yearID <= TEST_LAST) & r.age.notna() & r.retroID.notna()]
    out = {"train": f"{PITCH_FIRST}-{PITCH_TRAIN_LAST}", "test": f"{PITCH_TRAIN_LAST + 1}-{TEST_LAST}"}
    for kind in ("hitter", "pitcher"):
        d = r[r.kind == kind].reset_index(drop=True)
        train = d.yearID <= PITCH_TRAIN_LAST
        test_ = ~train
        y, yt = d.star.to_numpy(), d.star.to_numpy()[test_]
        X, _ = features(d, kind, train)
        P, ks = pitch_features(d, counts, kind, train)
        card = pd.concat([d[["age"]].astype(float), back_of_the_card(d, kind)], axis=1)
        sets = {
            "Age and the back of the card": card,
            "Model": X,
            "Model and the pitches": pd.concat([X, P], axis=1),
            "Age, the card and the pitches": pd.concat([card, P], axis=1),
        }
        res = {
            "rookies_train": int(train.sum()),
            "rookies_test": int(test_.sum()),
            "stars_test": int(yt.sum()),
            "shrink_strengths": ks,
            "methods": {},
        }
        preds = {}
        for name, Z in sets.items():
            fit = _logit(Z[train], y[train])
            p = np.clip(fit.predict_proba(Z[test_])[:, 1], 1e-6, 1 - 1e-6)
            preds[name] = p
            res["methods"][name] = {
                "log_loss": float(-(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()),
                "auc": float(roc_auc_score(yt, p)),
                "top5": top_k(d[test_], p),
            }
            if name == "Model and the pitches":
                res["coefficients"] = dict(zip(Z.columns, map(float, fit[-1].coef_[0]), strict=True))
        res["gap_pitches_vs_model_95"] = bootstrap_gap(yt, preds["Model and the pitches"], preds["Model"])
        res["gap_pitches_vs_card_95"] = bootstrap_gap(
            yt, preds["Model and the pitches"], preds["Age and the back of the card"]
        )
        out[kind] = res
    return out
