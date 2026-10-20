"""whiff-lab <step>: fetch the data, parse it, test the models, export the site's tables, build the notes.

fetch     Retrosheet's event files and game logs (only what changed since the last fetch)
parse     event files -> data/work/games.parquet and data/work/pa.parquet
evaluate  the four models on the honest test (fit 1908-2016, score 2017 on) -> results/holdout.json
export    the final model and the findings -> site/data/ and results/findings.json
notes     the lab notes' figures (lab-notes/figures/) and their pages on the site (site/notes/)
all       every step in order
"""

from __future__ import annotations

import argparse
import time

from whiff_lab.data import WORK

STEPS = ("fetch", "parse", "evaluate", "export", "notes")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="whiff-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=[*STEPS, "all", "rookies"])
    args = ap.parse_args()
    for step in STEPS if args.step == "all" else (args.step,):
        t = time.time()
        print(f"== {step}", flush=True)
        if step == "fetch":
            from whiff_lab.data import fetch

            commit = fetch()
            from whiff_lab.data import record_source

            record_source(commit)
            print("Retrosheet (official release, Chadwick Bureau mirror) at commit", commit)
        elif step == "parse":
            from whiff_lab.data import build

            games, pas = build()
            print(
                f"{len(games):,} games parsed, {games.score_ok.mean():.2%} end on the game log's score, "
                f"{int(games.used.sum()):,} used; {len(pas):,} plate appearances"
            )
        elif step == "evaluate":
            from whiff_lab.evaluate import era_holdout, holdout, rule_change, save

            r = holdout()
            save(r, "holdout")
            save(rule_change(), "rule_change")
            save(era_holdout(), "era_holdout")
            for m in r["models"]:
                print(f"  {m['name']:32s} log loss {m['log_loss']['all']:.4f}  Brier {m['brier']['all']:.4f}")
        elif step == "export":
            from whiff_lab.export import run

            f = run()
            print(f"  {f['summary']['games_used']:,} games, {len(f['comebacks'])} comebacks listed")
        elif step == "rookies":
            from whiff_lab.evaluate import save
            from whiff_lab.rookies import fetch, rookies, tables, test

            r = rookies(tables(fetch()))
            r.to_parquet(WORK / "rookies.parquet", index=False)
            print(" ", save(test(r), "rookies_test"))
            from whiff_lab.pitches import build as pitch_counts
            from whiff_lab.rookies import test_pitches

            print(" ", save(test_pitches(r, pitch_counts()), "rookies_pitches_test"))
        elif step == "notes":
            from whiff_lab.site import build_notes

            for path in build_notes():
                print(" ", path)
        print(f"   {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
