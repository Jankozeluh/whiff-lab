"""whiff-lab <step>: fetch the data, parse it, train the models, export the site's tables."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(prog="whiff-lab", description=__doc__)
    ap.add_argument("step", choices=["fetch", "parse", "all"])
    ap.add_argument("--years", help="a range such as 1950-1960 (default: every season with play-by-play)")
    args = ap.parse_args()
    years = None
    if args.years:
        a, _, b = args.years.partition("-")
        years = list(range(int(a), int(b or a) + 1))
    t = time.time()
    if args.step in ("fetch", "all"):
        from whiff_lab.data import fetch

        print("Retrosheet (official release) at", fetch())
    if args.step in ("parse", "all"):
        from whiff_lab.data import build

        games, pas = build(years)
        used = games[games.used]
        print(
            f"{len(games):,} games parsed, {games.score_ok.mean():.2%} end on the game log's score, "
            f"{len(used):,} used; {len(pas):,} plate appearances"
        )
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
