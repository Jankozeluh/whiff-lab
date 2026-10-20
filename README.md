# Whiff Lab

Open baseball models built only from free, openly licensed data, in the look of [The Daily Whiff](https://thedailywhiff.com). Each experiment asks one question, answers it step by step from the plainest model to the strongest, tests every step on seasons it never saw, and explains it in plain words.

## Experiment 1: What are the odds?

A win probability model for any game state (inning, half, outs, runners, score) in every era since 1908, learned from 15.2 million plate appearances of 197,645 Retrosheet games. The calculator gives the home team's chance to win, a 90% interval, how many real games reached that spot in the era, and the latest of them.

What it found (the numbers are rebuilt every year; the site and `results/` always have the latest):

- **A lookup table of game states is very hard to beat.** On 2017-2025, scored after fitting on 1908-2016, the table cuts the log loss from 0.693 (coin flip) to 0.467. Empirical Bayes shrinkage gains a little overall and about 9% on rare states; a six-number logistic regression built on run expectancy is best on rare states and, told the new rule, in the extra innings since 2020; gradient boosting ties the table on average and is the best once it has seen a few seasons of the runner on second.
- **The odds of a given spot barely change across eras**, and telling a model the era's scoring level does not help it predict a left-out era. Where eras differ, it follows the runs: a home team down one in the bottom of the 9th came back 15% of the time in the dead-ball era, 20% in the live-ball era.
- **Comebacks follow scoring.** The share of games won by a team that fell below 10% moves with runs per game (correlation 0.76 over 118 seasons). The longest odds ever overcome: the Cubs on June 29, 1952, down 8-2 with two outs and nobody on in the 9th (about 1 in 14,000).
- **The runner on second moved the odds.** A tie at the start of the 10th was 52% for the home team; with the visitors' free runner it is 49%, and the home team's share of extra-inning wins fell from 52.4% to 50.0%.

## Experiment 2: Rookie Report

Every rookie of the last two seasons, with his chance of making at least two All-Star teams in his next nine seasons, the five most similar rookies since 1932 and what became of them, and how much of his first season was luck.

Tested before it was built, on the rookies of 1995-2016 after fitting on 1932-1994: among each year's five highest-ranked rookies, the model found 59 future star hitters, the back of the card 43 and the Rookie of the Year voters 46 (pitchers: 36, 23 and 26). Most of the edge is one number the card prints in small type: age. Age alone beats everything else on the card for hitters, and the voters for both. Shrinking the rookie's rates toward the league barely changed the result, and adding his pitches (how often his swings miss, since 1988) did not measurably help.

The [lab notes](lab-notes/) explain every step: the data and the parser, the lookup table and how forecasts are scored, empirical Bayes, logistic regression, gradient boosting, the eras, and the rookies.

## Data

- [Retrosheet](https://www.retrosheet.org) play-by-play (regular-season event files, AL and NL, 1908 on) and game logs, from Retrosheet's official releases as mirrored by the [Chadwick Baseball Bureau](https://github.com/chadwickbureau/retrosheet) (branch `official`). The raw files are not in this repository; `whiff-lab fetch` downloads only what it needs and, later, only what changed.

  The information used here was obtained free of charge from and is copyrighted by Retrosheet.  Interested parties may contact Retrosheet at "www.retrosheet.org".

- The [Lahman Baseball Database](https://sabr.org/lahman-database/) (season statistics, birth dates, positions, All-Star teams, Rookie of the Year votes) for the Rookie Report, © Sean Lahman, licensed CC BY-SA 3.0, read from the R package [Lahman on CRAN](https://cran.r-project.org/package=Lahman). The tables derived from it (`results/rookies_*.json`, `site/data/rookies/`) are shared under CC BY-SA 3.0 with the same attribution.

- The typeface is [Archivo](https://github.com/Omnibus-Type/Archivo), SIL Open Font License 1.1 (`site/assets/fonts/OFL-Archivo.txt`).

## Licence

The code is under the [MIT licence](LICENSE). The data and everything derived from it (`results/`, `site/data/`) carry Retrosheet's notice above; what comes from the Lahman database is CC BY-SA 3.0. The Archivo typeface is under the SIL Open Font License. The Daily Whiff mark (`site/assets/whiff-logo-chalk.png`, `site/assets/whiff-site-icon.png`) belongs to The Daily Whiff and is not covered by the MIT licence.

Only free, openly licensed data goes into this lab. If a later experiment needs more, it will be the Chadwick Register (ODC-By, with attribution).

## Rebuild it yourself

Needs [uv](https://docs.astral.sh/uv/) and git; everything else installs itself. The full rebuild takes a few minutes and about 1.5 GB of disk.

```bash
uv run whiff-lab all
```

That runs, in order: `fetch` (Retrosheet's files into `data/raw/`), `parse` (one row per plate appearance into `data/work/`, every game checked against its final score), `evaluate` (the model ladder on the honest test, into `results/`), `export` (the final model, the era tables and the findings, into `site/data/` and `results/`), `rookies` (the Lahman database, the rookie tests and the Rookie Report) and `notes` (the site's pages and the lab notes with the latest numbers and figures). Each step can be run alone, for example `uv run whiff-lab evaluate`.

To look at the site locally:

```bash
python3 -m http.server 8000 --directory site
```

Tests and lint: `uv run pytest` and `uv run ruff check src tests`.

## Layout

| Path | What |
| --- | --- |
| `src/whiff_lab/retrosheet.py` | the event-file parser: game state before every plate appearance |
| `src/whiff_lab/data.py` | fetching Retrosheet, parsing every season, which games are used |
| `src/whiff_lab/features.py` | game states as counts, the eras, run expectancy |
| `src/whiff_lab/models.py` | the four models: table, shrunk table, logistic regression, gradient boosting |
| `src/whiff_lab/evaluate.py` | the honest test, the rule-change test, the leave-one-era-out test |
| `src/whiff_lab/lab.py`, `export.py` | the final model, the era tables, the findings, the site's data |
| `src/whiff_lab/rookies.py`, `pitches.py` | experiment 2: rookie seasons, the tests, the report; pitches per player since 1988 |
| `src/whiff_lab/site.py`, `figures.py` | the pages and the lab notes, with every number filled in from `results/` |
| `pages/`, `notes/` | the sources of the site's pages and of the lab notes (edit these) |
| `lab-notes/` | the lab notes as built, with their figures |
| `site/` | the static site: plain HTML, CSS and JavaScript, no build step for the browser |
| `results/` | the numbers behind everything (small JSON files) |

## Automation

GitHub Actions, free on this public repository, with no account or service to pay for:

- `test.yml`: lint and tests on every push.
- `rebuild.yml`: on the 15th of every month, checks whether Retrosheet has published a new release (a new season comes around December) or the Lahman database a new version (early in the year). If so, it fetches what changed, reruns everything, and commits the new `results/`, `site/` and `lab-notes/`. Most months it stops after one check. It can also be run by hand from the Actions tab.

The site is not hosted yet.
