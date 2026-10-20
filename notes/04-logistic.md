# Step 3: a formula (logistic regression)

The table and the shrunk table store one number per state, about nine thousand in all. This step throws the table away and asks: can a handful of numbers, built from what a fan knows about baseball, do the same job?

## Three things a fan knows

**How many runs the batting team can expect before the inning ends.** That depends only on the runners and the outs, and the data gives it directly: from every state of innings 1 to 8, the runs the batting team scored before the third out, on average ({{h.train}}).

| Runners | 0 out | 1 out | 2 out |
| --- | --- | --- | --- |
| Bases empty | {{h.re.0|f2}} | {{h.re.1|f2}} | {{h.re.2|f2}} |
| First | {{h.re.3|f2}} | {{h.re.4|f2}} | {{h.re.5|f2}} |
| Second | {{h.re.6|f2}} | {{h.re.7|f2}} | {{h.re.8|f2}} |
| First and second | {{h.re.9|f2}} | {{h.re.10|f2}} | {{h.re.11|f2}} |
| Third | {{h.re.12|f2}} | {{h.re.13|f2}} | {{h.re.14|f2}} |
| First and third | {{h.re.15|f2}} | {{h.re.16|f2}} | {{h.re.17|f2}} |
| Second and third | {{h.re.18|f2}} | {{h.re.19|f2}} | {{h.re.20|f2}} |
| Bases loaded | {{h.re.21|f2}} | {{h.re.22|f2}} | {{h.re.23|f2}} |

This is the run expectancy table, one of the oldest tools of baseball analysis. A fresh inning is worth about half a run.

**How much game is left.** Each team has so many half-innings still to bat, and each is worth, on average, a fresh inning's runs.

**How many runs a season scores.** Half a run per inning in 2016 is not half a run in 1930. Each season's numbers are scaled by its runs per game.

## One number: z

Put them together into the lead the home team can expect at the end:

```
expected lead = lead now
              + the batting team's expected runs for the rest of this inning (+ for home, - for visitors)
              + a fresh inning's runs × (home half-innings left - visitors' half-innings left)
```

How sure is that? Runs come in bunches, so the uncertainty grows with the time left: roughly with the square root of the half-innings still to play. Divide one by the other:

```
z = expected lead / √(half-innings left × run scale + 0.25)
```

A big `z` means a lead that is big compared with what can still change.

![The home team's record against z](figures/logistic.svg)

Every orange dot is a group of plate appearances with about the same `z`. They lie on an S-shaped curve: that is the shape a logistic regression draws.

## The regression

Logistic regression says the log-odds of a home win are a weighted sum of a few inputs:

```
log(p / (1 - p)) = a + b₁·z + b₂·(expected lead) + b₃·(tie, bottom half) + b₄·(tie, 9th or later) + b₅·(z in the 9th or later)
```

The last three inputs let the formula bend where baseball is lumpy: a tie in the bottom half, when the home team can walk off; the 9th inning and later, when there is no "rest of the game" left. The weights are fitted to the counts by maximum likelihood (the same log loss as in the test, minimised on the training years):

| Weight | Value |
| --- | --- |
| a (intercept) | {{h.logistic.coef.intercept|f3}} |
| b₁ (z) | {{h.logistic.coef.z|f3}} |
| b₂ (expected lead) | {{h.logistic.coef.mu|f3}} |
| b₃ (tie, bottom) | {{h.logistic.coef.tie_bottom|f3}} |
| b₄ (tie, 9th+) | {{h.logistic.coef.tie_late|f3}} |
| b₅ (z, 9th+) | {{h.logistic.coef.z_late|f3}} |

Six numbers instead of nine thousand. `z` does nearly all the work.

## The rule it knew in advance

Since 2020, an extra inning starts with a runner on second. The formula can be told so: in the 10th, the home team's coming half-inning is worth what a runner on second with nobody out is worth ({{h.re.6|f2}} runs), not a fresh inning's {{h.re.0|f2}}. No data needed. The table and the shrunk table, fitted on games before 2020, have never seen the rule.

## Result

| Model | All | Rare states | Extra innings since 2020 |
| --- | --- | --- | --- |
| Lookup table | {{h.table.ll.all|f4}} | {{h.table.ll.rare|f4}} | {{h.table.ll.x20|f4}} |
| Shrunk table | {{h.shrunk.ll.all|f4}} | {{h.shrunk.ll.rare|f4}} | {{h.shrunk.ll.x20|f4}} |
| Logistic regression | {{h.logistic.ll.all|f4}} | {{h.logistic.ll.rare|f4}} | {{h.logistic.ll.x20|f4}} |

On average the formula is a touch worse than the table: six numbers can't capture every quirk of nine thousand states (in the 9th inning especially, where it scores {{h.logistic.ll.ninth|f4}} against the table's {{h.table.ll.ninth|f4}}). But it is the best of the three on rare states, where it never says 0% or 100% from four games, and by far the best in the extra innings since 2020, {{h.logistic.gain.x20|pct0}} better than the table, because it was told the new rule. A model built on understanding can be right about situations no data has shown it yet.

**Next:** [Step 4: trees](05-boosting.md).
