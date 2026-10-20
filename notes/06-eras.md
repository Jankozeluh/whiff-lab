# Eras: what changes, and what doesn't

Baseball in 1910 and baseball in 2010 are not quite the same game. Scoring has swung from {{s.rpg_low.rpg|f2}} runs per team per game ({{s.rpg_low.year|year}}) to {{s.rpg_high.rpg|f2}} ({{s.rpg_high.year|year}}), the ball, the parks and the strategy have changed, and since 2020 even the rules of extra innings. Does a two-run lead in the 7th mean something different in each era?

## The eras

| Era | Seasons | What marks it | Games used |
| --- | --- | --- | --- |
| {{eras.deadball.name}} | {{eras.deadball.first}}-{{eras.deadball.last}} | {{eras.deadball.note}} | {{eras.deadball.games|int}} |
| {{eras.liveball.name}} | {{eras.liveball.first}}-{{eras.liveball.last}} | {{eras.liveball.note}} | {{eras.liveball.games|int}} |
| {{eras.postwar.name}} | {{eras.postwar.first}}-{{eras.postwar.last}} | {{eras.postwar.note}} | {{eras.postwar.games|int}} |
| {{eras.expansion.name}} | {{eras.expansion.first}}-{{eras.expansion.last}} | {{eras.expansion.note}} | {{eras.expansion.games|int}} |
| {{eras.moundlow.name}} | {{eras.moundlow.first}}-{{eras.moundlow.last}} | {{eras.moundlow.note}} | {{eras.moundlow.games|int}} |
| {{eras.offense.name}} | {{eras.offense.first}}-{{eras.offense.last}} | {{eras.offense.note}} | {{eras.offense.games|int}} |
| {{eras.strikeout.name}} | {{eras.strikeout.first}}-{{eras.strikeout.last}} | {{eras.strikeout.note}} | {{eras.strikeout.games|int}} |
| {{eras.runner.name}} | {{eras.runner.first}}-{{s.last_season}} | {{eras.runner.note}} | {{eras.runner.games|int}} |

The borders are the usual ones of baseball history, not something the data chose. They matter less than they look: every era's numbers lean on the others (below).

## Does knowing the era help? A test

Leave one era out, fit the models on all the others, and score them on the era left out. The logistic regression is told the left-out era's scoring level (its run scale); the table and the trees are not. If eras mattered a lot, the model that knows the run environment should win.

| Era left out | Runs per game | Table | Logistic (knows the era's scoring) | Trees | The era's own table (unfair) |
| --- | --- | --- | --- | --- | --- |
| {{eras.deadball.name}} | {{e.deadball.rpg|f2}} | {{e.deadball.Lookup table|f4}} | {{e.deadball.Logistic regression|f4}} | {{e.deadball.Gradient boosting|f4}} | {{e.deadball.own era (in-sample)|f4}} |
| {{eras.liveball.name}} | {{e.liveball.rpg|f2}} | {{e.liveball.Lookup table|f4}} | {{e.liveball.Logistic regression|f4}} | {{e.liveball.Gradient boosting|f4}} | {{e.liveball.own era (in-sample)|f4}} |
| {{eras.postwar.name}} | {{e.postwar.rpg|f2}} | {{e.postwar.Lookup table|f4}} | {{e.postwar.Logistic regression|f4}} | {{e.postwar.Gradient boosting|f4}} | {{e.postwar.own era (in-sample)|f4}} |
| {{eras.expansion.name}} | {{e.expansion.rpg|f2}} | {{e.expansion.Lookup table|f4}} | {{e.expansion.Logistic regression|f4}} | {{e.expansion.Gradient boosting|f4}} | {{e.expansion.own era (in-sample)|f4}} |
| {{eras.moundlow.name}} | {{e.moundlow.rpg|f2}} | {{e.moundlow.Lookup table|f4}} | {{e.moundlow.Logistic regression|f4}} | {{e.moundlow.Gradient boosting|f4}} | {{e.moundlow.own era (in-sample)|f4}} |
| {{eras.offense.name}} | {{e.offense.rpg|f2}} | {{e.offense.Lookup table|f4}} | {{e.offense.Logistic regression|f4}} | {{e.offense.Gradient boosting|f4}} | {{e.offense.own era (in-sample)|f4}} |
| {{eras.strikeout.name}} | {{e.strikeout.rpg|f2}} | {{e.strikeout.Lookup table|f4}} | {{e.strikeout.Logistic regression|f4}} | {{e.strikeout.Gradient boosting|f4}} | {{e.strikeout.own era (in-sample)|f4}} |
| {{eras.runner.name}} | {{e.runner.rpg|f2}} | {{e.runner.Lookup table|f4}} | {{e.runner.Logistic regression|f4}} | {{e.runner.Gradient boosting|f4}} | {{e.runner.own era (in-sample)|f4}} |

The surprise: knowing the era's scoring level did not help. In every era, the table built from the *other* eras is as good as the formula that was told how many runs the era scored, or better. The last column is unfair on purpose (the era's own table, scored on its own games, has memorised them); it marks how much room there is at most, and it is small, a few thousandths.

So the plain answer is: **the odds of a given spot are nearly the same in every era.** A lead is worth about what it was worth a century ago.

## Where the eras do differ

"Nearly" hides a few real differences. To see them, each era's own record is combined with the final model (fitted on all eras) by empirical Bayes, as in step 2: an era moves away from the model only as far as its own games show it should. The fitted strength `k` is between {{ek.min|f0}} and {{ek.max|f0}} games, depending on the era: a spot needs several hundred games in an era before its own record outweighs the model.

![Three spots across the eras](figures/eras.svg)

The clearest case is the late comeback. A home team down a run in the bottom of the 9th, nobody on, nobody out, won {{f.spots.1.eras.deadball.p|pct1}} of the time in the dead-ball era and {{f.spots.1.eras.liveball.p|pct1}} in the live-ball era that came right after; the 90% intervals don't overlap. Fewer runs, fewer comebacks.

## Comebacks follow the runs

That points to a wider pattern. For every game, the lab finds the lowest chance the winner had at any point. {{f.comeback_share.under10|pct1}} of all games were won by a team that was below 10% at some point. By season, that share moves with scoring:

![Comebacks and runs per game, one dot per season](figures/comebacks.svg)

Correlation {{f.comeback_rpg_r|f2}} over {{nseasons}} seasons. The top season was 1930 ({{seasons.1930.under10|pct1}} of games won from below 10%), the lowest 1908 ({{seasons.1908.under10|pct1}}) and, after the dead-ball years, 1968, the year of the pitcher ({{seasons.1968.under10|pct1}}). The odds of a single spot hardly change between eras; how often a game *passes through* a hopeless-looking spot and gets out of it does, because more runs mean more big innings.

## The runner on second

Since 2020 every regular-season extra half-inning starts with a runner on second. It shortened extra innings ({{x.runner.over_in_10th|pct0}} of extra-inning games now end in the 10th, against {{x.before.over_in_10th|pct0}} from 1908 to 2019) and it moved the odds: with the visitors' runner on second, a tie at the start of the 10th is {{tie.runner.top.9|pct1}} for the home team, against {{tie.before.top.9|pct1}} with the bases empty before. The home team won {{x.before.home_win|pct1}} of extra-inning games before the rule and {{x.runner.home_win|pct1}} since (90% interval {{x.runner.home_win_lo|pct1}} to {{x.runner.home_win_hi|pct1}}, {{x.runner.games|int}} games): the edge of batting last may be gone, though a handful of seasons can't settle it.

## What the lab can't see

- **Who is playing.** The model knows the state, not the teams: a 3-run lead is worth more for a team with a great bullpen. Its odds are for an average team against an average team of the era.
- **The park, the weather, the pitcher's pitch count.** Not in the model.
- **Games before 1908, postseason games, and Retrosheet's "deduced" games** (reconstructed from newspaper accounts with partial play-by-play): left out for now.
- **The calculator's intervals** say how sure the combination of an era's record and the model is about the long-run rate in that spot. They are not a forecast of how sure anyone could be about one particular game.

The odds for every spot and every era are in the [calculator](SITE/odds/), and every number in these notes is rebuilt from the data once a year.
