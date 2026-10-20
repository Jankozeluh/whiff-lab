# Rookies: will he be a star?

Collectors, fans and front offices all ask the same thing about a rookie: is this the start of something? This experiment asks it of every rookie since 1932, tests the answer on rookies the model never saw, and turns it into the [Rookie Report](SITE/rookies/). The honest result: a model does find more future stars than the back of the card or the Rookie of the Year voters, and almost all of its edge comes from one number they ignore.

## The question

- **A rookie** is a player in the season he first goes past 130 at-bats (a hitter) or 50 innings (a pitcher), the limits MLB uses for rookie status.
- **A star** is a rookie who makes at least two All-Star teams in the nine seasons after his rookie year. All-Star teams are what card prices and fame follow; they are also a popularity contest (fans vote, every club gets one), which the label inherits.
- **The data** is the [Lahman Baseball Database](https://sabr.org/lahman-database/): every season's statistics, birth dates, positions, All-Star teams and Rookie of the Year votes, © Sean Lahman, CC BY-SA 3.0, read from the R package Lahman ({{rr.lahman}}). Tables derived from it are shared under the same licence.

## The test comes first

Before anything was built, the question was put to a test by time, like experiment 1: everything was fitted on the rookies of {{rk.train}} and scored on the rookies of {{rk.test}}, whose nine following seasons are all in the data: {{rk.hitter.rookies_test|int}} hitters ({{rk.hitter.stars_test|int}} became stars) and {{rk.pitcher.rookies_test|int}} pitchers ({{rk.pitcher.stars_test|int}}).

Five ways to rank a rookie class were compared: the main number on the card (OPS+ for hitters, ERA− for pitchers, both relative to the league), everything on the back of the card (average, home runs, runs batted in, steals, OPS+; wins, ERA−, strikeouts, saves), the share of Rookie of the Year points, age alone, and the model below. Each was turned into a probability by a logistic regression fitted on the training rookies.

![Stars among each year's five highest-ranked rookies](figures/rookie_test.svg)

| Ranked by | Hitters: log loss | AUC | Pitchers: log loss | AUC |
| --- | --- | --- | --- | --- |
| Nothing (every rookie gets the base rate) | {{rk.hitter.methods.Nothing (the base rate).log_loss|f3}} | 0.500 | {{rk.pitcher.methods.Nothing (the base rate).log_loss|f3}} | 0.500 |
| Everything on the back of the card | {{rk.hitter.methods.Back of the card: all of it.log_loss|f3}} | {{rk.hitter.methods.Back of the card: all of it.auc|f3}} | {{rk.pitcher.methods.Back of the card: all of it.log_loss|f3}} | {{rk.pitcher.methods.Back of the card: all of it.auc|f3}} |
| The Rookie of the Year voters | {{rk.hitter.methods.Rookie of the Year voters.log_loss|f3}} | {{rk.hitter.methods.Rookie of the Year voters.auc|f3}} | {{rk.pitcher.methods.Rookie of the Year voters.log_loss|f3}} | {{rk.pitcher.methods.Rookie of the Year voters.auc|f3}} |
| The model | {{rk.hitter.methods.Model (logistic).log_loss|f3}} | {{rk.hitter.methods.Model (logistic).auc|f3}} | {{rk.pitcher.methods.Model (logistic).log_loss|f3}} | {{rk.pitcher.methods.Model (logistic).auc|f3}} |

AUC is the chance that the model ranks a random future star above a random rookie who did not become one (0.5 is a coin flip, 1 is perfect). Resampling the test rookies, the model's log loss minus the card's lies between {{rk.hitter.gap_vs_card_95.0|f3}} and {{rk.hitter.gap_vs_card_95.1|f3}} for hitters and between {{rk.pitcher.gap_vs_card_95.0|f3}} and {{rk.pitcher.gap_vs_card_95.1|f3}} for pitchers (95%): below zero every time.

## The model

A logistic regression on what a scout would ask first: how old is he, how much did he play (and how much before), where does he play (a shortstop or catcher who hits is rarer than a first baseman who does), and how good are his skills. For hitters the skills are his strikeout, walk and home run rates and his average on balls in play; for pitchers, strikeout, walk and home run rates. Each is measured against his league and season, so a 1968 rookie and a 2000 rookie are on the same scale.

The skills are shrunk toward the league's rate by empirical Bayes, the same idea as step 2 of experiment 1: a rate counts as his own only as far as his sample supports it. The fitted strengths say how much a season tells:

| Hitters' rate | Strength k |
| --- | --- |
| Strikeouts per plate appearance | {{rk.hitter.shrink_strengths.k|f0}} |
| Walks | {{rk.hitter.shrink_strengths.bb|f0}} |
| Home runs | {{rk.hitter.shrink_strengths.hr|f0}} |
| Hits on balls in play | {{rk.hitter.shrink_strengths.babip|f0}} |

Strikeouts settle fast: after {{rk.hitter.shrink_strengths.k|f0}} plate appearances a rookie's strikeout rate is half his own. His average on balls in play needs {{rk.hitter.shrink_strengths.babip|f0}} balls in play to be half his own, more than a full season: most of a rookie's batting average beyond his strikeouts and home runs is luck. That is where the report's "luck" line comes from.

## What the edge is made of

| What goes in | Hitters: log loss | stars in top 5 | Pitchers: log loss | stars in top 5 |
| --- | --- | --- | --- | --- |
| Age alone | {{rk.hitter.parts.Age only.log_loss|f3}} | {{rk.hitter.parts.Age only.top5.stars_picked|int}} | {{rk.pitcher.parts.Age only.log_loss|f3}} | {{rk.pitcher.parts.Age only.top5.stars_picked|int}} |
| Age and the back of the card | {{rk.hitter.parts.Age and the back of the card.log_loss|f3}} | {{rk.hitter.parts.Age and the back of the card.top5.stars_picked|int}} | {{rk.pitcher.parts.Age and the back of the card.log_loss|f3}} | {{rk.pitcher.parts.Age and the back of the card.top5.stars_picked|int}} |
| The model without shrinking | {{rk.hitter.parts.Model without shrinking.log_loss|f3}} | {{rk.hitter.parts.Model without shrinking.top5.stars_picked|int}} | {{rk.pitcher.parts.Model without shrinking.log_loss|f3}} | {{rk.pitcher.parts.Model without shrinking.top5.stars_picked|int}} |
| The model without age | {{rk.hitter.parts.Model without age.log_loss|f3}} | {{rk.hitter.parts.Model without age.top5.stars_picked|int}} | {{rk.pitcher.parts.Model without age.log_loss|f3}} | {{rk.pitcher.parts.Model without age.top5.stars_picked|int}} |
| The model | {{rk.hitter.methods.Model (logistic).log_loss|f3}} | {{rk.hitter.methods.Model (logistic).top5.stars_picked|int}} | {{rk.pitcher.methods.Model (logistic).log_loss|f3}} | {{rk.pitcher.methods.Model (logistic).top5.stars_picked|int}} |

Age alone beats everything on the back of the card for hitters, and beats the Rookie of the Year voters for both (the voters' log loss minus age's: the interval for hitters is {{rk.hitter.gap_age_vs_voters_95.0|f3}} to {{rk.hitter.gap_age_vs_voters_95.1|f3}}). Take age out of the model and it loses most of its edge. Put age next to the card's numbers and you have nearly all of the model: for hitters the difference is {{rk.hitter.gap_model_vs_age_and_card_95.0|f3}} to {{rk.hitter.gap_model_vs_age_and_card_95.1|f3}}, not measurable; for pitchers the model keeps a small, real edge ({{rk.pitcher.gap_model_vs_age_and_card_95.0|f3}} to {{rk.pitcher.gap_model_vs_age_and_card_95.1|f3}}), from strikeouts with the luck taken out.

![The share of rookies who became stars, by age](figures/rookie_ages.svg)

Why age? A 20-year-old rookie who holds his own has a decade of growth ahead; a 27-year-old with the same line is at his peak. Voters reward the best season, cards print it, and the birth date sits in small type in the corner.

## His pitches don't help

Since 1988 Retrosheet has every pitch, which the card never shows: how often a hitter's swings miss, how often he swings, how often a pitcher makes hitters miss or throws a ball. Counted for every rookie of {{rp.train}} and {{rp.test}}, shrunk like the other rates, and added to the model (fitted on {{rp.train}}, scored on {{rp.test}}):

| | Hitters: log loss | stars in top 5 | Pitchers: log loss | stars in top 5 |
| --- | --- | --- | --- | --- |
| Age and the back of the card | {{rp.hitter.methods.Age and the back of the card.log_loss|f3}} | {{rp.hitter.methods.Age and the back of the card.top5.stars_picked|int}} | {{rp.pitcher.methods.Age and the back of the card.log_loss|f3}} | {{rp.pitcher.methods.Age and the back of the card.top5.stars_picked|int}} |
| The model | {{rp.hitter.methods.Model.log_loss|f3}} | {{rp.hitter.methods.Model.top5.stars_picked|int}} | {{rp.pitcher.methods.Model.log_loss|f3}} | {{rp.pitcher.methods.Model.top5.stars_picked|int}} |
| The model and the pitches | {{rp.hitter.methods.Model and the pitches.log_loss|f3}} | {{rp.hitter.methods.Model and the pitches.top5.stars_picked|int}} | {{rp.pitcher.methods.Model and the pitches.log_loss|f3}} | {{rp.pitcher.methods.Model and the pitches.top5.stars_picked|int}} |

The pitches' effect lies between {{rp.hitter.gap_pitches_vs_model_95.0|f3}} and {{rp.hitter.gap_pitches_vs_model_95.1|f3}} for hitters and {{rp.pitcher.gap_pitches_vs_model_95.0|f3}} and {{rp.pitcher.gap_pitches_vs_model_95.1|f3}} for pitchers: nothing measurable. A rookie's strikeouts already carry what his misses would say. A fancier model (gradient boosting) did worse than the plain logistic regression, with a few thousand rookies to learn from.

## The report

The final models are fitted on every rookie whose nine seasons are in ({{rr.fitted_on}}) and applied to the latest classes. For each rookie the report adds:

- **The five most similar rookies since 1932.** Similar in the model's own terms: each input is put on a common scale and weighted by how much it moves the odds, so two rookies of the same age, playing time and skills are close even if one hit .300 by luck. What became of them is the plainest way to read a probability.
- **Luck.** A hitter's average with his strikeouts and home runs as they were but an ordinary share of hits on balls in play (his own, shrunk); a pitcher's ERA rebuilt from his strikeouts, walks and home runs, shrunk (the FIP formula, on the league's ERA scale).
- **Players from Japan's, Korea's or Taiwan's top leagues** are marked: they arrive in MLB as grown pros, and the model's age penalty is unfair to them.

The percentages are calibrated: of the test hitters the model gave a {{rk.hitter.calibration.3.from|pct0}}–{{rk.hitter.calibration.3.to|pct0}} chance, {{rk.hitter.calibration.3.happened|pct0}} became stars; {{rk.hitter.calibration.4.from|pct0}}–{{rk.hitter.calibration.4.to|pct0}}, {{rk.hitter.calibration.4.happened|pct0}}; over {{rk.hitter.calibration.5.from|pct0}}, {{rk.hitter.calibration.5.happened|pct0}}. At the low end it is a little generous: of the hitters it gave under 5% (on average {{rk.hitter.calibration.0.said|pct1}}), {{rk.hitter.calibration.0.happened|pct1}} became stars.

## What it can't see

Injuries, minor-league records, scouting reports, how hard he hits the ball, how his pitches move: professional projections use all of these, and none of them is free and open. The report is a well-tested reading of what is on every rookie's record, not a scouting report. New rookie classes arrive when the Lahman database publishes a season, usually early in the following year.
