# Round Context Quality Audit

Audit af eksisterende round-/fixture-kontekst. Ingen EV eller strategioutput er genberegnet.

## Fordeling

| round_context_quality | rows |
|---|---:|
| distributed_but_plausible | 13 |
| real_fixture_specific_context | 2 |
| missing_ev_source | 1 |

## Fokusspillere

| player_name | team | position | round_1_ev | round_2_ev | round_3_ev | round_context_quality | suspected_issue | recommended_next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | FWD | 1.344 | 1.5231 | 1.1997 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Erling Haaland | NOR | FWD | 1.543 | 1.2006 | 0.6297 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Manuel Neuer | GER | GK | 1.9059 | 1.1177 | 1.005 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Jules Kounde | FRA | DEF | 0.9653 | 1.2076 | 0.676 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Wesley Franca | BRA | DEF | 0.769 | 0.948 | 0.6928 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Raphinha | BRA | FWD | 0.2041 | 0.1625 | 0.119 | real_fixture_specific_context | Fixture-specific round context exists. | Treat as model/role question, not missing data. |
| Deniz Undav | GER | FWD | 0.5467 | 0.4216 | 0.3165 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Jurrien Timber | NED | DEF | 0.6471 | 0.7492 | 0.7347 | distributed_but_plausible | Round EV appears distributed from fixture-specific match output. | No data repair required; use model calibration audit if player still looks wrong. |
| Mahmoud Trezeguet | EGY | MID | 0.0652 | 0.1339 | 0.1413 | real_fixture_specific_context | Fixture-specific round context exists. | Treat as model/role question, not missing data. |

## Konklusion

- Mange remaining flags skyldes ikke manglende spiller-match, men svag eller utilstrækkeligt forklarende round context.
- Haaland og Kane har reel model-EV, men deres specifikke runde-EV er relativt svag ift. premium-forventning; det peger på round/fixture-ceiling audit snarere end simpel value-fejl.
- Kounde og Neuer er rene missing EV-source cases og bør ikke bruges som modelkalibreringsbevis.
- Raphinha, Wesley Franca, Trezeguet og Timber har round context, men auditten viser fortsat, at rundeværdien skal forklares bedre, før de bruges som argument for vægtændring.
