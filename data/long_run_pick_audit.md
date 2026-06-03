# Long run pick audit

## Kort konklusion

Auditten finder ikke tydelige value-fillers; long_run-valgene passer overvejende til store nationer og starterprofil.

Labels: true_long_run_pick=10, acceptable_tradeoff=1, value_filler=0, questionable_long_run_pick=0.

## Dødboldsnote

data/set_piece_takers.csv findes, men dødbolde er ikke integreret i denne audit. Dødbolde bør kun være et lille tie-breaker-lag. De må ikke alene forklare eller retfærdiggøre long_run-valg, og en samlet set-piece-bonus bør være lavt capped, fx omkring 2-5 pct. af relevant strategi-score.

## Særligt tjek

| Spiller | Land | Label | Hvorfor valgt | Bedste stærkere alternativ | Anbefaling |
| --- | --- | --- | --- | --- | --- |

## Robuste long_run-kernevalg

| Spiller | Land | Pos | Score | Winner odds | Turneringsstyrke |
| --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 5.92276 | 12 | 0.597722 |
| Aymeric Laporte | ESP | DEF | 5.247334 | 5.5 | 0.957933 |
| Ruben Dias | POR | DEF | 5.221486 | 12 | 0.597722 |
| Mikel Oyarzabal | ESP | FWD | 7.608861 | 5.5 | 0.957933 |
| Nico Williams | ESP | FWD | 7.239726 | 5.5 | 0.957933 |
| Declan Rice | ENG | MID | 5.896083 | 6.5 | 0.809364 |
| Giovani Lo Celso | ARG | MID | 5.838152 | 9 | 0.730698 |
| Rodrigo de Paul | ARG | MID | 5.811481 | 9 | 0.730698 |
| Manu Koné | FRA | MID | 5.776565 | 7 | 0.791494 |
| Aurelien Tchouameni | FRA | MID | 5.764786 | 7 | 0.791494 |

## Alle long_run-valg

| Spiller | Land | Pos | EV | Score | Start | Winner odds | Turnering | Label | Bedste stærkere alternativ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 3.974216 | 5.92276 | 0.8431 | 12 | 0.597722 | true_long_run_pick | Marc Cucurella (ESP, 5.025435) |
| Aymeric Laporte | ESP | DEF | 1.647224 | 5.247334 | 0.902 | 5.5 | 0.957933 | true_long_run_pick |  |
| Ruben Dias | POR | DEF | 2.812349 | 5.221486 | 0.8506 | 12 | 0.597722 | true_long_run_pick | Cristian Romero (ARG, 4.405474) |
| Mikel Oyarzabal | ESP | FWD | 4.133042 | 7.608861 | 0.9432 | 5.5 | 0.957933 | true_long_run_pick |  |
| Nico Williams | ESP | FWD | 3.744479 | 7.239726 | 0.9091 | 5.5 | 0.957933 | true_long_run_pick |  |
| Diogo Costa | POR | GK | 3.560318 | 5.621557 | 0.7925 | 12 | 0.597722 | acceptable_tradeoff | Mike Maignan (FRA, 5.950781) |
| Declan Rice | ENG | MID | 2.764994 | 5.896083 | 0.8706 | 6.5 | 0.809364 | true_long_run_pick | Mikel Merino (ESP, 5.3013) |
| Giovani Lo Celso | ARG | MID | 3.082398 | 5.838152 | 0.8553 | 9 | 0.730698 | true_long_run_pick | Elliot Anderson (ENG, 5.464295) |
| Rodrigo de Paul | ARG | MID | 2.957481 | 5.811481 | 0.9545 | 9 | 0.730698 | true_long_run_pick | Elliot Anderson (ENG, 5.464295) |
| Manu Koné | FRA | MID | 2.799984 | 5.776565 | 0.8571 | 7 | 0.791494 | true_long_run_pick | Jordan Henderson (ENG, 4.956174) |
| Aurelien Tchouameni | FRA | MID | 2.787585 | 5.764786 | 0.88 | 7 | 0.791494 | true_long_run_pick | Jordan Henderson (ENG, 4.956174) |

## Forslag til fremtidig kalibrering (ikke implementeret)

- Gør weak_tournament_team_penalty hårdere for hold under en klar turneringsstyrkegrænse.
- Øg tournament_strength_bonus, så store nationer fylder mere end billige value-spillere.
- Tillad value-spillere fra mellemhold kun når EV er markant høj, conditional_start_prob er stærk, og der ikke findes et rimeligt stærkere nationsalternativ.
- Undgå at budgetpresset alene tvinger fyldspillere ind i long_run.
