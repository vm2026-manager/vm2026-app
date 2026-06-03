# Long run pick audit

## Kort konklusion

Auditten viser, at long_run stadig har flere value-/budgetvalg fra mellemhold. Det skyldes især, at scoreformlen stadig giver høj vægt til EV/value, mens turneringsstyrkeleddet og weak-team-penalty er relativt milde.

Labels: true_long_run_pick=5, acceptable_tradeoff=3, value_filler=1, questionable_long_run_pick=2.

## Dødboldsnote

data/set_piece_takers.csv findes, men dødbolde er ikke integreret i denne audit. Dødbolde bør kun være et lille tie-breaker-lag. De må ikke alene forklare eller retfærdiggøre long_run-valg, og en samlet set-piece-bonus bør være lavt capped, fx omkring 2-5 pct. af relevant strategi-score.

## Særligt tjek

| Spiller | Land | Label | Hvorfor valgt | Bedste stærkere alternativ | Anbefaling |
| --- | --- | --- | --- | --- | --- |
| Patrick Agyemang | USA | value_filler | høj EV/value trækker kraftigt; svagere/mellemhold i vinderodds; acceptabel starterprofil; samlet long_run-score 5.592 | Luiz Henrique (BRA, score 3.900918) | Ligner primært value/budget-fill fra mellemhold, ikke en ren stor-nationslogik. |
| Kerem Akturkoglu | TUR | questionable_long_run_pick | høj EV/value trækker kraftigt; svagere/mellemhold i vinderodds; stærk starterprofil; samlet long_run-score 5.524 | Casemiro (BRA, score 5.284413) | Der findes et stærkere nationsalternativ tæt på eller over scoren. |
| Roberto Alvarado | MEX | questionable_long_run_pick | høj EV/value trækker kraftigt; svagere/mellemhold i vinderodds; usikker starterprofil; samlet long_run-score 4.934 | Fabian Rieder (SUI, score 3.962972) | Svagere turneringsprofil kombineret med manuel/start-risiko. |

## Robuste long_run-kernevalg

| Spiller | Land | Pos | Score | Winner odds | Turneringsstyrke |
| --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 5.655984 | 12 | 0.597722 |
| Ruben Dias | POR | DEF | 4.541743 | 12 | 0.597722 |
| Vinicius Junior | BRA | FWD | 7.067443 | 9 | 0.691467 |
| Cristiano Ronaldo | POR | FWD | 7.039453 | 12 | 0.597722 |
| Joshua Kimmich | GER | MID | 5.664139 | 13 | 0.622665 |

## Alle long_run-valg

| Spiller | Land | Pos | EV | Score | Start | Winner odds | Turnering | Label | Bedste stærkere alternativ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 3.974216 | 5.655984 | 0.8431 | 12 | 0.597722 | true_long_run_pick | William Saliba (FRA, 4.006859) |
| Ruben Dias | POR | DEF | 2.812349 | 4.541743 | 0.8506 | 12 | 0.597722 | true_long_run_pick | William Saliba (FRA, 4.006859) |
| Timothy Castagne | BEL | DEF | 3.049967 | 4.34023 | 0.8194 | 34 | 0.397237 | acceptable_tradeoff | Cristian Romero (ARG, 3.171622) |
| Vinicius Junior | BRA | FWD | 5.044969 | 7.067443 | 0.8103 | 9 | 0.691467 | true_long_run_pick | Lautaro Martinez (ARG, 6.914008) |
| Cristiano Ronaldo | POR | FWD | 4.947107 | 7.039453 | 0.97 | 12 | 0.597722 | true_long_run_pick | Mikel Oyarzabal (ESP, 6.54886) |
| Patrick Agyemang | USA | FWD | 4.112482 | 5.592055 | 0.8571 | 41 | 0.267464 | value_filler | Luiz Henrique (BRA, 3.900918) |
| Diogo Costa | POR | GK | 3.560318 | 5.223306 | 0.7925 | 12 | 0.597722 | acceptable_tradeoff | Mike Maignan (FRA, 5.662455) |
| Joshua Kimmich | GER | MID | 3.719618 | 5.664139 | 0.9504 | 13 | 0.622665 | true_long_run_pick | Casemiro (BRA, 5.284413) |
| Kerem Akturkoglu | TUR | MID | 4.077992 | 5.524409 | 0.9107 | 101 | 0.200606 | questionable_long_run_pick | Casemiro (BRA, 5.284413) |
| Oscar Bobb | NOR | MID | 3.563792 | 5.098807 | 0.97 | 26 | 0.38982 | acceptable_tradeoff | Manu Koné (FRA, 4.724498) |
| Roberto Alvarado | MEX | MID | 4.125843 | 4.933902 | 0.7246 | 81 | 0.20066 | questionable_long_run_pick | Fabian Rieder (SUI, 3.962972) |

## Forslag til fremtidig kalibrering (ikke implementeret)

- Gør weak_tournament_team_penalty hårdere for hold under en klar turneringsstyrkegrænse.
- Øg tournament_strength_bonus, så store nationer fylder mere end billige value-spillere.
- Tillad value-spillere fra mellemhold kun når EV er markant høj, conditional_start_prob er stærk, og der ikke findes et rimeligt stærkere nationsalternativ.
- Undgå at budgetpresset alene tvinger fyldspillere ind i long_run.
