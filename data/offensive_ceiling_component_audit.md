# Offensive Ceiling Component Audit

Ren audit af eksisterende modeloutput. Ingen optimizer, strategioutput, EV-modelkalibrering eller frontend er koert.

## Komponenter i outputtet

- Eksisterer eksplicit: goal_ev, assist_ev, shots_on_target_ev, clean_sheet_ev, result/team-score/on-pitch/start-minutes, fixture multipliers, round EV, price_quality og strategy_score.
- Findes ikke eksplicit i nuvaerende output: match_winner_goal_ev, hattrick_ev, player_of_the_match_ev og penalty_ev. De er derfor blanke i CSV'en og maa ikke tolkes som nul-effekt.

## Kort konklusion

- Premium/ceiling-spillere har i gennemsnit goal_component 0.512, price_value_component 0.451.
- Billige FWD-rækker har i gennemsnit price_value_component 0.308, hvilket viser at price/value-laget ofte er positivt for value-spillere.
- Lav-upside MID-rækker har i gennemsnit strategy_score 5.811; de får især hjælp af start_security_effect og fixture/round-kontekst.
- Det stærkeste audit-signal er kombinationen af manglende eksplicit ceiling-bonusser og et tydeligt price/value-lag, ikke én isoleret datakolonne.

## Suspected issue distribution

| suspected_issue | rows |
|---|---:|
| no_component_issue_from_available_outputs | 620 |
| premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev | 168 |
| cheap_fwd_boosted_by_price_value_layer | 28 |
| low_upside_player_helped_by_price_value_layer | 28 |

## Premium og ceiling-spillere

| player_name | strategy | goal_component | assist_component | shot_component | round_context_component | price_value_component | strategy_score | suspected_issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Erling Haaland | next_round | 0.5489 | 0.1605 | 0.4088 | 3.3946 | 0.5441 | 7.8795 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Erling Haaland | round1_2 | 0.5489 | 0.1605 | 0.4088 | 5.2185 | 0.5441 | 8.402 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Erling Haaland | group_stage | 0.5489 | 0.1605 | 0.4088 | 5.7491 | 0.5441 | 9.1425 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Erling Haaland | long_run | 0.5489 | 0.1605 | 0.4088 | 0.0 | 0.5441 | 5.0225 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Harry Kane | next_round | 0.7064 | 0.2197 | 0.4194 | 2.9568 | 0.2472 | 7.2511 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Harry Kane | round1_2 | 0.7064 | 0.2197 | 0.4194 | 5.4284 | 0.2472 | 8.9936 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Harry Kane | group_stage | 0.7064 | 0.2197 | 0.4194 | 6.8229 | 0.2472 | 10.9184 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Harry Kane | long_run | 0.7064 | 0.2197 | 0.4194 | 0.0 | 0.2472 | 7.3675 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Jamal Musiala | next_round | 0.2915 | 0.2414 | 0.113 | 2.3858 | 0.3959 | 6.0753 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Jamal Musiala | round1_2 | 0.2915 | 0.2414 | 0.113 | 3.6485 | 0.3959 | 6.5006 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Jamal Musiala | group_stage | 0.2915 | 0.2414 | 0.113 | 4.0441 | 0.3959 | 7.0159 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Jamal Musiala | long_run | 0.2915 | 0.2414 | 0.113 | 0.0 | 0.3959 | 5.3428 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Kylian Mbappe | next_round | 0.781 | 0.2856 | 0.6104 | 3.9729 | -0.0715 | 9.0159 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Kylian Mbappe | round1_2 | 0.781 | 0.2856 | 0.6104 | 6.7927 | -0.0715 | 10.9618 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Kylian Mbappe | group_stage | 0.781 | 0.2856 | 0.6104 | 7.8347 | -0.0715 | 12.175 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Kylian Mbappe | long_run | 0.781 | 0.2856 | 0.6104 | 0.0 | -0.0715 | 7.6927 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Luis Diaz | next_round | 0.4174 | 0.263 | 0.1631 | 2.5016 | 0.6237 | 6.4254 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Luis Diaz | round1_2 | 0.4174 | 0.263 | 0.1631 | 4.3454 | 0.6237 | 7.4812 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Luis Diaz | group_stage | 0.4174 | 0.263 | 0.1631 | 4.7066 | 0.6237 | 7.8608 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Luis Diaz | long_run | 0.4174 | 0.263 | 0.1631 | 0.0 | 0.6237 | 2.7511 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Michael Olise | next_round | 0.3251 | 0.248 | 0.1277 | 1.9596 | 0.9689 | 5.5735 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Michael Olise | round1_2 | 0.3251 | 0.248 | 0.1277 | 3.3372 | 0.9689 | 6.39 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Michael Olise | group_stage | 0.3251 | 0.248 | 0.1277 | 3.8278 | 0.9689 | 6.9579 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |
| Michael Olise | long_run | 0.3251 | 0.248 | 0.1277 | 0.0 | 0.9689 | 6.0393 | premium_ceiling_components_missing_or_not_explicit: match_winner_goal_ev,hattrick_ev,player_of_the_match_ev,penalty_ev |

## Lav-upside MID/DEF reference

| player_name | strategy | start_security_effect | round_context_component | price_value_component | strategy_score | suspected_issue |
| --- | --- | --- | --- | --- | --- | --- |
| Aurelien Tchouameni | next_round | 0.37 | 2.0763 | 0.1842 | 5.3336 | no_component_issue_from_available_outputs |
| Aurelien Tchouameni | round1_2 | 0.37 | 3.5172 | 0.1842 | 6.3576 | no_component_issue_from_available_outputs |
| Aurelien Tchouameni | group_stage | 0.37 | 4.0238 | 0.1842 | 6.8756 | no_component_issue_from_available_outputs |
| Aurelien Tchouameni | long_run | 0.37 | 0.0 | 0.1842 | 5.6246 | no_component_issue_from_available_outputs |
| Declan Rice | next_round | 0.45 | 1.9411 | 0.2545 | 5.0851 | no_component_issue_from_available_outputs |
| Declan Rice | round1_2 | 0.45 | 3.5636 | 0.2545 | 6.3006 | no_component_issue_from_available_outputs |
| Declan Rice | group_stage | 0.45 | 4.4594 | 0.2545 | 7.5908 | no_component_issue_from_available_outputs |
| Declan Rice | long_run | 0.45 | 0.0 | 0.2545 | 6.0349 | no_component_issue_from_available_outputs |
| Joshua Kimmich | next_round | 0.45 | 3.1419 | -0.0336 | 7.2477 | no_component_issue_from_available_outputs |
| Joshua Kimmich | round1_2 | 0.45 | 4.8429 | -0.0336 | 8.0081 | no_component_issue_from_available_outputs |
| Joshua Kimmich | group_stage | 0.45 | 5.3728 | -0.0336 | 8.6876 | no_component_issue_from_available_outputs |
| Joshua Kimmich | long_run | 0.45 | 0.0 | -0.0336 | 5.8243 | no_component_issue_from_available_outputs |
| Konrad Laimer | next_round | 0.45 | 3.6873 | -0.2944 | 7.0771 | no_component_issue_from_available_outputs |
| Konrad Laimer | round1_2 | 0.45 | 4.1267 | -0.2944 | 6.2763 | no_component_issue_from_available_outputs |
| Konrad Laimer | group_stage | 0.45 | 5.2472 | -0.2944 | 7.8267 | no_component_issue_from_available_outputs |
| Konrad Laimer | long_run | 0.45 | 0.0 | -0.2944 | 0.6488 | no_component_issue_from_available_outputs |
| Manu Koné | next_round | -1.5 | 1.0354 | 0.7448 | 1.8609 | low_upside_player_helped_by_price_value_layer |
| Manu Koné | round1_2 | -1.5 | 1.7388 | 0.7448 | 2.3 | low_upside_player_helped_by_price_value_layer |
| Manu Koné | group_stage | -1.5 | 1.984 | 0.7448 | 2.5101 | low_upside_player_helped_by_price_value_layer |
| Manu Koné | long_run | -1.5 | 0.0 | 0.7448 | 2.8232 | low_upside_player_helped_by_price_value_layer |
| Rodrigo de Paul | next_round | 0.45 | 2.6834 | 0.0134 | 6.4846 | no_component_issue_from_available_outputs |
| Rodrigo de Paul | round1_2 | 0.45 | 4.2268 | 0.0134 | 7.1467 | no_component_issue_from_available_outputs |
| Rodrigo de Paul | group_stage | 0.45 | 5.3492 | 0.0134 | 8.6964 | no_component_issue_from_available_outputs |
| Rodrigo de Paul | long_run | 0.45 | 0.0 | 0.0134 | 6.0473 | no_component_issue_from_available_outputs |
| Scott McTominay | next_round | 0.37 | 3.7653 | 0.0215 | 7.4785 | no_component_issue_from_available_outputs |
| Scott McTominay | round1_2 | 0.37 | 4.86 | 0.0215 | 7.296 | no_component_issue_from_available_outputs |
| Scott McTominay | group_stage | 0.37 | 5.5552 | 0.0215 | 8.3452 | no_component_issue_from_available_outputs |
| Scott McTominay | long_run | 0.37 | 0.0 | 0.0215 | 0.9262 | no_component_issue_from_available_outputs |

## Svar paa price/value-spoergsmaal

- Ja, en billig spiller kan ranke hoejt delvist fordi price/value-laget er positivt, især hvis han samtidig har god start/fixture-kontekst.
- Auditoutputtet viser, at price/value er et stærkt forklaringslag for billige FWDs, men det er ikke alene: round_context og start_security driver også.
- 4-5-1 og 5-4-1 er udsatte, fordi de kun har én FWD-slot; hvis den slot går til value fremfor ceiling, forsvinder meget offensiv upside.
- Kounde og Neuer er fortsat missing EV-source og bruges ikke som kalibreringsbevis.

## Mulige fremtidige modeltests

1. Eksporter/estimer eksplicit offensive ceiling: multi-goal, matchwinner, player-of-match og penalty EV.
2. Cap eller positionsjuster price/value-effekten for FWD, så billig value ikke automatisk slår høj absolut ceiling.
3. Tilføj formation-aware ceiling floor for 4-5-1/5-4-1, fx krav om høj FWD absolut EV eller captain-growth.
4. Test om lav-upside MID/DEF skal have ceiling- eller role-penalty i offensive strategier.
