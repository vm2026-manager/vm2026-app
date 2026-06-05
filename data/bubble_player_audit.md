# Bubble Player Audit

Ren audit baseret paa eksisterende data. Audit-scriptet aendrer ikke optimizer, EV eller spillerpool.

## Summary

- Audit rows: 49
- Matched players: 49
- Model error flags: 17
- appearance_prob uses the explicit player_pool column when available; otherwise it falls back to availability_prob as a proxy.

## 1. Biggest likely model errors

| player_name | team_id | position | model_error_reason | recommended_manual_action |
| --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | MID | usage_start_high_but_model_start_low_usage=0.90; ev_and_fixture_values_missing_in_model_data | Keep avoid override. Check start-probability merge/fallback layer. Review player EV source and fixture mapping before strategy use. |
| Michael Olise | FRA | FWD | premium_offensive_low_next_round_score_in_good_fixture | Review premium attacker goal/upside weighting. |
| Jamal Musiala | GER | MID | premium_offensive_low_next_round_score_in_good_fixture; specific_round_1_but_low_next_round_score | Review premium attacker goal/upside weighting. Check round 1 fixture/upside assumptions. |
| Antonio Nusa | NOR | MID | requested_uncertain_start_check | Manual lineup/start check before selection. |
| Scott McTominay | SCO | MID | central_or_defensive_mid_may_be_overvalued | Review role/upside weighting for central MID. |
| Mahmoud Trezeguet | EGY | MID | specific_round_2_but_low_round_2_ev | Check round 2 fixture/upside assumptions. |
| Jurrien Timber | NED | DEF | requested_uncertain_start_check; specific_round_2_but_low_round_2_ev; specific_round_3_but_low_round_3_ev | Manual lineup/start check before selection. Check round 2 fixture/upside assumptions. Check round 3 fixture/upside/rotation assumptions. |
| Wesley Franca | BRA | DEF | specific_round_2_but_low_round_2_ev; specific_round_3_but_low_round_3_ev | Check round 2 fixture/upside assumptions. Check round 3 fixture/upside/rotation assumptions. |
| Ismael Saibari | MAR | MID | requested_uncertain_start_check | Manual lineup/start check before selection. |
| Raphinha | BRA | FWD | specific_round_2_but_low_round_2_ev; specific_round_3_but_low_round_3_ev | Check round 2 fixture/upside assumptions. Check round 3 fixture/upside/rotation assumptions. |

## 2. Players that look undervalued

| player_name | team_id | next_round_score | round_1_ev | model_error_reason |
| --- | --- | --- | --- | --- |
| Michael Olise | FRA | 5.5722 | 0.8907 | premium_offensive_low_next_round_score_in_good_fixture |
| Jamal Musiala | GER | 5.0319 | 1.0844 | premium_offensive_low_next_round_score_in_good_fixture; specific_round_1_but_low_next_round_score |

## 3. Players that look overvalued

| player_name | team_id | price | next_round_score | model_error_reason |
| --- | --- | --- | --- | --- |
| Scott McTominay | SCO | 4500000 | 7.4761 | central_or_defensive_mid_may_be_overvalued |

## 4. Manual start/availability checks

| player_name | team_id | start_prob | conditional_start_prob | availability_risk | model_error_reason |
| --- | --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | 0.0 | 0.0 | out | usage_start_high_but_model_start_low_usage=0.90; ev_and_fixture_values_missing_in_model_data |
| Antonio Nusa | NOR | 0.6924 | 0.7301 | low_risk | requested_uncertain_start_check |
| Jurrien Timber | NED | 0.6333 | 0.7364 | high_risk | requested_uncertain_start_check; specific_round_2_but_low_round_2_ev; specific_round_3_but_low_round_3_ev |
| Ismael Saibari | MAR | 0.6357 | 0.7381 | high_risk | requested_uncertain_start_check |
| Ismaila Sarr | SEN | 0.6015 | 0.6526 | medium_risk | requested_uncertain_start_check; specific_round_3_but_low_round_3_ev |
| Patrick Wimmer | AUT | 0.5787 | 0.6477 | medium_risk | requested_uncertain_start_check |
| Deniz Undav | GER | 0.3483 | 0.405 | high_risk | requested_uncertain_start_check; specific_round_1_but_low_next_round_score |
| Andreas Schjelderup | NOR | 0.3333 | 0.4545 | medium_risk | requested_uncertain_start_check |

## 5. Missing player/position/price/EV data

| player_name | team_id | position | price | model_error_reason |
| --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | MID | 3500000 | usage_start_high_but_model_start_low_usage=0.90; ev_and_fixture_values_missing_in_model_data |

## 6. Recommended model tracks to fix first

1. Fix player/team/fixture mappings that produce zero EV or placeholder teams.
2. Split true start probability, appearance probability, and availability probability explicitly.
3. Review premium attacker next-round scoring against low-upside MID/DEF upgrades.
4. Review central MID role/upside weighting so safe starters do not dominate purely on security/value.
5. Keep manual avoid controls active; Baumgartner is a control row and remains avoid.
