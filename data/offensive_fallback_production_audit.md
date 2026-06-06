# Offensive Fallback Production Audit

- Commit-status: **committed_all_preflight_checks_passed**
- Kandidater: 42
- Faktisk ændrede spillere: 37
- Spillere med eksisterende shares kontrolleret/udeladt: 705
- Spillere med eksisterende shares fejlagtigt ændret: 0
- Base-EV-løft >= 0,25: 37
- Optimizer-EV-løft > 0,25: 34
- Optimizer-EV-løft > 1,00: 1
- Maksimal kampfordelings-sumfejl: 4.4408920985e-16
- Price-quality-formelmismatches: 0
- Maksimal price-quality-formeldifference: 7.00000000187e-07
- Pool/EV-finalmismatches efter simuleret/udført sync: 0

## Stopkriterier

| Kriterium | Faktisk | Grænse | Bestået |
| --- | ---: | ---: | --- |
| fallback_players_le_45 | 37 | 45 | ja |
| optimizer_lifts_over_1_le_5 | 1 | 5 | ja |
| fallback_per_strategy_le_2 | 1 | 2 | ja |
| price_quality_formula_mismatches_eq_0 | 0 | 0 | ja |
| prospective_pool_ev_mismatches_eq_0 | 0 | 0 | ja |
| match_sum_error_le_1e_6 | 4.440892098500626e-16 | 1e-06 | ja |
| lowered_players_eq_0 | 0 | 0 | ja |
| existing_share_players_applied_eq_0 | 0 | 0 | ja |
| duplicate_ev_ids_eq_0 | 0 | 0 | ja |
| duplicate_pool_ids_eq_0 | 0 | 0 | ja |

## Backups

- EV: `data\player_ev_group_stage_v1.backup_before_offensive_fallback_20260606_093220.csv`
- player_pool: `data\player_pool_v1.backup_before_offensive_fallback_20260606_093220.json`

## Top 30 løft

| player_name | team_id | position | start_prob | current_base_ev | hybrid_base_ev | base_ev_lift | current_optimizer_ev | dry_run_optimizer_ev | optimizer_ev_lift | cap_reason | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | BRA | FWD | 0.86 | 0.5445 | 3.0445 | 2.5 | 2.0459 | 3.4199 | 1.3741 | fwd_price_ge_6m_lift_cap_2.50 | high |
| Maxi Araujo | URU | MID | 0.9094 | 0.5028 | 2.2528 | 1.75 | 1.5756 | 2.539 | 0.9634 | mid_price_ge_5m_lift_cap_1.75 | high |
| Christian Pulisic | USA | FWD | 0.827 | 0.441 | 2.191 | 1.75 | 1.6431 | 2.6057 | 0.9626 | fwd_price_4_to_5_9m_lift_cap_1.75 | high |
| Patrik Schick | CZE | FWD | 0.8543 | 0.2142 | 1.9642 | 1.75 | 1.6347 | 2.597 | 0.9623 | fwd_price_4_to_5_9m_lift_cap_1.75 | medium |
| Neymar Jr. | BRA | FWD | 0.8919 | 0.4958 | 2.2458 | 1.75 | 1.9318 | 2.8936 | 0.9618 | fwd_price_4_to_5_9m_lift_cap_1.75 | high |
| Federico Valverde | URU | MID | 0.8894 | 0.4886 | 2.178 | 1.6893 | 1.5913 | 2.5214 | 0.93 | variant_b_times_1_20 | high |
| Prince Adu | GHA | FWD | 0.8702 | 0.413 | 1.663 | 1.25 | 0.4707 | 1.1619 | 0.6912 | fwd_price_lt_4m_lift_cap_1.25 | high |
| Yan Diomande | CIV | FWD | 0.9123 | 0.2175 | 1.4675 | 1.25 | 1.0774 | 1.7664 | 0.689 | fwd_price_lt_4m_lift_cap_1.25 | medium |
| Mohamed Toure | AUS | FWD | 0.752 | 0.2576 | 1.5076 | 1.25 | 1.0995 | 1.7885 | 0.689 | fwd_price_lt_4m_lift_cap_1.25 | high |
| Mahmoud Trezeguet | EGY | MID | 0.92 | 0.4184 | 1.6684 | 1.25 | 1.1577 | 1.8463 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Ahmed Zizo | EGY | MID | 0.7866 | 0.3077 | 1.5577 | 1.25 | 1.0968 | 1.7854 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Marwan Ateya | EGY | MID | 0.8808 | 0.3617 | 1.6117 | 1.25 | 1.1265 | 1.8151 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Lukas Provod | CZE | MID | 0.8588 | 0.2516 | 1.5016 | 1.25 | 1.066 | 1.7546 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Tomas Soucek | CZE | MID | 0.9395 | 0.2498 | 1.4998 | 1.25 | 1.264 | 1.9525 | 0.6885 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Pavel Sulc | CZE | MID | 0.8997 | 0.2332 | 1.4832 | 1.25 | 1.338 | 2.0264 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Fabian Ruiz | ESP | MID | 0.8178 | 1.0453 | 2.2953 | 1.25 | 1.8379 | 2.5264 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Bruno Guimaraes | BRA | MID | 0.8763 | 0.4846 | 1.7346 | 1.25 | 1.5296 | 2.218 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Ryan Mendes Da Graça | CPV | FWD | 0.8829 | 0.2218 | 1.4235 | 1.2018 | 1.0798 | 1.7423 | 0.6625 | variant_b_times_1_20 | high |
| Salem Al-Dawsari | KSA | MID | 0.9032 | 0.3497 | 1.4856 | 1.1359 | 1.1199 | 1.7458 | 0.6258 | variant_a | medium |
| Kenan Yildiz | TUR | MID | 0.8953 | 0.4446 | 1.5622 | 1.1176 | 1.4542 | 2.0699 | 0.6156 | variant_b_times_1_20 | high |
| Viktor Gyökeres | SWE | FWD | 0.8141 | 0.3826 | 1.5014 | 1.1188 | 1.9245 | 2.539 | 0.6145 | variant_b_times_1_20 | high |
| Mohamed Kanno | KSA | MID | 0.8439 | 0.3174 | 1.4131 | 1.0957 | 1.1022 | 1.7059 | 0.6037 | variant_a | medium |
| Mousa Tamari | JOR | MID | 0.8437 | 0.274 | 1.1571 | 0.8831 | 1.0783 | 1.5651 | 0.4868 | variant_a | high |
| Hakan Calhanoglu | TUR | MID | 0.7024 | 0.4235 | 1.2882 | 0.8647 | 1.4427 | 1.9192 | 0.4765 | variant_b_times_1_20 | high |
| Deni Juric | AUS | FWD | 0.8537 | 0.3134 | 1.1462 | 0.8329 | 0.7585 | 1.2192 | 0.4607 | variant_b_times_1_20 | high |
| Ben Doak | SCO | MID | 0.8408 | 0.4795 | 1.2405 | 0.761 | 1.1913 | 1.611 | 0.4196 | variant_b_times_1_20 | high |
| Yousef Qashi | JOR | MID | 0.8559 | 0.28 | 1.03 | 0.75 | 0.3762 | 0.7902 | 0.414 | mid_price_lt_3m_lift_cap_0.75 | high |
| Dong-gyeong Lee | KOR | MID | 0.7198 | 0.2742 | 1.0242 | 0.75 | 0.373 | 0.787 | 0.414 | mid_price_lt_3m_lift_cap_0.75 | high |
| Alvaro Fidalgo | MEX | MID | 0.8834 | 0.5062 | 1.2562 | 0.75 | 0.8314 | 1.2452 | 0.4138 | mid_price_lt_3m_lift_cap_0.75 | high |
| Odildzhon Khamrobekov | UZB | MID | 0.939 | 0.3005 | 1.0505 | 0.75 | 0.7182 | 1.132 | 0.4138 | mid_price_lt_3m_lift_cap_0.75 | medium |

## Sanity

| player_name | shares_status | fallback_candidate | fallback_applied | current_base_ev | hybrid_base_ev | current_optimizer_ev | dry_run_optimizer_ev | cap_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | missing_shares | yes | yes | 0.5445 | 3.0445 | 2.0459 | 3.4199 | fwd_price_ge_6m_lift_cap_2.50 |
| Christian Pulisic | missing_shares | yes | yes | 0.441 | 2.191 | 1.6431 | 2.6057 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Patrik Schick | missing_shares | yes | yes | 0.2142 | 1.9642 | 1.6347 | 2.597 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Neymar Jr. | missing_shares | yes | yes | 0.4958 | 2.2458 | 1.9318 | 2.8936 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Federico Valverde | missing_shares | yes | yes | 0.4886 | 2.178 | 1.5913 | 2.5214 | variant_b_times_1_20 |
| Mahmoud Trezeguet | missing_shares | yes | yes | 0.4184 | 1.6684 | 1.1577 | 1.8463 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Tomas Soucek | missing_shares | yes | yes | 0.2498 | 1.4998 | 1.264 | 1.9525 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Bruno Guimaraes | missing_shares | yes | yes | 0.4846 | 1.7346 | 1.5296 | 2.218 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Salem Al-Dawsari | missing_shares | yes | yes | 0.3497 | 1.4856 | 1.1199 | 1.7458 | variant_a |
| Kenan Yildiz | missing_shares | yes | yes | 0.4446 | 1.5622 | 1.4542 | 2.0699 | variant_b_times_1_20 |
| Viktor Gyökeres | missing_shares | yes | yes | 0.3826 | 1.5014 | 1.9245 | 2.539 | variant_b_times_1_20 |
| Hakan Calhanoglu | missing_shares | yes | yes | 0.4235 | 1.2882 | 1.4427 | 1.9192 | variant_b_times_1_20 |
| Antonio Nusa | already_shares_excluded | no | no | 3.6978 | 3.6978 | 3.1605 | 3.1614 | already_has_offensive_shares_not_candidate |
| Brian Gutierrez | missing_shares | no | no | 0.4705 | 0.4705 | 0.5763 | 0.5763 | not_eligible_position_or_start_prob |
| Romelu Lukaku | missing_shares | no | no | 0.4372 | 0.4372 | 0.5355 | 0.5355 | not_eligible_position_or_start_prob |

## Optimizer før/efter

| strategy | formation_before | formation_after | price_before | price_after | ev_before | ev_after | score_before | score_after | players_in | players_out | fallback_player_count | fallback_players | high_risk_before | high_risk_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_round | 3-4-3 | 3-4-3 | 49500000 | 49500000 | 43.3972 | 43.3996 | 94.3646 | 94.3665 |  |  | 0 |  | 0 | 0 |
| round1_2 | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 44.5742 | 44.5749 | 113.1555 | 113.156 |  |  | 0 |  | 0 | 0 |
| group_stage | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 44.8837 | 44.8845 | 127.2461 | 127.2466 |  |  | 0 |  | 0 | 0 |
| long_run | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 38.8461 | 38.5008 | 70.9912 | 71.0763 | Fabian Ruiz | Declan Rice | 1 | Fabian Ruiz | 0 | 0 |
