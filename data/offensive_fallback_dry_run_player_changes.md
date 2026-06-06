# Offensive Fallback Production Dry-Run

## Hybridregel

`candidate = min(variant_a, variant_b * 1.20, current_base + position_price_cap)`

`raw_hybrid = max(current_base, candidate)`

Hoved-dry-run anvender kun ændringen, når løftet er mindst 0.25 EV.

Caps:

- FWD >= 6,0 mio.: +2,50
- FWD 4,0-5,9 mio.: +1,75
- FWD < 4,0 mio.: +1,25
- MID >= 5,0 mio.: +1,75
- MID 3,0-4,9 mio.: +1,25
- MID < 3,0 mio.: +0,75

Price-quality genberegnes på en hukommelseskopi med produktionens eksisterende 55/45-, likely-starter- og reservebeskyttelsesfunktion.

## Omfang

- Fallback-kandidater: 42
- Spillere med anvendt base-EV-løft >= 0,25: 37
- Optimizer-EV-løft > 0,25: 34
- Optimizer-EV-løft > 1,00: 1

## Top 30 løft

| player_name | team_id | position | price | start_prob | current_base_ev | hybrid_base_ev | base_ev_lift | current_optimizer_ev | dry_run_optimizer_ev | optimizer_ev_lift | cap_reason | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | BRA | FWD | 6500000 | 0.86 | 0.5445 | 3.0445 | 2.5 | 2.0459 | 3.4199 | 1.3741 | fwd_price_ge_6m_lift_cap_2.50 | high |
| Maxi Araujo | URU | MID | 5000000 | 0.9094 | 0.5028 | 2.2528 | 1.75 | 1.5756 | 2.539 | 0.9634 | mid_price_ge_5m_lift_cap_1.75 | high |
| Christian Pulisic | USA | FWD | 4000000 | 0.827 | 0.441 | 2.191 | 1.75 | 1.6431 | 2.6057 | 0.9626 | fwd_price_4_to_5_9m_lift_cap_1.75 | high |
| Patrik Schick | CZE | FWD | 4500000 | 0.8543 | 0.2142 | 1.9642 | 1.75 | 1.6347 | 2.597 | 0.9623 | fwd_price_4_to_5_9m_lift_cap_1.75 | medium |
| Neymar Jr. | BRA | FWD | 5500000 | 0.8919 | 0.4958 | 2.2458 | 1.75 | 1.9318 | 2.8936 | 0.9618 | fwd_price_4_to_5_9m_lift_cap_1.75 | high |
| Federico Valverde | URU | MID | 5500000 | 0.8894 | 0.4886 | 2.178 | 1.6893 | 1.5913 | 2.5214 | 0.93 | variant_b_times_1_20 | high |
| Prince Adu | GHA | FWD | 2000000 | 0.8702 | 0.413 | 1.663 | 1.25 | 0.4707 | 1.1619 | 0.6912 | fwd_price_lt_4m_lift_cap_1.25 | high |
| Yan Diomande | CIV | FWD | 3000000 | 0.9123 | 0.2175 | 1.4675 | 1.25 | 1.0774 | 1.7664 | 0.689 | fwd_price_lt_4m_lift_cap_1.25 | medium |
| Mohamed Toure | AUS | FWD | 3000000 | 0.752 | 0.2576 | 1.5076 | 1.25 | 1.0995 | 1.7885 | 0.689 | fwd_price_lt_4m_lift_cap_1.25 | high |
| Mahmoud Trezeguet | EGY | MID | 3000000 | 0.92 | 0.4184 | 1.6684 | 1.25 | 1.1577 | 1.8463 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Marwan Ateya | EGY | MID | 3000000 | 0.8808 | 0.3617 | 1.6117 | 1.25 | 1.1265 | 1.8151 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Ahmed Zizo | EGY | MID | 3000000 | 0.7866 | 0.3077 | 1.5577 | 1.25 | 1.0968 | 1.7854 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Lukas Provod | CZE | MID | 3000000 | 0.8588 | 0.2516 | 1.5016 | 1.25 | 1.066 | 1.7546 | 0.6886 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Tomas Soucek | CZE | MID | 3500000 | 0.9395 | 0.2498 | 1.4998 | 1.25 | 1.264 | 1.9525 | 0.6885 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Pavel Sulc | CZE | MID | 4000000 | 0.8997 | 0.2332 | 1.4832 | 1.25 | 1.338 | 2.0264 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | medium |
| Fabian Ruiz | ESP | MID | 4500000 | 0.8178 | 1.0453 | 2.2953 | 1.25 | 1.8379 | 2.5264 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Bruno Guimaraes | BRA | MID | 4500000 | 0.8763 | 0.4846 | 1.7346 | 1.25 | 1.5296 | 2.218 | 0.6884 | mid_price_3_to_4_9m_lift_cap_1.25 | high |
| Ryan Mendes Da Graça | CPV | FWD | 3000000 | 0.8829 | 0.2218 | 1.4235 | 1.2018 | 1.0798 | 1.7423 | 0.6625 | variant_b_times_1_20 | high |
| Salem Al-Dawsari | KSA | MID | 3000000 | 0.9032 | 0.3497 | 1.4856 | 1.1359 | 1.1199 | 1.7458 | 0.6258 | variant_a | medium |
| Kenan Yildiz | TUR | MID | 4000000 | 0.8953 | 0.4446 | 1.5622 | 1.1176 | 1.4542 | 2.0699 | 0.6156 | variant_b_times_1_20 | high |
| Viktor Gyökeres | SWE | FWD | 6000000 | 0.8141 | 0.3826 | 1.5014 | 1.1188 | 1.9245 | 2.539 | 0.6145 | variant_b_times_1_20 | high |
| Mohamed Kanno | KSA | MID | 3000000 | 0.8439 | 0.3174 | 1.4131 | 1.0957 | 1.1022 | 1.7059 | 0.6037 | variant_a | medium |
| Mousa Tamari | JOR | MID | 3000000 | 0.8437 | 0.274 | 1.1571 | 0.8831 | 1.0783 | 1.5651 | 0.4868 | variant_a | high |
| Hakan Calhanoglu | TUR | MID | 4000000 | 0.7024 | 0.4235 | 1.2882 | 0.8647 | 1.4427 | 1.9192 | 0.4765 | variant_b_times_1_20 | high |
| Deni Juric | AUS | FWD | 2500000 | 0.8537 | 0.3134 | 1.1462 | 0.8329 | 0.7585 | 1.2192 | 0.4607 | variant_b_times_1_20 | high |
| Ben Doak | SCO | MID | 3000000 | 0.8408 | 0.4795 | 1.2405 | 0.761 | 1.1913 | 1.611 | 0.4196 | variant_b_times_1_20 | high |
| Yousef Qashi | JOR | MID | 2000000 | 0.8559 | 0.28 | 1.03 | 0.75 | 0.3762 | 0.7902 | 0.414 | mid_price_lt_3m_lift_cap_0.75 | high |
| Dong-gyeong Lee | KOR | MID | 2000000 | 0.7198 | 0.2742 | 1.0242 | 0.75 | 0.373 | 0.787 | 0.414 | mid_price_lt_3m_lift_cap_0.75 | high |
| Alvaro Fidalgo | MEX | MID | 2500000 | 0.8834 | 0.5062 | 1.2562 | 0.75 | 0.8314 | 1.2452 | 0.4138 | mid_price_lt_3m_lift_cap_0.75 | high |
| Saleh Al-Shamat | KSA | MID | 2500000 | 0.7644 | 0.2741 | 1.0241 | 0.75 | 0.7037 | 1.1175 | 0.4138 | mid_price_lt_3m_lift_cap_0.75 | medium |

## Sanity

| player_name | fallback_candidate | fallback_applied | current_base_ev | variant_a_base_ev | variant_b_base_ev | hybrid_base_ev | current_optimizer_ev | dry_run_optimizer_ev | cap_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | yes | yes | 0.5445 | 4.501 | 3.0938 | 3.0445 | 2.0459 | 3.4199 | fwd_price_ge_6m_lift_cap_2.50 |
| Christian Pulisic | yes | yes | 0.441 | 3.2732 | 2.0521 | 2.191 | 1.6431 | 2.6057 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Patrik Schick | yes | yes | 0.2142 | 3.9864 | 3.6474 | 1.9642 | 1.6347 | 2.597 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Neymar Jr. | yes | yes | 0.4958 | 4.5747 | 3.1399 | 2.2458 | 1.9318 | 2.8936 | fwd_price_4_to_5_9m_lift_cap_1.75 |
| Federico Valverde | yes | yes | 0.4886 | 2.7788 | 1.815 | 2.178 | 1.5913 | 2.5214 | variant_b_times_1_20 |
| Mahmoud Trezeguet | yes | yes | 0.4184 | 2.2218 | 1.6259 | 1.6684 | 1.1577 | 1.8463 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Tomas Soucek | yes | yes | 0.2498 | 3.1269 | 2.8487 | 1.4998 | 1.264 | 1.9525 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Bruno Guimaraes | yes | yes | 0.4846 | 3.542 | 2.4768 | 1.7346 | 1.5296 | 2.218 | mid_price_3_to_4_9m_lift_cap_1.25 |
| Salem Al-Dawsari | yes | yes | 0.3497 | 1.4856 | 1.605 | 1.4856 | 1.1199 | 1.7458 | variant_a |
| Kenan Yildiz | yes | yes | 0.4446 | 2.7102 | 1.3018 | 1.5622 | 1.4542 | 2.0699 | variant_b_times_1_20 |
| Viktor Gyökeres | yes | yes | 0.3826 | 2.4279 | 1.2512 | 1.5014 | 1.9245 | 2.539 | variant_b_times_1_20 |
| Hakan Calhanoglu | yes | yes | 0.4235 | 2.1565 | 1.0735 | 1.2882 | 1.4427 | 1.9192 | variant_b_times_1_20 |
| Antonio Nusa | no | no | 3.6978 |  |  | 3.6978 | 3.1605 | 3.1614 | already_has_offensive_shares_not_candidate |
| Romelu Lukaku | no | no | 0.4372 |  |  | 0.4372 | 0.5355 | 0.5355 | not_eligible_position_or_start_prob |
| Brian Gutierrez | no | no | 0.4705 |  |  | 0.4705 | 0.5763 | 0.5763 | not_eligible_position_or_start_prob |

## Vurdering

- Hybridreglen forhindrer Variant A's rå estimater i at slå fuldt igennem og sænker aldrig en spiller.
- Spillere, der ikke er fallback-kandidater, kan få meget små final-EV-bevægelser, fordi price-quality-positionernes kvantiler genberegnes globalt. Det er forventet produktionsadfærd.
- Dry-run-resultatet er konservativt nok til en afgrænset, auditeret produktionstest, men ikke til en ukontrolleret fuld aktivering uden efterfølgende sanity- og optimizer-audit.
