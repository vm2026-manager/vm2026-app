# Offensive Share Missing Audit

## Omfang

- Spillere uden `goal_share_norm`: 539
- Spillere uden `assist_share_norm`: 539
- Spillere uden `sot_share_norm`: 539
- Spillere uden mindst én offensiv share: 539
- Sandsynlige startere (`start_prob >= 0.70`) uden shares: 86
- MID/FWD med `start_prob >= 0.70` uden shares: 42
- Med `round_context_source = distributed_from_existing_optimizer_ev`: 539
- Disse med base-EV under 1.00: 438

Alle rækker i denne audit har alle tre shares manglende; der er ingen delvist udfyldte share-rækker.

## Fordeling pr. position

| position | players |
| --- | --- |
| DEF | 169 |
| FWD | 114 |
| GK | 93 |
| MID | 163 |

## Lande med flest gaps

| team_id | players |
| --- | --- |
| CIV | 26 |
| CZE | 24 |
| EGY | 17 |
| BRA | 16 |
| TUR | 16 |
| UZB | 16 |
| MEX | 16 |
| CPV | 16 |
| BEL | 14 |
| GHA | 14 |
| SCO | 14 |
| HAI | 14 |
| KOR | 13 |
| USA | 13 |
| QAT | 12 |
| CUW | 12 |
| JOR | 12 |
| KSA | 12 |
| PAN | 12 |
| TUN | 12 |

## Top offensive gaps

Rangeringen bruger startchance, prisrang, nuværende optimizer-EV, position og dokumenteret dødboldsrolle som audit-prioritering. Pris anvendes ikke som direkte EV.

| player_name | team_id | position | price | start_prob | current_weighted_group_stage_ev_before_price_quality | current_optimizer_ev | offensive_role_score | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | BRA | FWD | 6500000 | 0.86 | 0.5445 | 2.0459 | 1.0307 | high |
| Neymar Jr. | BRA | FWD | 5500000 | 0.8919 | 0.4958 | 1.9318 | 1.0142 | high |
| Viktor Gyökeres | SWE | FWD | 6000000 | 0.8141 | 0.3826 | 1.9245 | 1.0022 | high |
| Christian Pulisic | USA | FWD | 4000000 | 0.827 | 0.441 | 1.6431 | 0.9527 | high |
| Yoane Wissa | COD | FWD | 3500000 | 0.8342 | 0.2736 | 1.3733 | 0.9364 | low |
| Edin Dzeko | BIH | FWD | 4500000 | 0.676 | 0.396 | 0.4851 | 0.8891 | high |
| Romelu Lukaku | BEL | FWD | 4500000 | 0.6346 | 0.4372 | 0.5355 | 0.878 | high |
| Memphis Depay | NED | FWD | 4000000 | 0.6669 | 0.4181 | 0.5122 | 0.8756 | high |
| Federico Valverde | URU | MID | 5500000 | 0.8894 | 0.4886 | 1.5913 | 0.8547 | high |
| Patrik Schick | CZE | FWD | 4500000 | 0.8543 | 0.2142 | 1.6347 | 0.8522 | medium |
| Darwin Nunez | URU | FWD | 6000000 | 0.3943 | 0.3845 | 0.471 | 0.8389 | high |
| Yan Diomande | CIV | FWD | 3000000 | 0.9123 | 0.2175 | 1.0774 | 0.8207 | medium |
| Ryan Mendes Da Graça | CPV | FWD | 3000000 | 0.8829 | 0.2218 | 1.0798 | 0.8119 | high |
| Josue Casimir | HAI | FWD | 2500000 | 0.8939 | 0.2695 | 0.7344 | 0.7948 | high |
| Amine Gouiri | ALG | FWD | 3000000 | 0.8025 | 0.3088 | 1.1277 | 0.789 | low |
| Almoez Ali | QAT | FWD | 3000000 | 0.4561 | 0.2309 | 0.2828 | 0.7833 | high |
| Deni Juric | AUS | FWD | 2500000 | 0.8537 | 0.3134 | 0.7585 | 0.7833 | high |
| Aymen Hussein | IRQ | FWD | 3500000 | 0.3986 | 0.1923 | 0.2355 | 0.7764 | high |
| Mohamed Toure | AUS | FWD | 3000000 | 0.752 | 0.2576 | 1.0995 | 0.7731 | high |
| Mousa Tamari | JOR | MID | 3000000 | 0.8437 | 0.274 | 1.0783 | 0.7701 | high |
| Prince Adu | GHA | FWD | 2000000 | 0.8702 | 0.413 | 0.4707 | 0.7693 | high |
| Hakan Calhanoglu | TUR | MID | 4000000 | 0.7024 | 0.4235 | 1.4427 | 0.7602 | high |
| Jovo Lukic | BIH | FWD | 2000000 | 0.7493 | 0.3036 | 0.4105 | 0.7315 | high |
| Ermedin Demirovic | BIH | FWD | 3000000 | 0.6712 | 0.3449 | 0.4225 | 0.7315 | high |
| Maxi Araujo | URU | MID | 5000000 | 0.9094 | 0.5028 | 1.5756 | 0.7288 | high |
| Cyle Christopher Larin | CAN | FWD | 3000000 | 0.6238 | 0.4251 | 0.5208 | 0.7198 | high |
| Luis Suarez Charris | COL | FWD | 4500000 | 0.5148 | 0.3212 | 0.3935 | 0.7184 | high |
| Timothy Weah | USA | FWD | 3000000 | 0.6148 | 0.412 | 0.5047 | 0.7167 | high |
| Wilson Isidor | HAI | FWD | 3000000 | 0.6294 | 0.2564 | 0.3141 | 0.7161 | high |
| Juan 'Cucho' Hernandez | COL | FWD | 4000000 | 0.5405 | 0.3683 | 0.4512 | 0.7161 | high |
| Nico Gonzalez | ARG | FWD | 4000000 | 0.5303 | 0.3619 | 0.4434 | 0.7128 | high |
| Rayan | BRA | FWD | 3000000 | 0.5901 | 0.3537 | 0.4333 | 0.7074 | high |
| Jean-Ricner Bellegarde | HAI | MID | 3000000 | 0.6978 | 0.2733 | 0.3347 | 0.7072 | high |
| Bruno Guimaraes | BRA | MID | 4500000 | 0.8763 | 0.4846 | 1.5296 | 0.7061 | high |
| Elye Wahi | CIV | FWD | 2500000 | 0.6294 | 0.2131 | 0.261 | 0.7032 | medium |

## Sanity-listen

| requested_name | player_name | team_id | position | start_prob | status | current_base_ev |
| --- | --- | --- | --- | --- | --- | --- |
| Raphinha | Raphinha | BRA | FWD | 0.86 | missing_shares_estimated | 0.5445 |
| Mahmoud Trezeguet | Mahmoud Trezeguet | EGY | MID | 0.92 | missing_shares_estimated | 0.4184 |
| Neymar Jr. | Neymar Jr. | BRA | FWD | 0.8919 | missing_shares_estimated | 0.4958 |
| Kenan Yildiz | Kenan Yildiz | TUR | MID | 0.8953 | missing_shares_estimated | 0.4446 |
| Christian Pulisic | Christian Pulisic | USA | FWD | 0.827 | missing_shares_estimated | 0.441 |
| Viktor Gyökeres | Viktor Gyökeres | SWE | FWD | 0.8141 | missing_shares_estimated | 0.3826 |
| Patrik Schick | Patrik Schick | CZE | FWD | 0.8543 | missing_shares_estimated | 0.2142 |
| Hakan Calhanoglu | Hakan Calhanoglu | TUR | MID | 0.7024 | missing_shares_estimated | 0.4235 |
| Salem Al-Dawsari | Salem Al-Dawsari | KSA | MID | 0.9032 | missing_shares_estimated | 0.3497 |
| Federico Valverde | Federico Valverde | URU | MID | 0.8894 | missing_shares_estimated | 0.4886 |
| Bruno Guimaraes | Bruno Guimaraes | BRA | MID | 0.8763 | missing_shares_estimated | 0.4846 |
| Romelu Lukaku | Romelu Lukaku | BEL | FWD | 0.6346 | missing_shares_estimated | 0.4372 |
| Tomas Soucek | Tomas Soucek | CZE | MID | 0.9395 | missing_shares_estimated | 0.2498 |
| Antonio Nusa | Antonio Nusa | NOR | MID | 0.82 | shares_present_no_fallback_needed | 3.6978 |
| Brian Gutierrez | Brian Gutierrez | MEX | MID | 0.6683 | missing_shares_estimated | 0.4705 |
