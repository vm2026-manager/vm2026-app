# Offensive Share Fallback Experiment

## Metode

- Referencepopulation: 705 spillere med alle tre shares og kampkomponenter.
- Variant A: positionsspecifik nearest-reference-model med prisrang inden for position, startchance, minutproxy og holdets offensive fixturemiljø.
- Variant B: team-residualmodel. Variant A bruges kun som fordelingsvægt; kendte shares bevares, residualen fordeles op til et empirisk teammål og et hårdt loft på 0.90.
- Variant B har desuden positionsbaserede individuelle caps, så én spiller ikke absorberer en stor rest-share alene.
- Kampkomponenter beregnes fra robuste teamrater med positionsfallback og de eksisterende Holdet.dk-pointregler.
- Base-EV består af en reference-estimeret ikke-offensiv baseline plus de estimerede offensive komponenter.
- Auditens optimizer-estimat bruger den eksisterende formel `0.55 * base_ev + 0.45 * price_quality_ev`; produktionsfelter ændres ikke.
- Transfermarkt-usage indeholder caps/startbrug, men ikke en stabil kamp-for-kamp goal/assist/SOT-rate for hele populationen. Derfor er der ikke bygget en variant C.

## Leave-One-Out-validering

Hver referencespillers shares skjules, spilleren fjernes fra sit eget referencegrundlag, og fallbacken sammenlignes med de faktiske shares, kampkomponenter og base-EV.

### Samlet base-EV-fejl

Variant A:

| players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 705 | 0.2305 | 0.3388 | 0.1867 | 0.0427 | 256 | 95 | 37 |

Variant B:

| players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 705 | 0.4253 | 0.6321 | 0.3309 | 0.357 | 425 | 234 | 103 |

### Share-fejl

Variant A:

| component | median_absolute_error | mean_absolute_error | median_relative_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- |
| goal | 0.0057 | 0.0125 | 0.2576 | 349 | 158 | 42 |
| assist | 0.006 | 0.0097 | 0.2132 | 268 | 98 | 20 |
| sot | 0.0059 | 0.0154 | 0.3849 | 415 | 255 | 91 |

Variant B:

| component | median_absolute_error | mean_absolute_error | median_relative_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- |
| goal | 0.0161 | 0.0266 | 0.9844 | 573 | 492 | 308 |
| assist | 0.0228 | 0.0238 | 0.7586 | 511 | 424 | 219 |
| sot | 0.014 | 0.0286 | 1.1547 | 559 | 482 | 344 |

### Kampkomponent-fejl

Variant A:

| component | median_match_component_absolute_error | mean_match_component_absolute_error |
| --- | --- | --- |
| goal | 0.0043 | 0.0135 |
| assist | 0.0034 | 0.0071 |
| sot | 0.0044 | 0.0134 |

Variant B:

| component | median_match_component_absolute_error | mean_match_component_absolute_error |
| --- | --- | --- |
| goal | 0.0154 | 0.0311 |
| assist | 0.0121 | 0.0169 |
| sot | 0.0148 | 0.0259 |

### Base-EV-fejl pr. position

Variant A:

| position | players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEF | 239 | 0.2157 | 0.272 | 0.1503 | 0.022 | 65 | 23 | 11 |
| FWD | 160 | 0.2275 | 0.4023 | 0.1647 | 0.0284 | 51 | 21 | 9 |
| GK | 59 | 0.2243 | 0.2908 | 0.2106 | 0.1795 | 24 | 15 | 5 |
| MID | 247 | 0.2425 | 0.3739 | 0.2359 | 0.0415 | 116 | 36 | 12 |

Variant B:

| position | players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEF | 239 | 0.341 | 0.3952 | 0.2266 | 0.2947 | 113 | 38 | 12 |
| FWD | 160 | 0.5542 | 0.854 | 0.3868 | 0.4948 | 103 | 61 | 22 |
| GK | 59 | 0.2878 | 0.3378 | 0.2643 | 0.2634 | 34 | 18 | 7 |
| MID | 247 | 0.548 | 0.7877 | 0.4673 | 0.4408 | 175 | 117 | 62 |

### Base-EV-fejl pr. prisniveau

Variant A:

| price_band | players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget | 76 | 0.1379 | 0.1776 | 0.2294 | 0.0268 | 34 | 20 | 11 |
| lower_mid | 217 | 0.1684 | 0.2726 | 0.2112 | 0.0136 | 89 | 32 | 16 |
| premium | 212 | 0.3201 | 0.474 | 0.1383 | 0.0602 | 65 | 23 | 6 |
| upper_mid | 200 | 0.2468 | 0.3287 | 0.1798 | 0.1018 | 68 | 20 | 4 |

Variant B:

| price_band | players | median_absolute_error | mean_absolute_error | median_relative_error | median_signed_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget | 76 | 0.1629 | 0.2357 | 0.3493 | 0.1266 | 44 | 27 | 16 |
| lower_mid | 217 | 0.2667 | 0.3908 | 0.3066 | 0.2288 | 120 | 67 | 30 |
| premium | 212 | 0.7806 | 1.0813 | 0.3768 | 0.7462 | 136 | 82 | 37 |
| upper_mid | 200 | 0.4477 | 0.5682 | 0.3121 | 0.3775 | 125 | 58 | 20 |

## Missing-share-estimater

Sikkerhed:

| confidence | players |
| --- | --- |
| high | 382 |
| low | 79 |
| medium | 78 |

Hyppigste advarsler:

| warning_flag | players |
| --- | --- |
| legacy_aggregate_round_context | 539 |
| variant_a_goal_team_share_over_cap | 341 |
| variant_a_assist_team_share_over_cap | 337 |
| variant_a_sot_team_share_over_cap | 283 |
| goal_no_team_residual | 53 |
| match_1_goal_position_rate | 50 |
| match_2_goal_position_rate | 50 |
| match_3_goal_position_rate | 50 |
| assist_no_team_residual | 47 |
| sot_no_team_residual | 47 |
| match_1_assist_position_rate | 44 |
| match_1_sot_position_rate | 44 |
| match_2_assist_position_rate | 44 |
| match_2_sot_position_rate | 44 |
| match_3_assist_position_rate | 44 |

## Sanity

| requested_name | player_name | team_id | position | start_prob | status | current_base_ev | fallback_base_ev_variant_a | fallback_base_ev_variant_b | current_optimizer_ev | fallback_optimizer_ev_variant_a_estimate | fallback_optimizer_ev_variant_b_estimate | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raphinha | Raphinha | BRA | FWD | 0.86 | missing_shares_estimated | 0.5445 | 4.501 | 3.0938 | 2.0459 | 4.2219 | 3.448 | high |
| Mahmoud Trezeguet | Mahmoud Trezeguet | EGY | MID | 0.92 | missing_shares_estimated | 0.4184 | 2.2218 | 1.6259 | 1.1577 | 2.1496 | 1.8218 | high |
| Neymar Jr. | Neymar Jr. | BRA | FWD | 0.8919 | missing_shares_estimated | 0.4958 | 4.5747 | 3.1399 | 1.9318 | 4.1752 | 3.386 | high |
| Kenan Yildiz | Kenan Yildiz | TUR | MID | 0.8953 | missing_shares_estimated | 0.4446 | 2.7102 | 1.3018 | 1.4542 | 2.7004 | 1.9257 | high |
| Christian Pulisic | Christian Pulisic | USA | FWD | 0.827 | missing_shares_estimated | 0.441 | 3.2732 | 2.0521 | 1.6431 | 3.2008 | 2.5292 | high |
| Viktor Gyökeres | Viktor Gyökeres | SWE | FWD | 0.8141 | missing_shares_estimated | 0.3826 | 2.4279 | 1.2512 | 1.9245 | 3.0494 | 2.4022 | high |
| Patrik Schick | Patrik Schick | CZE | FWD | 0.8543 | missing_shares_estimated | 0.2142 | 3.9864 | 3.6474 | 1.6347 | 3.7094 | 3.523 | medium |
| Hakan Calhanoglu | Hakan Calhanoglu | TUR | MID | 0.7024 | missing_shares_estimated | 0.4235 | 2.1565 | 1.0735 | 1.4427 | 2.3958 | 1.8002 | high |
| Salem Al-Dawsari | Salem Al-Dawsari | KSA | MID | 0.9032 | missing_shares_estimated | 0.3497 | 1.4856 | 1.605 | 1.1199 | 1.7447 | 1.8103 | medium |
| Federico Valverde | Federico Valverde | URU | MID | 0.8894 | missing_shares_estimated | 0.4886 | 2.7788 | 1.815 | 1.5913 | 2.851 | 2.3208 | high |
| Bruno Guimaraes | Bruno Guimaraes | BRA | MID | 0.8763 | missing_shares_estimated | 0.4846 | 3.542 | 2.4768 | 1.5296 | 3.2111 | 2.6253 | high |
| Romelu Lukaku | Romelu Lukaku | BEL | FWD | 0.6346 | missing_shares_estimated | 0.4372 | 3.1264 | 2.8197 | 0.5355 | 2.0146 | 1.8459 | high |
| Tomas Soucek | Tomas Soucek | CZE | MID | 0.9395 | missing_shares_estimated | 0.2498 | 3.1269 | 2.8487 | 1.264 | 2.8464 | 2.6935 | medium |
| Antonio Nusa | Antonio Nusa | NOR | MID | 0.82 | shares_present_no_fallback_needed | 3.6978 |  |  | 3.1605 |  |  |  |
| Brian Gutierrez | Brian Gutierrez | MEX | MID | 0.6683 | missing_shares_estimated | 0.4705 | 2.0989 | 1.153 | 0.5763 | 1.4719 | 0.9517 | high |

## Anbefaling

- Bedste validerede variant: **Variant A**.
- Produktionsvurdering: **egnet kun med konservativ cap/floor**.
- Variant B er den stærkeste sikkerhedsbarriere mod dobbeltallokering, men kan give nul/lav residual på hold, hvor kendte shares allerede fylder teammålet.
- Spillere med `low` confidence eller `*_no_team_residual` bør ikke auto-opdateres uden manuel review.
- Dette eksperiment skriver ikke shares eller EV tilbage til produktionsdata.
