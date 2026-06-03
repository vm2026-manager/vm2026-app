# Fixture Strength EV Sanity

Denne rapport er kun en sanity-check af fixture-strength impact. Den ændrer ikke modeldata.

## 1. Datadækning

| Kilde | Rækker |
| --- | --- |
| player_ev_group_stage_v1.csv | 1244 |
| player_ev_fixture_strength_impact_report.csv | 1244 |
| fixture_strength_multipliers.csv | 72 |
| player_pool_v1.json | 1292 |

## 2. EV Diff Statistik

| Felt | N | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| ev_diff | 1244 | 0.067216 | 0 | -0.519385 | 0.988106 |

## 3. EV Diff Pct Statistik

| Felt | N | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| ev_diff_pct | 1244 | 0.018297 | 0 | -1 | 0.278407 |

## 4. Retning

| Kategori | Antal |
| --- | --- |
| ev_diff > 0 | 431 |
| ev_diff = 0 | 576 |
| ev_diff < 0 | 237 |
| new_ev < 0 | 0 |

## 5. Top 30 EV-stigninger

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vinicius Junior | BRA | FWD | 7500000 | 4.056863 | 5.044969 | 0.988106 | 24.36% | match_2_clean_sheet_multiplier=1.437 |
| Lautaro Martinez | ARG | FWD | 8000000 | 3.75313 | 4.692971 | 0.939841 | 25.04% | match_1_goal_multiplier=1.350 |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054403 | 4.947107 | 0.892704 | 22.02% | match_1_goal_multiplier=1.350 |
| Kylian Mbappe | FRA | FWD | 10000000 | 3.605401 | 4.441727 | 0.836326 | 23.20% | match_2_clean_sheet_multiplier=1.450 |
| Lionel Messi | ARG | FWD | 8000000 | 3.433214 | 4.265962 | 0.832748 | 24.26% | match_1_goal_multiplier=1.350 |
| Raul Jimenez | MEX | FWD | 4500000 | 4.164282 | 4.992947 | 0.828665 | 19.90% | match_1_goal_multiplier=1.350 |
| Ousmane Dembele | FRA | FWD | 5500000 | 3.528337 | 4.355447 | 0.82711 | 23.44% | match_2_clean_sheet_multiplier=1.450 |
| Lamine Yamal | ESP | FWD | 9000000 | 3.428285 | 4.230405 | 0.80212 | 23.40% | match_2_clean_sheet_multiplier=1.450 |
| Mike Maignan | FRA | GK | 5000000 | 3.51283 | 4.314948 | 0.802118 | 22.83% | match_2_clean_sheet_multiplier=1.450 |
| Mikel Oyarzabal | ESP | FWD | 7500000 | 3.341914 | 4.133042 | 0.791128 | 23.67% | match_2_clean_sheet_multiplier=1.450 |
| Jeremy Doku | BEL | FWD | 6500000 | 3.410427 | 4.200406 | 0.789979 | 23.16% | match_2_goal_multiplier=1.350 |
| Casemiro | BRA | MID | 4500000 | 3.105383 | 3.892444 | 0.787061 | 25.35% | match_2_clean_sheet_multiplier=1.437 |
| Leandro Trossard | BEL | FWD | 5500000 | 3.319496 | 4.099767 | 0.780271 | 23.51% | match_2_goal_multiplier=1.350 |
| Matheus Cunha | BRA | FWD | 4500000 | 2.952107 | 3.663594 | 0.711487 | 24.10% | match_2_clean_sheet_multiplier=1.437 |
| Nico Williams | ESP | FWD | 4500000 | 3.041058 | 3.744479 | 0.703421 | 23.13% | match_2_clean_sheet_multiplier=1.450 |
| Alexis Mac Allister | ARG | MID | 4500000 | 2.602604 | 3.294383 | 0.691779 | 26.58% | match_1_goal_multiplier=1.350 |
| Rafael Leão | POR | FWD | 5500000 | 3.246504 | 3.934193 | 0.687689 | 21.18% | match_1_goal_multiplier=1.350 |
| Joshua Kimmich | GER | MID | 5000000 | 3.032026 | 3.719618 | 0.687592 | 22.68% | match_1_clean_sheet_multiplier=1.450 |
| Roberto Alvarado | MEX | MID | 2500000 | 3.444965 | 4.125843 | 0.680878 | 19.76% | match_1_goal_multiplier=1.350 |
| Bruno Fernandes | POR | MID | 7000000 | 3.116567 | 3.794886 | 0.678319 | 21.76% | match_1_goal_multiplier=1.350 |
| Harry Kane | ENG | FWD | 9500000 | 2.806929 | 3.48523 | 0.678301 | 24.17% | match_2_goal_multiplier=1.350 |
| Pedro Neto | POR | FWD | 5000000 | 3.183168 | 3.8554 | 0.672232 | 21.12% | match_1_goal_multiplier=1.350 |
| Cody Gakpo | NED | FWD | 6500000 | 3.799432 | 4.462305 | 0.662873 | 17.45% | match_3_goal_multiplier=1.311 |
| Nuno Mendes | POR | DEF | 4500000 | 3.326338 | 3.974216 | 0.647878 | 19.48% | match_1_goal_multiplier=1.350 |
| Florian Wirtz | GER | MID | 7500000 | 2.785345 | 3.431015 | 0.64567 | 23.18% | match_1_clean_sheet_multiplier=1.450 |
| Julian Alvarez | ARG | FWD | 6000000 | 2.677767 | 3.319539 | 0.641772 | 23.97% | match_1_goal_multiplier=1.350 |
| Giovani Lo Celso | ARG | MID | 4000000 | 2.448174 | 3.082398 | 0.634224 | 25.91% | match_1_goal_multiplier=1.350 |
| Breel Embolo | SUI | FWD | 5000000 | 3.332048 | 3.963865 | 0.631817 | 18.96% | match_1_goal_multiplier=1.350 |
| Marcus Thuram | FRA | FWD | 5000000 | 2.642664 | 3.258079 | 0.615415 | 23.29% | match_2_clean_sheet_multiplier=1.450 |
| Michael Olise | FRA | FWD | 7000000 | 2.564607 | 3.176348 | 0.611741 | 23.85% | match_2_clean_sheet_multiplier=1.450 |

## 6. Top 30 EV-fald

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nathan Ake | NED | DEF | 3500000 | 0.519385 | 0 | -0.519385 | -100.00% | match_3_goal_multiplier=1.311 |
| Frantzdy Pierrot | HAI | FWD | 3000000 | 2.331353 | 1.905336 | -0.426017 | -18.27% | match_2_clean_sheet_multiplier=0.628 |
| Ismael Diaz | PAN | FWD | 2500000 | 2.824168 | 2.401938 | -0.42223 | -14.95% | match_3_clean_sheet_multiplier=0.709 |
| Elias Achouri | TUN | MID | 3500000 | 2.469108 | 2.079971 | -0.389137 | -15.76% | match_3_goal_multiplier=0.750 |
| Lyle Foster | RSA | FWD | 3500000 | 2.815006 | 2.452528 | -0.362478 | -12.88% | match_1_goal_multiplier=0.750 |
| Duckens Nazon | HAI | FWD | 3500000 | 1.800692 | 1.488241 | -0.312451 | -17.35% | match_2_clean_sheet_multiplier=0.628 |
| Danley Jean Jacques | HAI | MID | 3000000 | 1.491445 | 1.187914 | -0.303531 | -20.35% | match_2_clean_sheet_multiplier=0.628 |
| Oswin Appollis | RSA | FWD | 3500000 | 2.255149 | 1.962746 | -0.292403 | -12.97% | match_1_goal_multiplier=0.750 |
| Livano Comenencia | CUW | DEF | 2500000 | 1.350071 | 1.059567 | -0.290504 | -21.52% | match_1_clean_sheet_multiplier=0.611 |
| Hannibal Mejbri | TUN | MID | 3000000 | 1.865573 | 1.578473 | -0.2871 | -15.39% | match_3_goal_multiplier=0.750 |
| Akram Afif | QAT | FWD | 3500000 | 1.323931 | 1.044611 | -0.27932 | -21.10% | match_1_clean_sheet_multiplier=0.709 |
| Hazem Mastouri | TUN | FWD | 3500000 | 1.912805 | 1.636301 | -0.276504 | -14.46% | match_3_goal_multiplier=0.750 |
| Firas Al-Buraikan | KSA | FWD | 3500000 | 2.585341 | 2.31824 | -0.267101 | -10.33% | match_2_clean_sheet_multiplier=0.665 |
| Sarpreet Singh | NZL | MID | 2500000 | 1.513428 | 1.251183 | -0.262245 | -17.33% | match_3_clean_sheet_multiplier=0.709 |
| Juninho Bacuna | CUW | MID | 3000000 | 1.21257 | 0.952393 | -0.260177 | -21.46% | match_1_clean_sheet_multiplier=0.611 |
| Ellyes Skhiri | TUN | MID | 2500000 | 1.597664 | 1.341558 | -0.256106 | -16.03% | match_3_goal_multiplier=0.750 |
| Teboho Mokoena | RSA | MID | 3500000 | 1.89809 | 1.646024 | -0.252066 | -13.28% | match_1_goal_multiplier=0.750 |
| Ali Abdi | TUN | DEF | 3000000 | 1.673985 | 1.428574 | -0.245411 | -14.66% | match_3_goal_multiplier=0.750 |
| Finn Surman | NZL | DEF | 2000000 | 1.298724 | 1.054848 | -0.243876 | -18.78% | match_3_clean_sheet_multiplier=0.709 |
| Thalente Mbatha | RSA | MID | 3000000 | 0.616673 | 0.377633 | -0.23904 | -38.76% | match_1_goal_multiplier=0.750 |
| Leverton Pierre | HAI | MID | 2500000 | 1.209537 | 0.974655 | -0.234882 | -19.42% | match_2_clean_sheet_multiplier=0.628 |
| Anibal Godoy | PAN | MID | 2500000 | 1.458358 | 1.224906 | -0.233452 | -16.01% | match_3_clean_sheet_multiplier=0.709 |
| Marko Stamenic | NZL | MID | 2500000 | 1.330574 | 1.100007 | -0.230567 | -17.33% | match_3_clean_sheet_multiplier=0.709 |
| Jearl Margaritha | CUW | FWD | 3000000 | 1.118505 | 0.88842 | -0.230085 | -20.57% | match_1_clean_sheet_multiplier=0.611 |
| Cristian Martinez | PAN | MID | 2500000 | 1.387274 | 1.160103 | -0.227171 | -16.38% | match_3_clean_sheet_multiplier=0.709 |
| Amir Al Ammari | IRQ | MID | 3000000 | 1.157317 | 0.930982 | -0.226335 | -19.56% | match_2_clean_sheet_multiplier=0.652 |
| Eldor Shomurodov | UZB | FWD | 3000000 | 1.775374 | 1.549788 | -0.225586 | -12.71% | match_2_clean_sheet_multiplier=0.695 |
| Ali Jasim | IRQ | MID | 3000000 | 1.137157 | 0.912654 | -0.224503 | -19.74% | match_2_clean_sheet_multiplier=0.652 |
| Montassar Talbi | TUN | DEF | 3000000 | 1.543979 | 1.319562 | -0.224417 | -14.54% | match_3_goal_multiplier=0.750 |
| Carlos Harvey | PAN | MID | 2500000 | 1.413141 | 1.190545 | -0.222596 | -15.75% | match_3_clean_sheet_multiplier=0.709 |

## 7. Top 20 EV-stigninger, pris <= 3.0 mio.

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Roberto Alvarado | MEX | MID | 2500000 | 3.444965 | 4.125843 | 0.680878 | 19.76% | match_1_goal_multiplier=1.350 |
| Alexis Vega | MEX | FWD | 2500000 | 3.027265 | 3.611238 | 0.583973 | 19.29% | match_1_goal_multiplier=1.350 |
| Cesar Montes | MEX | DEF | 3000000 | 2.755528 | 3.324063 | 0.568535 | 20.63% | match_1_goal_multiplier=1.350 |
| Fabian Rieder | SUI | MID | 3000000 | 2.470008 | 2.968334 | 0.498326 | 20.18% | match_1_goal_multiplier=1.350 |
| Luiz Henrique | BRA | FWD | 3000000 | 1.788892 | 2.226198 | 0.437306 | 24.45% | match_2_clean_sheet_multiplier=1.437 |
| Amadou Onana | BEL | MID | 3000000 | 1.802136 | 2.237847 | 0.435711 | 24.18% | match_2_goal_multiplier=1.350 |
| Johan Vasquez | MEX | DEF | 3000000 | 2.263573 | 2.686858 | 0.423285 | 18.70% | match_1_goal_multiplier=1.350 |
| Juan Manuel Sanabria | URU | MID | 3000000 | 2.389704 | 2.805928 | 0.416224 | 17.42% | match_2_goal_multiplier=1.350 |
| N'Golo Kante | FRA | MID | 3000000 | 1.692853 | 2.106372 | 0.413519 | 24.43% | match_2_clean_sheet_multiplier=1.450 |
| Patrick Agyemang | USA | FWD | 3000000 | 3.709555 | 4.112482 | 0.402927 | 10.86% | match_2_goal_multiplier=1.237 |
| Arthur Theate | BEL | DEF | 3000000 | 2.088237 | 2.490073 | 0.401836 | 19.24% | match_2_goal_multiplier=1.350 |
| Warren Zaire-Emery | FRA | MID | 3000000 | 1.523593 | 1.909363 | 0.38577 | 25.32% | match_2_clean_sheet_multiplier=1.450 |
| Martin Baturina | CRO | MID | 3000000 | 2.891106 | 3.275011 | 0.383905 | 13.28% | match_2_goal_multiplier=1.329 |
| Ibrahima Konate | FRA | DEF | 3000000 | 1.669525 | 2.044122 | 0.374597 | 22.44% | match_2_clean_sheet_multiplier=1.450 |
| Jose Manuel Lopez | ARG | FWD | 2500000 | 1.454752 | 1.80328 | 0.348528 | 23.96% | match_1_goal_multiplier=1.350 |
| Jefferson Lerma | COL | MID | 3000000 | 1.821683 | 2.169622 | 0.347939 | 19.10% | match_1_goal_multiplier=1.350 |
| Felix Nmecha | GER | MID | 3000000 | 1.440595 | 1.788186 | 0.347591 | 24.13% | match_1_clean_sheet_multiplier=1.450 |
| Julian Quinones | MEX | FWD | 3000000 | 1.77013 | 2.117636 | 0.347506 | 19.63% | match_1_goal_multiplier=1.350 |
| Ricardo Rodriguez | SUI | DEF | 3000000 | 1.897863 | 2.230302 | 0.332439 | 17.52% | match_1_goal_multiplier=1.350 |
| Quinten Timber | NED | MID | 3000000 | 1.202874 | 1.530903 | 0.328029 | 27.27% | match_3_goal_multiplier=1.311 |

## 8. Top 20 EV-fald, pris >= 4.0 mio.

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Antoine Semenyo | GHA | FWD | 4000000 | 1.501987 | 1.441094 | -0.060893 | -4.05% | match_2_clean_sheet_multiplier=0.727 |
| Antonio Sanabria | PAR | FWD | 4000000 | 2.106056 | 2.066053 | -0.040003 | -1.90% | match_1_goal_multiplier=0.845 |
| Kang-In Lee | KOR | MID | 5000000 | 3.04752 | 3.024448 | -0.023072 | -0.76% | match_2_goal_multiplier=0.772 |
| Alexander Isak | SWE | FWD | 5000000 | 1.874739 | 1.854659 | -0.02008 | -1.07% | match_2_goal_multiplier=0.750 |
| Heung-Min Son | KOR | FWD | 6000000 | 2.794526 | 2.774766 | -0.01976 | -0.71% | match_2_goal_multiplier=0.772 |
| Jae-Sung Lee | KOR | MID | 4000000 | 1.817598 | 1.803417 | -0.014181 | -0.78% | match_2_goal_multiplier=0.772 |
| Raphinha | BRA | FWD | 6500000 | 1.496455 | 1.496455 | 0 | 0.00% | neutral |
| Darwin Nunez | URU | FWD | 6000000 | 1.46289 | 1.46289 | 0 | 0.00% | neutral |
| Viktor Gyökeres | SWE | FWD | 6000000 | 1.46289 | 1.46289 | 0 | 0.00% | neutral |
| Edin Dzeko | BIH | FWD | 4500000 | 1.292266 | 1.292266 | 0 | 0.00% | neutral |
| Luis Suarez Charris | COL | FWD | 4500000 | 1.292266 | 1.292266 | 0 | 0.00% | neutral |
| Federico Valverde | URU | MID | 5500000 | 1.261357 | 1.261357 | 0 | 0.00% | neutral |
| Lennart Karl | GER | MID | 5000000 | 1.237331 | 1.237331 | 0 | 0.00% | neutral |
| Rodri Hernandez | ESP | MID | 5000000 | 1.237331 | 1.237331 | 0 | 0.00% | neutral |
| Nico Paz | ARG | MID | 5000000 | 1.237331 | 1.237331 | 0 | 0.00% | neutral |
| Maxi Araujo | URU | MID | 5000000 | 1.237331 | 1.237331 | 0 | 0.00% | neutral |
| Matheus Nunes | POR | MID | 4500000 | 1.198289 | 1.198289 | 0 | 0.00% | neutral |
| Fabian Ruiz | ESP | MID | 4500000 | 1.198289 | 1.198289 | 0 | 0.00% | neutral |
| Alexis Saelemaekers | BEL | FWD | 4000000 | 1.185976 | 1.185976 | 0 | 0.00% | neutral |
| Christian Pulisic | USA | FWD | 4000000 | 1.185976 | 1.185976 | 0 | 0.00% | neutral |

## 9. Holdniveau: sum ev_diff pr. team_id

| Team | Spillere | Sum ev_diff |
| --- | --- | --- |
| POR | 27 | 9.669976 |
| FRA | 26 | 9.162932 |
| ARG | 26 | 8.336412 |
| ESP | 26 | 7.037324 |
| ENG | 26 | 6.740571 |
| GER | 26 | 6.555735 |
| BEL | 26 | 6.062315 |
| SUI | 26 | 5.586016 |
| BRA | 26 | 4.578904 |
| MEX | 26 | 4.465443 |
| CRO | 26 | 4.357168 |
| NED | 24 | 4.322504 |
| COL | 26 | 4.268313 |
| MAR | 29 | 3.641905 |
| AUT | 25 | 3.544841 |
| URU | 26 | 3.504804 |
| CAN | 28 | 3.484388 |
| ECU | 28 | 3.149524 |
| USA | 25 | 2.687265 |
| NOR | 26 | 2.36151 |
| JPN | 26 | 1.688971 |
| SEN | 28 | 1.368003 |
| TUR | 27 | 1.271769 |
| EGY | 27 | 0.971749 |
| SCO | 26 | 0.760417 |
| ALG | 30 | 0.204188 |
| BIH | 26 | 0.193074 |
| IRN | 23 | 0.177842 |
| HOLDET_584 | 24 | 0 |
| CIV | 26 | 0 |
| KOR | 26 | -0.09978 |
| SWE | 26 | -0.129099 |
| PAR | 29 | -0.384031 |
| GHA | 25 | -0.464052 |
| KSA | 19 | -0.975652 |
| CPV | 26 | -1.115454 |
| UZB | 25 | -1.276821 |
| QAT | 22 | -1.404298 |
| JOR | 26 | -1.518918 |
| AUS | 28 | -1.520536 |
| COD | 27 | -1.722948 |
| HAI | 25 | -2.035234 |
| IRQ | 26 | -2.115618 |
| PAN | 24 | -2.136391 |
| NZL | 26 | -2.227355 |
| RSA | 25 | -2.267533 |
| CUW | 26 | -2.438495 |
| TUN | 26 | -2.704978 |

## 10. Positionniveau: mean ev_diff pr. position

| Position | Spillere | Mean ev_diff |
| --- | --- | --- |
| DEF | 408 | 0.045435 |
| FWD | 274 | 0.114454 |
| GK | 152 | 0.019267 |
| MID | 410 | 0.075098 |

## 11. Potentielt mistænkelige outliers

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason | Outlier reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 12. Kort vurdering

- Multipliers ser ud til at skubbe EV i den forventede retning: 28 hold har samlet positiv ændring, og 18 hold har samlet negativ ændring.
- Gennemsnitlig ændring er 0.067216, medianen er 0, og der er 576 uændrede spillere. Det peger på en moderat samlet effekt snarere end en total omskalering.
- Største absolutte spillerændring er 0.988106. Outlier-listen bør gennemgås, men totalbilledet ser rimeligt ud som første fixture-strength lag.
