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
| ev_diff | 1244 | 0.033192 | 0 | -0.961618 | 0.943638 |

## 3. EV Diff Pct Statistik

| Felt | N | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| ev_diff_pct | 1244 | 0.007501 | 0 | -1 | 0.770417 |

## 4. Retning

| Kategori | Antal |
| --- | --- |
| ev_diff > 0 | 383 |
| ev_diff = 0 | 539 |
| ev_diff < 0 | 322 |
| new_ev < 0 | 0 |

## 5. Top 30 EV-stigninger

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 5000000 | 2.569192 | 3.51283 | 0.943638 | 36.73% | match_1_clean_sheet_multiplier=1.450 |
| Roberto Alvarado | MEX | MID | 2500000 | 2.556785 | 3.444965 | 0.88818 | 34.74% | match_1_clean_sheet_multiplier=1.450 |
| Christoph Baumgartner | AUT | MID | 3500000 | 3.322945 | 4.080953 | 0.758008 | 22.81% | match_1_clean_sheet_multiplier=1.450 |
| Nuno Mendes | POR | DEF | 4500000 | 2.573691 | 3.326338 | 0.752647 | 29.24% | match_1_clean_sheet_multiplier=1.450 |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.007239 | 3.75827 | 0.751031 | 24.97% | match_1_clean_sheet_multiplier=1.332 |
| Vinicius Junior | BRA | FWD | 7500000 | 3.306427 | 4.056863 | 0.750436 | 22.70% | match_1_clean_sheet_multiplier=1.450 |
| Diogo Costa | POR | GK | 5000000 | 2.310142 | 3.039543 | 0.729401 | 31.57% | match_1_clean_sheet_multiplier=1.450 |
| Lautaro Martinez | ARG | FWD | 8000000 | 3.034915 | 3.75313 | 0.718215 | 23.67% | match_1_clean_sheet_multiplier=1.450 |
| Bart Verbruggen | NED | GK | 4500000 | 2.269777 | 2.961594 | 0.691817 | 30.48% | match_3_clean_sheet_multiplier=1.450 |
| Cristiano Ronaldo | POR | FWD | 7000000 | 3.372238 | 4.054403 | 0.682165 | 20.23% | match_1_clean_sheet_multiplier=1.450 |
| Raul Jimenez | MEX | FWD | 4500000 | 3.510416 | 4.164282 | 0.653866 | 18.63% | match_1_clean_sheet_multiplier=1.450 |
| Lionel Messi | ARG | FWD | 8000000 | 2.790715 | 3.433214 | 0.642499 | 23.02% | match_1_clean_sheet_multiplier=1.450 |
| Kylian Mbappe | FRA | FWD | 10000000 | 2.969889 | 3.605401 | 0.635512 | 21.40% | match_1_clean_sheet_multiplier=1.450 |
| Ousmane Dembele | FRA | FWD | 5500000 | 2.893142 | 3.528337 | 0.635195 | 21.96% | match_1_clean_sheet_multiplier=1.450 |
| Timothy Castagne | BEL | DEF | 3500000 | 1.916605 | 2.550533 | 0.633928 | 33.08% | match_2_clean_sheet_multiplier=1.450 |
| Fabian Rieder | SUI | MID | 3000000 | 1.83623 | 2.470008 | 0.633778 | 34.52% | match_1_clean_sheet_multiplier=1.450 |
| Gregor Kobel | SUI | GK | 4000000 | 2.040619 | 2.665576 | 0.624957 | 30.63% | match_1_clean_sheet_multiplier=1.450 |
| Lamine Yamal | ESP | FWD | 9000000 | 2.809651 | 3.428285 | 0.618634 | 22.02% | match_1_clean_sheet_multiplier=1.450 |
| Casemiro | BRA | MID | 4500000 | 2.495189 | 3.105383 | 0.610194 | 24.45% | match_1_clean_sheet_multiplier=1.450 |
| Jeremy Doku | BEL | FWD | 6500000 | 2.802155 | 3.410427 | 0.608272 | 21.71% | match_2_clean_sheet_multiplier=1.450 |
| Zeno Debast | BEL | DEF | 3500000 | 1.804955 | 2.40618 | 0.601225 | 33.31% | match_2_clean_sheet_multiplier=1.450 |
| Leandro Trossard | BEL | FWD | 5500000 | 2.721542 | 3.319496 | 0.597954 | 21.97% | match_2_clean_sheet_multiplier=1.450 |
| Mikel Oyarzabal | ESP | FWD | 7500000 | 2.744877 | 3.341914 | 0.597037 | 21.75% | match_1_clean_sheet_multiplier=1.450 |
| Cesar Montes | MEX | DEF | 3000000 | 2.163869 | 2.755528 | 0.591659 | 27.34% | match_1_clean_sheet_multiplier=1.450 |
| Jonathan Tah | GER | DEF | 4500000 | 1.816891 | 2.39536 | 0.578469 | 31.84% | match_1_clean_sheet_multiplier=1.450 |
| Martin Baturina | CRO | MID | 3000000 | 2.32408 | 2.891106 | 0.567026 | 24.40% | match_2_clean_sheet_multiplier=1.450 |
| Oscar Bobb | NOR | MID | 3000000 | 2.68048 | 3.237886 | 0.557406 | 20.79% | match_1_clean_sheet_multiplier=1.450 |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.061145 | 3.610009 | 0.548864 | 17.93% | match_1_clean_sheet_multiplier=1.450 |
| Juan Manuel Sanabria | URU | MID | 3000000 | 1.84833 | 2.389704 | 0.541374 | 29.29% | match_1_clean_sheet_multiplier=1.450 |
| Nico Williams | ESP | FWD | 4500000 | 2.502816 | 3.041058 | 0.538242 | 21.51% | match_1_clean_sheet_multiplier=1.450 |

## 6. Top 30 EV-fald

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hamdy Fathy | EGY | MID | 2500000 | 1.428735 | 0.467117 | -0.961618 | -67.31% | match_1_clean_sheet_multiplier=0.590 |
| Krepin Diatta | SEN | FWD | 2500000 | 1.400327 | 0.456416 | -0.943911 | -67.41% | match_1_clean_sheet_multiplier=0.550 |
| Micky van de Ven | NED | DEF | 4000000 | 0.849174 | 0 | -0.849174 | -100.00% | match_3_clean_sheet_multiplier=1.450 |
| Niko Sigur | CAN | MID | 2500000 | 1.676235 | 1.017955 | -0.65828 | -39.27% | match_2_clean_sheet_multiplier=1.450 |
| Akram Afif | QAT | FWD | 3500000 | 1.94504 | 1.323931 | -0.621109 | -31.93% | match_1_clean_sheet_multiplier=0.550 |
| Ali Abdi | TUN | DEF | 3000000 | 2.265619 | 1.673985 | -0.591634 | -26.11% | match_3_clean_sheet_multiplier=0.550 |
| Aymen Dahmen | TUN | GK | 3000000 | 1.901349 | 1.311814 | -0.589535 | -31.01% | match_3_clean_sheet_multiplier=0.550 |
| Montassar Talbi | TUN | DEF | 3000000 | 2.118769 | 1.543979 | -0.57479 | -27.13% | match_3_clean_sheet_multiplier=0.550 |
| Frantzdy Pierrot | HAI | FWD | 3000000 | 2.893683 | 2.331353 | -0.56233 | -19.43% | match_1_clean_sheet_multiplier=0.550 |
| Ivan Perisic | CRO | FWD | 4500000 | 1.937416 | 1.3818 | -0.555616 | -28.68% | match_2_clean_sheet_multiplier=1.450 |
| Orlando Mosquera | PAN | GK | 2500000 | 1.669728 | 1.127558 | -0.54217 | -32.47% | match_2_clean_sheet_multiplier=0.550 |
| Ismael Diaz | PAN | FWD | 2500000 | 3.366027 | 2.824168 | -0.541859 | -16.10% | match_2_clean_sheet_multiplier=0.550 |
| Finn Surman | NZL | DEF | 2000000 | 1.8252 | 1.298724 | -0.526476 | -28.84% | match_3_clean_sheet_multiplier=0.550 |
| Alan Franco | ECU | MID | 2500000 | 1.102035 | 0.579431 | -0.522604 | -47.42% | match_2_clean_sheet_multiplier=1.450 |
| Nestory Irankunda | AUS | FWD | 3500000 | 1.84769 | 1.332498 | -0.515192 | -27.88% | match_2_clean_sheet_multiplier=0.627 |
| Jearl Margaritha | CUW | FWD | 3000000 | 1.613283 | 1.118505 | -0.494778 | -30.67% | match_1_clean_sheet_multiplier=0.550 |
| Fidel Escobar | PAN | DEF | 2500000 | 1.721089 | 1.228796 | -0.492293 | -28.60% | match_2_clean_sheet_multiplier=0.550 |
| Yan Valery | TUN | DEF | 2500000 | 1.588539 | 1.136814 | -0.451725 | -28.44% | match_3_clean_sheet_multiplier=0.550 |
| Lyle Foster | RSA | FWD | 3500000 | 3.26504 | 2.815006 | -0.450034 | -13.78% | match_1_clean_sheet_multiplier=0.550 |
| Hussein Ali | IRQ | DEF | 3000000 | 1.187069 | 0.744013 | -0.443056 | -37.32% | match_1_clean_sheet_multiplier=0.550 |
| Jorge Gutierrez | PAN | DEF | 2500000 | 1.662789 | 1.237578 | -0.425211 | -25.57% | match_2_clean_sheet_multiplier=0.550 |
| Saud Abdulhamid | KSA | DEF | 3000000 | 1.817369 | 1.398391 | -0.418978 | -23.05% | match_1_clean_sheet_multiplier=0.550 |
| Duckens Nazon | HAI | FWD | 3500000 | 2.21124 | 1.800692 | -0.410548 | -18.57% | match_1_clean_sheet_multiplier=0.550 |
| Dzenis Burnic | BIH | MID | 2500000 | 0.923835 | 0.515193 | -0.408642 | -44.23% | match_2_clean_sheet_multiplier=0.550 |
| Kenji Gorre | CUW | FWD | 2500000 | 1.332677 | 0.924969 | -0.407708 | -30.59% | match_1_clean_sheet_multiplier=0.550 |
| Jeremy Antonisse | CUW | FWD | 2000000 | 1.152861 | 0.749243 | -0.403618 | -35.01% | match_1_clean_sheet_multiplier=0.550 |
| Nawaf Al-Aqidi | KSA | GK | 2500000 | 1.540478 | 1.147148 | -0.39333 | -25.53% | match_1_clean_sheet_multiplier=0.550 |
| Danley Jean Jacques | HAI | MID | 3000000 | 1.88078 | 1.491445 | -0.389335 | -20.70% | match_1_clean_sheet_multiplier=0.550 |
| Merchas Doski | IRQ | DEF | 2500000 | 0.971439 | 0.605149 | -0.36629 | -37.71% | match_1_clean_sheet_multiplier=0.550 |
| Firas Al-Buraikan | KSA | FWD | 3500000 | 2.94714 | 2.585341 | -0.361799 | -12.28% | match_1_clean_sheet_multiplier=0.550 |

## 7. Top 20 EV-stigninger, pris <= 3.0 mio.

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Roberto Alvarado | MEX | MID | 2500000 | 2.556785 | 3.444965 | 0.88818 | 34.74% | match_1_clean_sheet_multiplier=1.450 |
| Fabian Rieder | SUI | MID | 3000000 | 1.83623 | 2.470008 | 0.633778 | 34.52% | match_1_clean_sheet_multiplier=1.450 |
| Cesar Montes | MEX | DEF | 3000000 | 2.163869 | 2.755528 | 0.591659 | 27.34% | match_1_clean_sheet_multiplier=1.450 |
| Martin Baturina | CRO | MID | 3000000 | 2.32408 | 2.891106 | 0.567026 | 24.40% | match_2_clean_sheet_multiplier=1.450 |
| Oscar Bobb | NOR | MID | 3000000 | 2.68048 | 3.237886 | 0.557406 | 20.79% | match_1_clean_sheet_multiplier=1.450 |
| Juan Manuel Sanabria | URU | MID | 3000000 | 1.84833 | 2.389704 | 0.541374 | 29.29% | match_1_clean_sheet_multiplier=1.450 |
| Arthur Theate | BEL | DEF | 3000000 | 1.562169 | 2.088237 | 0.526068 | 33.68% | match_2_clean_sheet_multiplier=1.450 |
| Daniel Svensson | SWE | DEF | 3000000 | 0.646969 | 1.145405 | 0.498436 | 77.04% | match_2_clean_sheet_multiplier=0.587 |
| Johan Vasquez | MEX | DEF | 3000000 | 1.782169 | 2.263573 | 0.481404 | 27.01% | match_1_clean_sheet_multiplier=1.450 |
| John Yeboah | ECU | MID | 3000000 | 2.34058 | 2.811576 | 0.470996 | 20.12% | match_2_clean_sheet_multiplier=1.450 |
| Alexis Vega | MEX | FWD | 2500000 | 2.565227 | 3.027265 | 0.462038 | 18.01% | match_1_clean_sheet_multiplier=1.450 |
| Chris Richards | USA | DEF | 3000000 | 2.435569 | 2.845278 | 0.409709 | 16.82% | match_2_clean_sheet_multiplier=1.373 |
| Ricardo Rodriguez | SUI | DEF | 3000000 | 1.490669 | 1.897863 | 0.407194 | 27.32% | match_1_clean_sheet_multiplier=1.450 |
| Ibrahima Konate | FRA | DEF | 3000000 | 1.294319 | 1.669525 | 0.375206 | 28.99% | match_1_clean_sheet_multiplier=1.450 |
| Nikola Vlasic | CRO | MID | 3000000 | 1.39788 | 1.768191 | 0.370311 | 26.49% | match_2_clean_sheet_multiplier=1.450 |
| Jorge Carrascal | COL | MID | 2500000 | 1.093235 | 1.440715 | 0.34748 | 31.78% | match_1_clean_sheet_multiplier=1.450 |
| Iliman Ndiaye | SEN | MID | 3000000 | 2.25258 | 2.594624 | 0.342044 | 15.18% | match_1_clean_sheet_multiplier=0.550 |
| Amadou Onana | BEL | MID | 3000000 | 1.46278 | 1.802136 | 0.339356 | 23.20% | match_2_clean_sheet_multiplier=1.450 |
| Patrick Agyemang | USA | FWD | 3000000 | 3.371633 | 3.709555 | 0.337922 | 10.02% | match_2_clean_sheet_multiplier=1.373 |
| Luiz Henrique | BRA | FWD | 3000000 | 1.454883 | 1.788892 | 0.334009 | 22.96% | match_1_clean_sheet_multiplier=1.450 |

## 8. Top 20 EV-fald, pris >= 4.0 mio.

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Micky van de Ven | NED | DEF | 4000000 | 0.849174 | 0 | -0.849174 | -100.00% | match_3_clean_sheet_multiplier=1.450 |
| Ivan Perisic | CRO | FWD | 4500000 | 1.937416 | 1.3818 | -0.555616 | -28.68% | match_2_clean_sheet_multiplier=1.450 |
| Antoine Semenyo | GHA | FWD | 4000000 | 1.789876 | 1.501987 | -0.287889 | -16.08% | match_2_clean_sheet_multiplier=0.550 |
| Alexander Isak | SWE | FWD | 5000000 | 1.952847 | 1.874739 | -0.078108 | -4.00% | match_2_clean_sheet_multiplier=0.587 |
| Antonio Sanabria | PAR | FWD | 4000000 | 2.172676 | 2.106056 | -0.06662 | -3.07% | match_1_clean_sheet_multiplier=0.756 |
| Heung-Min Son | KOR | FWD | 6000000 | 2.85439 | 2.794526 | -0.059864 | -2.10% | match_2_clean_sheet_multiplier=0.642 |
| Riyad Mahrez | ALG | FWD | 4000000 | 2.321176 | 2.26175 | -0.059426 | -2.56% | match_1_clean_sheet_multiplier=0.550 |
| Scott McTominay | SCO | MID | 4500000 | 3.048489 | 3.024042 | -0.024447 | -0.80% | match_1_clean_sheet_multiplier=1.450 |
| Jose Sa | POR | GK | 4500000 | 0.810077 | 0.794018 | -0.016059 | -1.98% | match_1_clean_sheet_multiplier=1.450 |
| Zion Suzuki | JPN | GK | 4000000 | 0.748669 | 0.744325 | -0.004344 | -0.58% | match_2_clean_sheet_multiplier=1.337 |
| Ørjan Nyland | NOR | GK | 4000000 | 0.805869 | 0.802053 | -0.003816 | -0.47% | match_1_clean_sheet_multiplier=1.450 |
| Emiliano Martinez | ARG | GK | 5000000 | 0.898292 | 0.894533 | -0.003759 | -0.42% | match_1_clean_sheet_multiplier=1.450 |
| Alexander Nübel | GER | GK | 4500000 | 0.836477 | 0.833302 | -0.003175 | -0.38% | match_1_clean_sheet_multiplier=1.450 |
| Alejandro 'Alex' Grimaldo | ESP | DEF | 4000000 | 0.853574 | 0.853574 | 0 | 0.00% | neutral |
| Nico O'Reilly | ENG | DEF | 4000000 | 0.853574 | 0.853574 | 0 | 0.00% | neutral |
| Alphonso Davies | CAN | DEF | 4000000 | 0.853574 | 0.853574 | 0 | 0.00% | neutral |
| Senne Lammens | BEL | GK | 4000000 | 0.654069 | 0.654069 | 0 | 0.00% | neutral |
| Thibaut Courtois | BEL | GK | 4500000 | 0.728677 | 0.728677 | 0 | 0.00% | neutral |
| Alvaro Montero | COL | GK | 4000000 | 0.654069 | 0.654069 | 0 | 0.00% | neutral |
| Sergio Rochet | URU | GK | 4000000 | 0.654069 | 0.654069 | 0 | 0.00% | neutral |

## 9. Holdniveau: sum ev_diff pr. team_id

| Team | Spillere | Sum ev_diff |
| --- | --- | --- |
| POR | 27 | 8.078583 |
| FRA | 26 | 7.039038 |
| ARG | 26 | 6.152519 |
| ESP | 26 | 5.449831 |
| SUI | 26 | 5.274206 |
| BEL | 26 | 5.019273 |
| GER | 26 | 4.871277 |
| ENG | 26 | 4.765432 |
| MEX | 26 | 4.296338 |
| BRA | 26 | 3.514285 |
| NED | 24 | 3.32739 |
| CRO | 26 | 3.277699 |
| USA | 25 | 3.163115 |
| COL | 26 | 2.996979 |
| AUT | 25 | 2.744982 |
| NOR | 26 | 2.431878 |
| URU | 26 | 2.244879 |
| MAR | 29 | 2.04105 |
| ECU | 28 | 1.897383 |
| JPN | 26 | 1.883851 |
| CAN | 28 | 1.831906 |
| TUR | 27 | 1.529593 |
| KOR | 26 | 0.268695 |
| HOLDET_584 | 24 | 0 |
| CIV | 26 | 0 |
| SWE | 26 | -0.025126 |
| BIH | 26 | -0.041229 |
| SCO | 26 | -0.317348 |
| SEN | 28 | -0.396651 |
| EGY | 27 | -0.480728 |
| ALG | 30 | -0.520587 |
| PAR | 29 | -0.631156 |
| IRN | 23 | -0.701621 |
| GHA | 25 | -1.20594 |
| CPV | 26 | -1.451099 |
| UZB | 25 | -1.626721 |
| KSA | 19 | -1.775689 |
| JOR | 26 | -2.103777 |
| QAT | 22 | -2.211423 |
| AUS | 28 | -2.356622 |
| COD | 27 | -2.644558 |
| NZL | 26 | -3.077122 |
| HAI | 25 | -3.163486 |
| IRQ | 26 | -3.166229 |
| PAN | 24 | -3.440612 |
| RSA | 25 | -3.579271 |
| CUW | 26 | -3.91435 |
| TUN | 26 | -3.977594 |

## 10. Positionniveau: mean ev_diff pr. position

| Position | Spillere | Mean ev_diff |
| --- | --- | --- |
| DEF | 408 | 0.021556 |
| FWD | 274 | 0.039916 |
| GK | 152 | 0.011115 |
| MID | 410 | 0.048463 |

## 11. Potentielt mistænkelige outliers

| Spiller | Hold | Pos | Pris | Old EV | New EV | Diff | Diff pct | Main reason | Outlier reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Micky van de Ven | NED | DEF | 4000000 | 0.849174 | 0 | -0.849174 | -100.00% | match_3_clean_sheet_multiplier=1.450 | høj pris og stort fald |
| Ivan Perisic | CRO | FWD | 4500000 | 1.937416 | 1.3818 | -0.555616 | -28.68% | match_2_clean_sheet_multiplier=1.450 | høj pris og stort fald |
| Daniel Svensson | SWE | DEF | 3000000 | 0.646969 | 1.145405 | 0.498436 | 77.04% | match_2_clean_sheet_multiplier=0.587 | ev_diff_pct > 50% |

## 12. Kort vurdering

- Multipliers ser ud til at skubbe EV i den forventede retning: 23 hold har samlet positiv ændring, og 23 hold har samlet negativ ændring.
- Gennemsnitlig ændring er 0.033192, medianen er 0, og der er 539 uændrede spillere. Det peger på en moderat samlet effekt snarere end en total omskalering.
- Største absolutte spillerændring er 0.961618. Outlier-listen bør gennemgås, men totalbilledet ser rimeligt ud som første fixture-strength lag.
