# Strategy Sanity Report

Denne rapport sammenligner strategi-presets uden at ændre optimizer, EV, player_pool eller UI.

## 1. Bedste Hold Pr. Strategi

| Strategi | Formation | Pris | Total score | Total EV | High risk |
| --- | --- | --- | --- | --- | --- |
| balanced | 3-4-3 | 50,000,000 | 39.581 | 39.666 | 4 |
| safe_starters | 3-4-3 | 49,500,000 | 39.535 | 38.893 | 1 |
| fixture_attack | 3-4-3 | 48,500,000 | 47.479 | 39.610 | 3 |
| clean_sheet_stack | 3-4-3 | 49,000,000 | 40.900 | 39.315 | 2 |
| long_run_value | 3-4-3 | 49,000,000 | 41.819 | 39.405 | 2 |

## 2. Spillerliste Pr. Strategi

### balanced

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.326 | 3.307 | 0.766 | 0.843 | medium_risk |
| Chris Richards | USA | DEF | 3000000 | 2.845 | 2.822 | 0.671 | 0.780 | high_risk |
| Cesar Montes | MEX | DEF | 3000000 | 2.756 | 2.723 | 0.642 | 0.744 | high_risk |
| Raul Jimenez | MEX | FWD | 4500000 | 4.164 | 4.123 | 0.701 | 0.815 | high_risk |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054 | 4.031 | 0.908 | 0.970 | medium_risk |
| Vinicius Junior | BRA | FWD | 7500000 | 4.057 | 4.013 | 0.744 | 0.810 | medium_risk |
| Mike Maignan | FRA | GK | 5000000 | 3.513 | 3.468 | 0.484 | 0.513 | medium_risk |
| Christoph Baumgartner | AUT | MID | 3500000 | 4.081 | 4.138 | 0.895 | 0.933 | low_risk |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.758 | 3.759 | 0.875 | 0.911 | low_risk |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.610 | 3.645 | 0.661 | 0.714 | medium_risk |
| Malik Tillman | USA | MID | 4000000 | 3.501 | 3.552 | 0.676 | 0.786 | high_risk |

### safe_starters

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.326 | 3.307 | 0.766 | 0.843 | medium_risk |
| Josko Gvardiol | CRO | DEF | 3500000 | 2.508 | 2.679 | 0.856 | 0.922 | medium_risk |
| Chris Richards | USA | DEF | 3000000 | 2.845 | 2.582 | 0.671 | 0.780 | high_risk |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054 | 4.211 | 0.908 | 0.970 | medium_risk |
| Jonathan David | CAN | FWD | 4500000 | 3.903 | 4.064 | 0.900 | 0.970 | medium_risk |
| Vinicius Junior | BRA | FWD | 7500000 | 4.057 | 4.013 | 0.744 | 0.810 | medium_risk |
| Mike Maignan | FRA | GK | 5000000 | 3.513 | 3.288 | 0.484 | 0.513 | medium_risk |
| Christoph Baumgartner | AUT | MID | 3500000 | 4.081 | 4.318 | 0.895 | 0.933 | low_risk |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.758 | 3.939 | 0.875 | 0.911 | low_risk |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.610 | 3.645 | 0.661 | 0.714 | medium_risk |
| Oscar Bobb | NOR | MID | 3000000 | 3.238 | 3.491 | 0.910 | 0.970 | medium_risk |

### fixture_attack

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.326 | 3.834 | 0.766 | 0.843 | medium_risk |
| Cesar Montes | MEX | DEF | 3000000 | 2.756 | 3.138 | 0.642 | 0.744 | high_risk |
| Chris Richards | USA | DEF | 3000000 | 2.845 | 3.109 | 0.671 | 0.780 | high_risk |
| Vinicius Junior | BRA | FWD | 7500000 | 4.057 | 4.979 | 0.744 | 0.810 | medium_risk |
| Raul Jimenez | MEX | FWD | 4500000 | 4.164 | 4.900 | 0.701 | 0.815 | high_risk |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054 | 4.890 | 0.908 | 0.970 | medium_risk |
| Mike Maignan | FRA | GK | 5000000 | 3.513 | 4.128 | 0.484 | 0.513 | medium_risk |
| Christoph Baumgartner | AUT | MID | 3500000 | 4.081 | 5.021 | 0.895 | 0.933 | low_risk |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.758 | 4.640 | 0.875 | 0.911 | low_risk |
| Roberto Alvarado | MEX | MID | 2500000 | 3.445 | 4.537 | 0.639 | 0.725 | medium_risk |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.610 | 4.304 | 0.661 | 0.714 | medium_risk |

### clean_sheet_stack

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.326 | 3.779 | 0.766 | 0.843 | medium_risk |
| Timothy Castagne | BEL | DEF | 3500000 | 2.551 | 3.058 | 0.777 | 0.819 | low_risk |
| Cesar Montes | MEX | DEF | 3000000 | 2.756 | 3.005 | 0.642 | 0.744 | high_risk |
| Raul Jimenez | MEX | FWD | 4500000 | 4.164 | 4.123 | 0.701 | 0.815 | high_risk |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054 | 4.031 | 0.908 | 0.970 | medium_risk |
| Vinicius Junior | BRA | FWD | 7500000 | 4.057 | 4.013 | 0.744 | 0.810 | medium_risk |
| Mike Maignan | FRA | GK | 5000000 | 3.513 | 3.860 | 0.484 | 0.513 | medium_risk |
| Christoph Baumgartner | AUT | MID | 3500000 | 4.081 | 4.138 | 0.895 | 0.933 | low_risk |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.758 | 3.759 | 0.875 | 0.911 | low_risk |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.610 | 3.645 | 0.661 | 0.714 | medium_risk |
| Roberto Alvarado | MEX | MID | 2500000 | 3.445 | 3.490 | 0.639 | 0.725 | medium_risk |

### long_run_value

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.326 | 3.684 | 0.766 | 0.843 | medium_risk |
| Chris Richards | USA | DEF | 3000000 | 2.845 | 2.830 | 0.671 | 0.780 | high_risk |
| Timothy Castagne | BEL | DEF | 3500000 | 2.551 | 2.781 | 0.777 | 0.819 | low_risk |
| Vinicius Junior | BRA | FWD | 7500000 | 4.057 | 4.451 | 0.744 | 0.810 | medium_risk |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.054 | 4.408 | 0.908 | 0.970 | medium_risk |
| Raul Jimenez | MEX | FWD | 4500000 | 4.164 | 4.091 | 0.701 | 0.815 | high_risk |
| Mike Maignan | FRA | GK | 5000000 | 3.513 | 3.966 | 0.484 | 0.513 | medium_risk |
| Christoph Baumgartner | AUT | MID | 3500000 | 4.081 | 4.215 | 0.895 | 0.933 | low_risk |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.610 | 3.889 | 0.661 | 0.714 | medium_risk |
| Kerem Akturkoglu | TUR | MID | 4500000 | 3.758 | 3.885 | 0.875 | 0.911 | low_risk |
| Roberto Alvarado | MEX | MID | 2500000 | 3.445 | 3.618 | 0.639 | 0.725 | medium_risk |

## 3. Overlap Mod Balanced

| Strategi | Overlap | Kun i strategi | Kun i balanced |
| --- | --- | --- | --- |
| balanced | 11 | - | - |
| safe_starters | 8 | Jonathan David, Josko Gvardiol, Oscar Bobb | Cesar Montes, Malik Tillman, Raul Jimenez |
| fixture_attack | 10 | Roberto Alvarado | Malik Tillman |
| clean_sheet_stack | 9 | Roberto Alvarado, Timothy Castagne | Chris Richards, Malik Tillman |
| long_run_value | 9 | Roberto Alvarado, Timothy Castagne | Cesar Montes, Malik Tillman |

## 4. Spillere Kun Valgt I Én Strategi

| Strategi | Spiller | Hold | Pos | Strategy score |
| --- | --- | --- | --- | --- |
| balanced | Malik Tillman | USA | MID | 3.552 |
| safe_starters | Josko Gvardiol | CRO | DEF | 2.679 |
| safe_starters | Jonathan David | CAN | FWD | 4.064 |
| safe_starters | Oscar Bobb | NOR | MID | 3.491 |

## 5. Hold-/Landefordeling

| Strategi | Fordeling |
| --- | --- |
| balanced | AUT:1; BRA:1; FRA:1; MEX:2; NOR:1; POR:2; TUR:1; USA:2 |
| safe_starters | AUT:1; BRA:1; CAN:1; CRO:1; FRA:1; NOR:2; POR:2; TUR:1; USA:1 |
| fixture_attack | AUT:1; BRA:1; FRA:1; MEX:3; NOR:1; POR:2; TUR:1; USA:1 |
| clean_sheet_stack | AUT:1; BEL:1; BRA:1; FRA:1; MEX:3; NOR:1; POR:2; TUR:1 |
| long_run_value | AUT:1; BEL:1; BRA:1; FRA:1; MEX:2; NOR:1; POR:2; TUR:1; USA:1 |

## 6. Start Og Availability

| Strategi | Avg start_prob | Avg conditional_start_prob | Availability risk |
| --- | --- | --- | --- |
| balanced | 0.7294 | 0.8019 | high_risk:4; low_risk:2; medium_risk:5 |
| safe_starters | 0.7883 | 0.8488 | high_risk:1; low_risk:2; medium_risk:8 |
| fixture_attack | 0.7261 | 0.7964 | high_risk:3; low_risk:2; medium_risk:6 |
| clean_sheet_stack | 0.7357 | 0.7999 | high_risk:2; low_risk:3; medium_risk:6 |
| long_run_value | 0.7384 | 0.8032 | high_risk:2; low_risk:3; medium_risk:6 |

## 7. Fixture Attack Ind Ift. Balanced

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Roberto Alvarado | MEX | MID | 2500000 | 3.445 | 4.537 | 0.639 | 0.725 | medium_risk |

## 8. Safe Starters Ind Ift. Balanced

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jonathan David | CAN | FWD | 4500000 | 3.903 | 4.064 | 0.900 | 0.970 | medium_risk |
| Josko Gvardiol | CRO | DEF | 3500000 | 2.508 | 2.679 | 0.856 | 0.922 | medium_risk |
| Oscar Bobb | NOR | MID | 3000000 | 3.238 | 3.491 | 0.910 | 0.970 | medium_risk |

## 9. Long Run Value Ind Ift. Balanced

| Spiller | Hold | Pos | Pris | EV | Strategy score | Start | Conditional | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Roberto Alvarado | MEX | MID | 2500000 | 3.445 | 3.618 | 0.639 | 0.725 | medium_risk |
| Timothy Castagne | BEL | DEF | 3500000 | 2.551 | 2.781 | 0.777 | 0.819 | low_risk |

## 10. Kort Vurdering

- Strategierne giver forskellige hold, men ikke radikalt forskellige: fixture_attack overlapper 10/11 med balanced, safe_starters overlapper 8/11.
- safe_starters ser relevant ud som preset, fordi high_risk falder fra 4 til 1 og avg conditional_start_prob stiger.
- fixture_attack bør vises med strategy score separat fra total_ev, fordi dens score indeholder fixture-boost og derfor ikke er direkte sammenlignelig med balanced total_ev.
- clean_sheet_stack differentierer noget, men bør stadig vurderes mod balanced overlap og defensiv sammensætning.
- Datagrundlag brugt til rapporten: 1244 EV-rækker og 1292 player_pool-rækker.
