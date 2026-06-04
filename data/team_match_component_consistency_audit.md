# Team Match Component Consistency Audit

High-start betyder `start_prob >= 0.70`. Spreads er beregnet pr. hold/kamp/komponent.

## Foer/efter

- Team/match/on_pitch high-start spreads > 0.05 foer: 0
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.031182
- Stoerste high-start on_pitch spread efter: 0.031182

## Stoerste resterende high-start on_pitch spreads

| match_no | team_id | opponent | high_start_players | high_start_min | high_start_max | high_start_spread | high_start_negative_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AUT | JOR | 12 | 0.034528 | 0.06571 | 0.031182 | 0 |
| 2 | AUT | ARG | 12 | 0.034528 | 0.06571 | 0.031182 | 0 |
| 3 | AUT | ALG | 12 | 0.034528 | 0.06571 | 0.031182 | 0 |
| 1 | GER | CUW | 11 | 0.035512 | 0.063517 | 0.028005 | 0 |
| 2 | GER | CIV | 11 | 0.035512 | 0.063517 | 0.028005 | 0 |
| 3 | GER | ECU | 11 | 0.035512 | 0.063517 | 0.028005 | 0 |
| 1 | CRO | ENG | 10 | 0.034384 | 0.062267 | 0.027883 | 0 |
| 2 | CRO | PAN | 10 | 0.034384 | 0.062267 | 0.027883 | 0 |
| 3 | CRO | GHA | 10 | 0.034384 | 0.062267 | 0.027883 | 0 |
| 1 | SEN | FRA | 12 | 0.03616 | 0.063606 | 0.027446 | 0 |
| 2 | SEN | NOR | 12 | 0.03616 | 0.063606 | 0.027446 | 0 |
| 3 | SEN | IRQ | 12 | 0.03616 | 0.063606 | 0.027446 | 0 |
| 1 | NOR | IRQ | 13 | 0.034936 | 0.061312 | 0.026376 | 0 |
| 2 | NOR | SEN | 13 | 0.034936 | 0.061312 | 0.026376 | 0 |
| 3 | NOR | FRA | 13 | 0.034936 | 0.061312 | 0.026376 | 0 |
| 1 | AUS | TUR | 11 | 0.034252 | 0.059911 | 0.025659 | 0 |
| 2 | AUS | USA | 11 | 0.034252 | 0.059911 | 0.025659 | 0 |
| 3 | AUS | PAR | 11 | 0.034252 | 0.059911 | 0.025659 | 0 |
| 1 | IRQ | NOR | 8 | 0.036772 | 0.062224 | 0.025452 | 0 |
| 2 | IRQ | FRA | 8 | 0.036772 | 0.062224 | 0.025452 | 0 |

## NOR vs IRQ sanity

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Oscar Bobb | 0.9101 | 0.156927 | 0.070127 | -0.058307 | 0.061312 |
| Erling Haaland | 0.8883 | 0.150278 | 0.067155 | -0.055836 | 0.056596 |
| Martin Ødegaard | 0.8375 | 0.141684 | 0.063315 | -0.052643 | 0.0505 |
| Antonio Nusa | 0.8157 | 0.137996 | 0.061667 | -0.051273 | 0.047884 |
| Alexander Sørloth | 0.8078 | 0.13666 | 0.061069 | -0.050776 | 0.046936 |
