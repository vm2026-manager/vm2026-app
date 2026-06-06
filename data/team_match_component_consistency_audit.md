# Team Match Component Consistency Audit

High-start betyder `start_prob >= 0.70`. Spreads er beregnet pr. hold/kamp/komponent.

## Foer/efter

- Team/match/on_pitch high-start spreads > 0.05 foer: 3
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.067600
- Stoerste high-start on_pitch spread efter: 0.027252

## Stoerste resterende high-start on_pitch spreads

| match_no | team_id | opponent | high_start_players | high_start_min | high_start_max | high_start_spread | high_start_negative_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | IRN | NZL | 7 | 0.036004 | 0.063256 | 0.027252 | 0 |
| 2 | IRN | BEL | 7 | 0.036004 | 0.063256 | 0.027252 | 0 |
| 3 | IRN | EGY | 7 | 0.036004 | 0.063256 | 0.027252 | 0 |
| 1 | PAN | GHA | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 2 | PAN | CRO | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 3 | PAN | ENG | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 1 | BRA | MAR | 9 | 0.040336 | 0.0652 | 0.024864 | 0 |
| 2 | BRA | HAI | 9 | 0.040336 | 0.0652 | 0.024864 | 0 |
| 3 | BRA | SCO | 9 | 0.040336 | 0.0652 | 0.024864 | 0 |
| 1 | UZB | COL | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 2 | UZB | POR | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 3 | UZB | COD | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 1 | CAN | BIH | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 2 | CAN | QAT | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 3 | CAN | SUI | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 1 | ARG | ALG | 6 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 2 | ARG | AUT | 6 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 3 | ARG | JOR | 6 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 1 | EGY | BEL | 8 | 0.04408 | 0.0676 | 0.02352 | 0 |
| 2 | EGY | NZL | 8 | 0.04408 | 0.0676 | 0.02352 | 0 |

## NOR vs IRQ sanity

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.068481 | -0.056939 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.06684 | -0.055575 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066568 | -0.055349 | 0.05566 |
| Antonio Nusa | 0.82 | 0.162408 | 0.072578 | -0.060346 | 0.0652 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070129 | -0.058309 | 0.061312 |
