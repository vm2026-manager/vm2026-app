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
| 1 | UZB | COL | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 2 | UZB | POR | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 3 | UZB | COD | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 1 | CAN | BIH | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 2 | CAN | QAT | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 3 | CAN | SUI | 8 | 0.03826 | 0.062164 | 0.023904 | 0 |
| 1 | ARG | ALG | 7 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 2 | ARG | AUT | 7 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 3 | ARG | JOR | 7 | 0.040672 | 0.064312 | 0.02364 | 0 |
| 1 | POR | COD | 6 | 0.039784 | 0.063136 | 0.023352 | 0 |
| 2 | POR | UZB | 6 | 0.039784 | 0.063136 | 0.023352 | 0 |
| 3 | POR | COL | 6 | 0.039784 | 0.063136 | 0.023352 | 0 |
| 1 | AUS | TUR | 7 | 0.04024 | 0.063556 | 0.023316 | 0 |
| 2 | AUS | USA | 7 | 0.04024 | 0.063556 | 0.023316 | 0 |

## NOR vs IRQ sanity

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.06848 | -0.056938 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066839 | -0.055574 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066567 | -0.055347 | 0.05566 |
| Antonio Nusa | 0.6563 | 0.137996 | 0.061668 | -0.051274 | 0.047884 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070128 | -0.058308 | 0.061312 |
