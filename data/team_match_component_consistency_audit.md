# Team Match Component Consistency Audit

High-start betyder `start_prob >= 0.70`. Spreads er beregnet pr. hold/kamp/komponent.

## Foer/efter

- Team/match/on_pitch high-start spreads > 0.05 foer: 6
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
| 1 | CAN | BIH | 9 | 0.03724 | 0.062428 | 0.025188 | 0 |
| 2 | CAN | QAT | 9 | 0.03724 | 0.062428 | 0.025188 | 0 |
| 3 | CAN | SUI | 9 | 0.03724 | 0.062428 | 0.025188 | 0 |
| 1 | PAN | GHA | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 2 | PAN | CRO | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 3 | PAN | ENG | 3 | 0.03508 | 0.06 | 0.02492 | 0 |
| 1 | UZB | COL | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 2 | UZB | POR | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 3 | UZB | COD | 8 | 0.038212 | 0.06268 | 0.024468 | 0 |
| 1 | EGY | BEL | 6 | 0.044392 | 0.0676 | 0.023208 | 0 |
| 2 | EGY | NZL | 6 | 0.044392 | 0.0676 | 0.023208 | 0 |
| 3 | EGY | IRN | 6 | 0.044392 | 0.0676 | 0.023208 | 0 |
| 1 | NED | JPN | 6 | 0.040612 | 0.063256 | 0.022644 | 0 |
| 2 | NED | SWE | 6 | 0.040612 | 0.063256 | 0.022644 | 0 |
| 3 | NED | TUN | 6 | 0.040612 | 0.063256 | 0.022644 | 0 |
| 1 | ARG | ALG | 5 | 0.0418 | 0.064312 | 0.022512 | 0 |
| 2 | ARG | AUT | 5 | 0.0418 | 0.064312 | 0.022512 | 0 |

## NOR vs IRQ sanity

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.068484 | -0.056942 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066844 | -0.055579 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066572 | -0.055352 | 0.05566 |
| Antonio Nusa | 0.82 | 0.162408 | 0.072582 | -0.06035 | 0.0652 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070133 | -0.058313 | 0.061312 |
