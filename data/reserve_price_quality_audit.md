# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 246 | 68 | 13 |
| Appearance-skaleret | 246 | 0 | 13 |
| Base-cappet | 246 | 2 | 13 |
| Valgt hybrid | 246 | 0 | 13 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 174 | 1.787889 | 1.787889 | 1.732866 | 1.787889 | 0 |
| likely_starter | 95 | 3.099312 | 3.099312 | 3.065162 | 3.099312 | 0 |
| other | 702 | 1.437985 | 1.127639 | 1.189617 | 1.061246 | -26.199142 |
| premium | 27 | 3.336706 | 3.196944 | 3.158285 | 3.135062 | -6.04322 |
| reserve_start_lt_0_10 | 246 | 0.66201 | 0.152969 | 0.253731 | 0.152874 | -76.907566 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.747057 | 1.483131 | 1.578029 | 1.470127 | -15.851224 |
| FWD | 274 | 1.583458 | 1.264314 | 1.266623 | 1.180116 | -25.472222 |
| GK | 152 | 1.344584 | 0.904967 | 0.977067 | 0.903701 | -32.789526 |
| MID | 410 | 1.26072 | 1.053619 | 1.063676 | 1.005486 | -20.245086 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.511834 | 4.511834 | 4.511834 | 4.511834 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.59947 | 4.342809 | 4.342809 | 4.342809 | 4.342809 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344694 | 4.161869 | 4.161869 | 4.161869 | 4.161869 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797084 | 3.90794 | 3.846855 | 3.90794 | 3.846855 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442622 | 3.693655 | 3.693655 | 3.693655 | 3.693655 |
| Raphinha | FWD | 6.5 | 0.86 | 3.044547 | 3.419943 | 3.419943 | 3.419943 | 3.419943 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042961 | 3.19032 | 3.19032 | 3.19032 | 3.19032 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697805 | 3.161444 | 3.161444 | 3.161444 | 3.161444 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766437 | 3.038232 | 3.038232 | 3.038232 | 3.038232 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047049 | 2.910433 | 2.910433 | 2.910433 | 2.910433 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702044 | 2.921778 | 2.88363 | 2.921778 | 2.88363 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.78777 | 2.247221 | 2.247221 | 2.190018 | 2.247221 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086725 | 2.919986 | 2.084986 | 2.556238 | 2.084986 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950099 | 2.200206 | 2.018815 | 2.200206 | 2.018815 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326488 | 2.318195 | 1.220908 | 1.624948 | 1.220908 |
| Yousef Qashi | MID | 2 | 0.8559 | 1.03004 | 0.790231 | 0.790231 | 0.790231 | 0.790231 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.578325 | 0.677117 | 1.378704 | 0.677117 |
| Igor Thiago | FWD | 4 | 0.047 | 0.482282 | 1.665961 | 0.359302 | 0.590795 | 0.359302 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.709206 | 0.308375 | 0.496238 | 0.308375 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
