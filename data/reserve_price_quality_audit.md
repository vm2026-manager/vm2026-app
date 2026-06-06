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
| cheap_real_starter | 174 | 1.771169 | 1.771169 | 1.705718 | 1.771169 | 0 |
| likely_starter | 93 | 3.07582 | 3.07582 | 3.007095 | 3.07582 | 0 |
| other | 704 | 1.440781 | 1.130533 | 1.192851 | 1.064337 | -26.127802 |
| premium | 27 | 3.313 | 3.173256 | 3.117288 | 3.111416 | -6.084646 |
| reserve_start_lt_0_10 | 246 | 0.661994 | 0.152968 | 0.253731 | 0.152873 | -76.907182 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.747058 | 1.483131 | 1.578029 | 1.470128 | -15.851221 |
| FWD | 274 | 1.573285 | 1.253092 | 1.248482 | 1.168919 | -25.702013 |
| GK | 152 | 1.344586 | 0.904967 | 0.977068 | 0.903702 | -32.789562 |
| MID | 410 | 1.250218 | 1.042484 | 1.044811 | 0.994351 | -20.465759 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.511834 | 4.511834 | 4.511834 | 4.511834 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.59947 | 4.342521 | 4.342521 | 4.342521 | 4.342521 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344694 | 4.161869 | 4.161869 | 4.161869 | 4.161869 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797085 | 3.907652 | 3.846577 | 3.907652 | 3.846577 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442622 | 3.693369 | 3.693369 | 3.693369 | 3.693369 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042962 | 3.190083 | 3.190083 | 3.190083 | 3.190083 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697805 | 3.161444 | 3.161444 | 3.161444 | 3.161444 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766437 | 3.037994 | 3.037994 | 3.037994 | 3.037994 |
| Raphinha | FWD | 6.5 | 0.86 | 2.198878 | 2.954548 | 2.954548 | 2.693626 | 2.954548 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047049 | 2.910433 | 2.910433 | 2.910433 | 2.910433 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702045 | 2.921779 | 2.883631 | 2.921779 | 2.883631 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787772 | 2.247222 | 2.247222 | 2.190021 | 2.247222 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086725 | 2.919986 | 2.084986 | 2.556238 | 2.084986 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950099 | 2.200206 | 2.018815 | 2.200206 | 2.018815 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326489 | 2.318196 | 1.220909 | 1.624949 | 1.220909 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.578182 | 0.677109 | 1.378704 | 0.677109 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.793914 | 0.660362 | 0.660362 | 0.660362 | 0.660362 |
| Igor Thiago | FWD | 4 | 0.047 | 0.482283 | 1.665743 | 0.359289 | 0.590797 | 0.359289 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.709206 | 0.308375 | 0.496238 | 0.308375 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
