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
| cheap_real_starter | 174 | 1.721724 | 1.721724 | 1.609663 | 1.721724 | 0 |
| likely_starter | 95 | 3.004645 | 3.004645 | 2.862489 | 3.004645 | 0 |
| other | 702 | 1.437179 | 1.127171 | 1.189375 | 1.060968 | -26.177075 |
| premium | 27 | 3.26348 | 3.123677 | 3.00583 | 3.061659 | -6.184206 |
| reserve_start_lt_0_10 | 246 | 0.661241 | 0.152909 | 0.2537 | 0.152814 | -76.889807 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.747057 | 1.483131 | 1.578029 | 1.470127 | -15.851224 |
| FWD | 274 | 1.549937 | 1.231642 | 1.201015 | 1.147644 | -25.95548 |
| GK | 152 | 1.344584 | 0.904967 | 0.977067 | 0.903701 | -32.789526 |
| MID | 410 | 1.226442 | 1.019777 | 0.997802 | 0.971826 | -20.760524 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.511834 | 4.511834 | 4.511834 | 4.511834 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.59947 | 4.343945 | 4.343945 | 4.343945 | 4.343945 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344694 | 4.161869 | 4.161869 | 4.161869 | 4.161869 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797084 | 3.909096 | 3.847972 | 3.909096 | 3.847972 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442622 | 3.694751 | 3.694751 | 3.694751 | 3.694751 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042961 | 3.190535 | 3.190535 | 3.190535 | 3.190535 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697805 | 3.160453 | 3.160453 | 3.160453 | 3.160453 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766437 | 3.038447 | 3.038447 | 3.038447 | 3.038447 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047049 | 2.910433 | 2.910433 | 2.910433 | 2.910433 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702044 | 2.921778 | 2.88363 | 2.921778 | 2.88363 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.78777 | 2.246309 | 2.246309 | 2.190018 | 2.246309 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086725 | 2.919986 | 2.084986 | 2.556238 | 2.084986 |
| Raphinha | FWD | 6.5 | 0.86 | 0.544547 | 2.045868 | 2.045868 | 0.66707 | 2.045868 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950099 | 2.199215 | 2.017983 | 2.199215 | 2.017983 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326488 | 2.318195 | 1.220908 | 1.624948 | 1.220908 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.576809 | 0.677026 | 1.378704 | 0.677026 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280039 | 0.376213 | 0.376213 | 0.343048 | 0.376213 |
| Igor Thiago | FWD | 4 | 0.047 | 0.482282 | 1.665816 | 0.359293 | 0.590795 | 0.359293 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.709206 | 0.308375 | 0.496238 | 0.308375 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
