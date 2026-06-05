# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 245 | 67 | 13 |
| Appearance-skaleret | 245 | 0 | 13 |
| Base-cappet | 245 | 2 | 13 |
| Valgt hybrid | 245 | 0 | 13 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 170 | 1.682022 | 1.682022 | 1.568217 | 1.682022 | 0 |
| likely_starter | 93 | 2.989768 | 2.989768 | 2.844555 | 2.989768 | 0 |
| other | 709 | 1.44235 | 1.133497 | 1.199736 | 1.068437 | -25.923869 |
| premium | 27 | 3.257055 | 3.115228 | 2.997995 | 3.002758 | -7.807571 |
| reserve_start_lt_0_10 | 245 | 0.657316 | 0.152094 | 0.252464 | 0.152001 | -76.875481 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.72804 | 1.463124 | 1.564703 | 1.450967 | -16.033968 |
| FWD | 274 | 1.549303 | 1.230809 | 1.200242 | 1.141839 | -26.299854 |
| GK | 152 | 1.339844 | 0.902809 | 0.978202 | 0.901564 | -32.711264 |
| MID | 410 | 1.225356 | 1.018519 | 0.996587 | 0.970568 | -20.792922 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.503453 | 4.503453 | 4.503453 | 4.503453 |
| Harry Kane | FWD | 9.5 | 0.92 | 4.340567 | 4.201547 | 4.201547 | 4.201547 | 4.201547 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344691 | 4.153486 | 4.153486 | 4.153486 | 4.153486 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797083 | 3.909095 | 3.847971 | 3.909095 | 3.847971 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442615 | 3.694746 | 3.694746 | 3.694746 | 3.694746 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.04296 | 3.190534 | 3.190534 | 3.190534 | 3.190534 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766432 | 3.038444 | 3.038444 | 3.038444 | 3.038444 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047048 | 2.904688 | 2.904688 | 2.904688 | 2.904688 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702042 | 2.902611 | 2.864974 | 2.902611 | 2.864974 |
| Antonio Nusa | MID | 3.5 | 0.6563 | 2.966811 | 2.758405 | 2.688069 | 2.758405 | 2.688069 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787764 | 2.246305 | 2.246305 | 2.190011 | 2.246305 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086723 | 2.911604 | 2.080552 | 2.556236 | 2.080552 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950097 | 2.199212 | 2.017981 | 2.199212 | 2.017981 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326487 | 2.296467 | 1.214187 | 1.624946 | 1.214187 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.576808 | 0.677026 | 1.378704 | 0.677026 |
| Raphinha | FWD | 6.5 | 0.6781 | 0.488147 | 2.014848 | 1.960211 | 0.59798 | 0.59798 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280038 | 0.376212 | 0.376212 | 0.343047 | 0.376212 |
| Igor Thiago | FWD | 4 | 0.047 | 0.482279 | 1.665813 | 0.359291 | 0.590792 | 0.359291 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.702226 | 0.307973 | 0.496238 | 0.307973 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
