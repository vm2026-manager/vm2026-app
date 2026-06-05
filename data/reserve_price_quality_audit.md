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
| cheap_real_starter | 170 | 1.68511 | 1.68511 | 1.570833 | 1.68511 | 0 |
| likely_starter | 94 | 2.993856 | 2.993856 | 2.848792 | 2.993856 | 0 |
| other | 708 | 1.444064 | 1.134561 | 1.198919 | 1.068587 | -26.001375 |
| premium | 27 | 3.274975 | 3.131395 | 3.006695 | 3.013159 | -7.994433 |
| reserve_start_lt_0_10 | 245 | 0.658837 | 0.152177 | 0.252283 | 0.152069 | -76.918551 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.728667 | 1.463751 | 1.565177 | 1.451594 | -16.028144 |
| FWD | 274 | 1.565516 | 1.244673 | 1.207794 | 1.153083 | -26.344859 |
| GK | 152 | 1.339446 | 0.903017 | 0.97832 | 0.901773 | -32.675693 |
| MID | 410 | 1.225084 | 1.018247 | 0.996147 | 0.970241 | -20.802091 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.503453 | 4.503453 | 4.503453 | 4.503453 |
| Harry Kane | FWD | 9.5 | 0.92 | 4.307649 | 4.21109 | 4.21109 | 4.21109 | 4.21109 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344689 | 4.153485 | 4.153485 | 4.153485 | 4.153485 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.79708 | 3.936845 | 3.874791 | 3.936845 | 3.874791 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442623 | 3.722188 | 3.722188 | 3.722188 | 3.722188 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042958 | 3.213345 | 3.213345 | 3.213345 | 3.213345 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766437 | 3.061258 | 3.061258 | 3.061258 | 3.061258 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047047 | 2.904686 | 2.904686 | 2.904686 | 2.904686 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.70204 | 2.90261 | 2.864972 | 2.90261 | 2.864972 |
| Antonio Nusa | MID | 3.5 | 0.6563 | 2.966819 | 2.758409 | 2.688073 | 2.758409 | 2.688073 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787772 | 2.246309 | 2.246309 | 2.190021 | 2.246309 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.08672 | 2.911602 | 2.08055 | 2.556232 | 2.08055 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950095 | 2.199211 | 2.01798 | 2.199211 | 2.01798 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326483 | 2.296464 | 1.214185 | 1.624942 | 1.214185 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125472 | 1.590525 | 0.677856 | 1.378703 | 0.677856 |
| Raphinha | FWD | 6.5 | 0.6781 | 0.485575 | 2.039976 | 1.98451 | 0.59483 | 0.59483 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280035 | 0.376209 | 0.376209 | 0.343043 | 0.376209 |
| Igor Thiago | FWD | 4 | 0.047 | 0.47973 | 1.68533 | 0.359294 | 0.587669 | 0.359294 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.702226 | 0.307973 | 0.496238 | 0.307973 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
