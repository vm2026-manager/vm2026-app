# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 246 | 70 | 12 |
| Appearance-skaleret | 246 | 0 | 12 |
| Base-cappet | 246 | 2 | 12 |
| Valgt hybrid | 246 | 0 | 12 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 173 | 1.730135 | 1.730135 | 1.63169 | 1.730135 | 0 |
| likely_starter | 91 | 3.060549 | 3.060549 | 2.953165 | 3.060549 | 0 |
| other | 707 | 1.442637 | 1.132898 | 1.195479 | 1.066845 | -26.049007 |
| premium | 27 | 3.31383 | 3.188948 | 3.068708 | 3.127003 | -5.637775 |
| reserve_start_lt_0_10 | 246 | 0.669625 | 0.153188 | 0.253204 | 0.153077 | -77.139878 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.744701 | 1.48165 | 1.577965 | 1.468975 | -15.803617 |
| FWD | 274 | 1.557854 | 1.236691 | 1.214396 | 1.152644 | -26.010785 |
| GK | 152 | 1.365124 | 0.912633 | 0.982188 | 0.911258 | -33.247254 |
| MID | 410 | 1.233611 | 1.025565 | 1.013465 | 0.976808 | -20.817185 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.556739 | 4.556739 | 4.556739 | 4.556739 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.599477 | 4.343365 | 4.343365 | 4.343365 | 4.343365 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344699 | 4.206777 | 4.206777 | 4.206777 | 4.206777 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797089 | 3.908507 | 3.847403 | 3.908507 | 3.847403 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442629 | 3.694189 | 3.694189 | 3.694189 | 3.694189 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042966 | 3.190353 | 3.190353 | 3.190353 | 3.190353 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697811 | 3.160298 | 3.160298 | 3.160298 | 3.160298 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766444 | 3.038266 | 3.038266 | 3.038266 | 3.038266 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047052 | 2.941206 | 2.941206 | 2.941206 | 2.941206 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702048 | 2.914288 | 2.876339 | 2.914288 | 2.876339 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787778 | 2.24592 | 2.24592 | 2.190028 | 2.24592 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086729 | 2.964894 | 2.108736 | 2.556243 | 2.108736 |
| Raphinha | FWD | 6.5 | 0.86 | 0.55719 | 2.05233 | 2.05233 | 0.682558 | 2.05233 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950103 | 2.199058 | 2.017852 | 2.199058 | 2.017852 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326492 | 2.309625 | 1.218259 | 1.624953 | 1.218259 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.577373 | 0.67706 | 1.378704 | 0.67706 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280046 | 0.377616 | 0.377616 | 0.343056 | 0.377616 |
| Igor Thiago | FWD | 4 | 0.047 | 0.493777 | 1.672109 | 0.365613 | 0.604877 | 0.365613 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.746596 | 0.310528 | 0.496238 | 0.310528 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
