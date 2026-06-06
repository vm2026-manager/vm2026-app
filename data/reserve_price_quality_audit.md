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
| cheap_real_starter | 173 | 1.727392 | 1.727392 | 1.630458 | 1.727392 | 0 |
| likely_starter | 91 | 3.045791 | 3.045791 | 2.93841 | 3.045791 | 0 |
| other | 707 | 1.442246 | 1.131901 | 1.195161 | 1.065847 | -26.098104 |
| premium | 27 | 3.272992 | 3.133254 | 3.027978 | 3.071366 | -6.160307 |
| reserve_start_lt_0_10 | 246 | 0.661439 | 0.152978 | 0.253865 | 0.152883 | -76.886264 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.741755 | 1.477866 | 1.574963 | 1.4652 | -15.877976 |
| FWD | 274 | 1.555043 | 1.232162 | 1.212203 | 1.148059 | -26.17186 |
| GK | 152 | 1.344652 | 0.905033 | 0.977158 | 0.903767 | -32.788008 |
| MID | 410 | 1.2333 | 1.025229 | 1.013154 | 0.976471 | -20.824523 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.511835 | 4.511835 | 4.511835 | 4.511835 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.599476 | 4.342916 | 4.342916 | 4.342916 | 4.342916 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344697 | 4.161871 | 4.161871 | 4.161871 | 4.161871 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797088 | 3.908051 | 3.846962 | 3.908051 | 3.846962 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442625 | 3.693751 | 3.693751 | 3.693751 | 3.693751 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042964 | 3.190209 | 3.190209 | 3.190209 | 3.190209 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697808 | 3.160295 | 3.160295 | 3.160295 | 3.160295 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.76644 | 3.038121 | 3.038121 | 3.038121 | 3.038121 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.04705 | 2.910434 | 2.910434 | 2.910434 | 2.910434 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702047 | 2.914378 | 2.876427 | 2.914378 | 2.876427 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787776 | 2.245918 | 2.245918 | 2.190026 | 2.245918 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086728 | 2.919988 | 2.084988 | 2.556242 | 2.084988 |
| Raphinha | FWD | 6.5 | 0.86 | 0.557185 | 2.051948 | 2.051948 | 0.682552 | 2.051948 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950101 | 2.199056 | 2.01785 | 2.199056 | 2.01785 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.32649 | 2.309805 | 1.218314 | 1.62495 | 1.218314 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.577805 | 0.677086 | 1.378704 | 0.677086 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280042 | 0.377614 | 0.377614 | 0.343051 | 0.377614 |
| Igor Thiago | FWD | 4 | 0.047 | 0.493776 | 1.672085 | 0.365611 | 0.604876 | 0.365611 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.709206 | 0.308375 | 0.496238 | 0.308375 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
