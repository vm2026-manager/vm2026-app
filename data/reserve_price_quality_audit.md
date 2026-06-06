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
| cheap_real_starter | 173 | 1.740014 | 1.740014 | 1.656512 | 1.740014 | 0 |
| likely_starter | 92 | 3.054949 | 3.054949 | 2.958317 | 3.054949 | 0 |
| other | 706 | 1.440899 | 1.131197 | 1.194145 | 1.065028 | -26.085865 |
| premium | 27 | 3.272737 | 3.133021 | 3.027816 | 3.07118 | -6.158668 |
| reserve_start_lt_0_10 | 246 | 0.661558 | 0.152987 | 0.253864 | 0.152892 | -76.889129 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.741754 | 1.477865 | 1.574961 | 1.465198 | -15.877987 |
| FWD | 274 | 1.561168 | 1.240975 | 1.223779 | 1.156824 | -25.900058 |
| GK | 152 | 1.344651 | 0.905032 | 0.977157 | 0.903767 | -32.788016 |
| MID | 410 | 1.238236 | 1.030165 | 1.023371 | 0.981408 | -20.741466 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.511834 | 4.511834 | 4.511834 | 4.511834 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.59947 | 4.342522 | 4.342522 | 4.342522 | 4.342522 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344695 | 4.161869 | 4.161869 | 4.161869 | 4.161869 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797086 | 3.907653 | 3.846578 | 3.907653 | 3.846578 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442623 | 3.693371 | 3.693371 | 3.693371 | 3.693371 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042964 | 3.190085 | 3.190085 | 3.190085 | 3.190085 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697805 | 3.16029 | 3.16029 | 3.16029 | 3.16029 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766439 | 3.037996 | 3.037996 | 3.037996 | 3.037996 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047048 | 2.910432 | 2.910432 | 2.910432 | 2.910432 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702047 | 2.914377 | 2.876426 | 2.914377 | 2.876426 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787772 | 2.245912 | 2.245912 | 2.190021 | 2.245912 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086726 | 2.919987 | 2.084986 | 2.556239 | 2.084986 |
| Raphinha | FWD | 6.5 | 0.86 | 0.557184 | 2.051617 | 2.051617 | 0.68255 | 2.051617 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.9501 | 2.199053 | 2.017848 | 2.199053 | 2.017848 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326489 | 2.309804 | 1.218313 | 1.624949 | 1.218313 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.578182 | 0.677109 | 1.378704 | 0.677109 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.531552 | 0.515944 | 0.515944 | 0.515944 | 0.515944 |
| Igor Thiago | FWD | 4 | 0.047 | 0.493775 | 1.672065 | 0.365609 | 0.604875 | 0.365609 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.709205 | 0.308375 | 0.496238 | 0.308375 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
