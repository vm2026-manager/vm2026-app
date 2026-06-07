# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 500 | 217 | 187 |
| Appearance-skaleret | 500 | 144 | 187 |
| Base-cappet | 500 | 161 | 187 |
| Valgt hybrid | 500 | 144 | 187 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 185 | 10.477663 | 10.477663 | 9.768837 | 10.477663 | 0 |
| likely_starter | 97 | 15.150794 | 15.150794 | 14.346019 | 15.150794 | 0 |
| other | 710 | 9.81815 | 7.955713 | 8.887824 | 7.656789 | -22.013938 |
| premium | 27 | 19.398352 | 18.761734 | 18.594671 | 18.499209 | -4.635151 |
| reserve_start_lt_0_10 | 500 | 2.786411 | 1.268234 | 2.071566 | 1.266371 | -54.55188 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 502 | 8.387258 | 7.123531 | 7.608109 | 6.945066 | -17.195038 |
| FWD | 318 | 8.353978 | 6.926388 | 7.528803 | 6.722364 | -19.530991 |
| GK | 185 | 6.493364 | 4.599628 | 5.387145 | 4.592304 | -29.27696 |
| MID | 514 | 8.224905 | 6.94105 | 7.464865 | 6.815697 | -17.133433 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.86 | 71.521132 | 47.687076 | 47.687076 | 47.687076 | 47.687076 |
| Victor Munoz | FWD | 3 | 0.0424 | 68.634525 | 42.665292 | 38.046776 | 42.665292 | 38.046776 |
| Mike Maignan | GK | 5 | 0.9407 | 17.250604 | 17.838286 | 17.838286 | 17.838286 | 17.838286 |
| Kylian Mbappe | FWD | 10 | 0.6759 | 16.918035 | 17.810524 | 17.517688 | 17.810524 | 17.517688 |
| Raphinha | FWD | 6.5 | 0.86 | 14.236737 | 16.024286 | 16.024286 | 16.024286 | 16.024286 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 15.94523 | 16.002295 | 16.002295 | 16.002295 | 16.002295 |
| Harry Kane | FWD | 9.5 | 0.97 | 13.020765 | 15.639936 | 15.639936 | 15.639936 | 15.639936 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 14.11902 | 15.550132 | 15.343281 | 15.550132 | 15.343281 |
| Alexander Schlager | GK | 3.5 | 0.9212 | 13.732106 | 13.593864 | 13.593864 | 13.593864 | 13.593864 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 8.129447 | 12.895533 | 12.895533 | 9.958572 | 12.895533 |
| Manuel Neuer | GK | 5 | 0.2771 | 16.33198 | 17.333043 | 12.288176 | 17.333043 | 12.288176 |
| Antonio Nusa | MID | 3.5 | 0.82 | 8.685508 | 11.938769 | 11.938769 | 10.639747 | 11.938769 |
| Joan Garcia | GK | 4 | 0.0404 | 19.738036 | 17.980476 | 11.267109 | 17.980476 | 11.267109 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 6.136805 | 11.239292 | 11.239292 | 7.517586 | 11.239292 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 7.218725 | 11.202717 | 11.202717 | 8.842938 | 11.202717 |
| Antonio Rüdiger | DEF | 4.5 | 0.208 | 15.338368 | 17.005278 | 10.982372 | 17.005278 | 10.982372 |
| Igor Thiago | FWD | 4 | 0.2955 | 12.93362 | 13.858306 | 9.960767 | 13.858306 | 9.960767 |
| Kerim Alajbegovic | MID | 3.5 | 0.8345 | 4.309422 | 9.531921 | 9.531921 | 5.279042 | 9.531921 |
| Yousef Qashi | MID | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
