# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 246 | 71 | 12 |
| Appearance-skaleret | 246 | 0 | 12 |
| Base-cappet | 246 | 2 | 12 |
| Valgt hybrid | 246 | 0 | 12 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 175 | 1.75173 | 1.75173 | 1.65252 | 1.75173 | 0 |
| likely_starter | 96 | 3.082661 | 3.082661 | 2.979676 | 3.082661 | 0 |
| other | 700 | 1.433626 | 1.119759 | 1.180569 | 1.052084 | -26.613754 |
| premium | 27 | 3.317452 | 3.192296 | 3.071816 | 3.130351 | -5.639889 |
| reserve_start_lt_0_10 | 246 | 0.671317 | 0.153336 | 0.253229 | 0.153225 | -77.175478 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.755995 | 1.491735 | 1.585315 | 1.478444 | -15.805893 |
| FWD | 274 | 1.559301 | 1.238527 | 1.215843 | 1.15448 | -25.961692 |
| GK | 152 | 1.365124 | 0.912633 | 0.982188 | 0.911258 | -33.247242 |
| MID | 410 | 1.242801 | 1.032994 | 1.018609 | 0.98321 | -20.887581 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.8605 | 4.980994 | 4.556738 | 4.556738 | 4.556738 | 4.556738 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.599477 | 4.343365 | 4.343365 | 4.343365 | 4.343365 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344698 | 4.206776 | 4.206776 | 4.206776 | 4.206776 |
| Kylian Mbappe | FWD | 10 | 0.6765 | 3.797089 | 3.908507 | 3.847403 | 3.908507 | 3.847403 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442629 | 3.694189 | 3.694189 | 3.694189 | 3.694189 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.042967 | 3.190354 | 3.190354 | 3.190354 | 3.190354 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697813 | 3.171804 | 3.171804 | 3.171804 | 3.171804 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.766444 | 3.038267 | 3.038267 | 3.038267 | 3.038267 |
| Alexander Schlager | GK | 3.5 | 0.921 | 3.047053 | 2.941206 | 2.941206 | 2.941206 | 2.941206 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702048 | 2.92744 | 2.889142 | 2.92744 | 2.889142 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787779 | 2.25898 | 2.25898 | 2.190029 | 2.25898 |
| Manuel Neuer | GK | 5 | 0.3702 | 2.086732 | 2.964894 | 2.108737 | 2.556247 | 2.108737 |
| Raphinha | FWD | 6.5 | 0.86 | 0.55719 | 2.05233 | 2.05233 | 0.682558 | 2.05233 |
| Kerim Alajbegovic | MID | 3.5 | 0.5874 | 1.950104 | 2.210564 | 2.027508 | 2.210564 | 2.027508 |
| Antonio Rüdiger | DEF | 4.5 | 0.2165 | 1.326494 | 2.324536 | 1.222872 | 1.624955 | 1.222872 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.577373 | 0.67706 | 1.378704 | 0.67706 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280048 | 0.378814 | 0.378814 | 0.343059 | 0.378814 |
| Igor Thiago | FWD | 4 | 0.047 | 0.493778 | 1.67211 | 0.365614 | 0.604878 | 0.365614 |
| Joan Garcia | GK | 4 | 0.0403 | 0.405092 | 1.746596 | 0.310528 | 0.496238 | 0.310528 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
