# Reserve-safe price-quality audit

## Metode

- Rå variant: eksisterende pris-/positionsbaserede price-quality-signal.
- Variant 1: rå price-quality skaleres med `min(1, start_prob / 0.70)`.
- Variant 2: rå price-quality cappes ved `max(0.15, 1.50 * base_ev)`.
- Valgt metode: sandsynlige startere (`start_prob >= 0.70`) beholder rå value; øvrige appearance-skaleres og base-cappes. 55/45-formlen er uændret.

## Reserver med start_prob < 0.10

| Variant | Antal | optimizer_ev > 1.00 | base_ev > 0.50 |
| --- | ---: | ---: | ---: |
| Rå | 250 | 69 | 14 |
| Appearance-skaleret | 250 | 0 | 14 |
| Base-cappet | 250 | 2 | 14 |
| Valgt hybrid | 250 | 0 | 14 |

## Kohorter

| cohort | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| cheap_real_starter | 180 | 1.783864 | 1.783864 | 1.681672 | 1.783864 | 0 |
| likely_starter | 93 | 3.004355 | 3.004355 | 2.888851 | 3.004355 | 0 |
| other | 694 | 1.444176 | 1.120817 | 1.182445 | 1.054017 | -27.016076 |
| premium | 27 | 3.30627 | 3.179583 | 3.062794 | 3.119799 | -5.639919 |
| reserve_start_lt_0_10 | 250 | 0.660718 | 0.15126 | 0.252793 | 0.151162 | -77.121614 |

## Positioner

| position | players | mean_raw_final | mean_appearance_final | mean_base_cap_final | mean_selected_final | selected_vs_raw_pct |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 408 | 1.764619 | 1.489854 | 1.579915 | 1.476959 | -16.301537 |
| FWD | 274 | 1.549613 | 1.235029 | 1.212763 | 1.155241 | -25.449675 |
| GK | 152 | 1.351147 | 0.906686 | 0.979034 | 0.905392 | -32.990891 |
| MID | 410 | 1.237256 | 1.019357 | 1.005266 | 0.968923 | -21.687815 |

## Sanity

| player_name | position | price_m | start_prob | base_ev | raw_optimizer_ev | appearance_scaled_optimizer_ev | base_capped_optimizer_ev | selected_optimizer_ev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | GK | 5 | 0.86 | 4.978706 | 4.521933 | 4.521933 | 4.521933 | 4.521933 |
| Harry Kane | FWD | 9.5 | 0.97 | 4.599483 | 4.326504 | 4.326504 | 4.326504 | 4.326504 |
| Mike Maignan | GK | 5 | 0.9407 | 4.344702 | 4.173231 | 4.173231 | 4.173231 | 4.173231 |
| Kylian Mbappe | FWD | 10 | 0.6759 | 3.797092 | 3.89158 | 3.829499 | 3.89158 | 3.829499 |
| Erling Haaland | FWD | 8.5 | 0.9058 | 3.442635 | 3.677456 | 3.677456 | 3.677456 | 3.677456 |
| Antonio Nusa | MID | 3.5 | 0.82 | 3.697816 | 3.179446 | 3.179446 | 3.179446 | 3.179446 |
| Donyell Malen | FWD | 4.5 | 0.7971 | 3.04297 | 3.176441 | 3.176441 | 3.176441 | 3.176441 |
| Alexander Sørloth | FWD | 4.5 | 0.8805 | 2.76645 | 3.024355 | 3.024355 | 3.024355 | 3.024355 |
| Alexander Schlager | GK | 3.5 | 0.9212 | 3.047606 | 2.918294 | 2.918294 | 2.918294 | 2.918294 |
| Jules Kounde | DEF | 3.5 | 0.6814 | 2.702052 | 2.951512 | 2.912575 | 2.951512 | 2.912575 |
| Kerim Alajbegovic | MID | 3.5 | 0.8345 | 2.611772 | 2.582122 | 2.582122 | 2.582122 | 2.582122 |
| Martin Ødegaard | MID | 4.5 | 0.8841 | 1.787782 | 2.267653 | 2.267653 | 2.190033 | 2.267653 |
| Raphinha | FWD | 6.5 | 0.86 | 0.557194 | 2.036141 | 2.036141 | 0.682563 | 2.036141 |
| Manuel Neuer | GK | 5 | 0.2645 | 1.654787 | 2.693778 | 1.584096 | 2.027114 | 1.584096 |
| Antonio Rüdiger | DEF | 4.5 | 0.208 | 1.295756 | 2.335784 | 1.194964 | 1.587301 | 1.194964 |
| Victor Munoz | FWD | 3 | 0.0424 | 1.125473 | 1.569006 | 0.676553 | 1.378704 | 0.676553 |
| Igor Thiago | FWD | 4 | 0.2955 | 0.493783 | 1.659352 | 0.857419 | 0.604884 | 0.604884 |
| Yousef Qashi | MID | 2 | 0.8559 | 0.280051 | 0.379613 | 0.379613 | 0.343062 | 0.379613 |
| Joan Garcia | GK | 4 | 0.0404 | 0.405548 | 1.718792 | 0.309377 | 0.496796 | 0.309377 |
| Christoph Baumgartner | MID | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 |
