# Player EV Price-Quality Consistency Audit

Price-quality-laget er genberegnet efter start_prob- og komponentrebuild. Optimizer, strategi-output og frontend er ikke genkoert.

## Rodarsag

`repair_ev_components_after_start_prob_repair.py` opdaterede matchkomponenter, men price-quality metadata og slut-EV byggede stadig paa gamle `*_before_price_quality`-kolonner. `apply_price_quality_to_ev.py` var idempotent ved at foretraekke gamle basekolonner, hvilket efter komponentrebuild blev forkert.

## Haandtering

- Komplette komponentraekker: base-EV = `match_1_weighted_match_ev + match_2_weighted_match_ev + match_3_weighted_match_ev`.
- Aggregate-only-raekker: eksisterende aggregerede EV bevares som base; der opfindes ikke basekomponenter.
- No-EV-source-raekker: base, price-quality og final EV holdes paa 0.
- Christoph Baumgartner: markeret ude og nulstillet som valgbaar modelkandidat.
- Price-quality-formlen er uændret: `0.55 * model_ev_before_price_quality + 0.45 * price_quality_ev`.

## Counts

- Komplette komponentraekker efter: 1189
- Aggregate-only-raekker efter: 0
- No-EV-source-raekker efter: 54
- Out-of-tournament efter: 1
- Price-quality formelmismatches foer: 859
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.704861
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1441.311
- Base-EV total efter: 1440.780
- Slut-EV total foer: 1442.852
- Slut-EV total efter: 1450.096
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260606_120530.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260606_120530.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 3.938417 | 3.938418 | 4.511834 | 4.511835 | 4.511834 | 4.511835 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.599476 | 4.599476 | 4.599476 | 4.599476 | 4.029341 | 4.029342 | 4.599476 | 4.342916 | 4.599476 | 4.342916 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344697 | 4.344697 | 4.344697 | 4.344697 | 3.938417 | 3.938418 | 4.344697 | 4.161872 | 4.344697 | 4.161872 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797088 | 3.797088 | 3.797088 | 3.797088 | 3.907919 | 3.90792 | 3.797088 | 3.846962 | 3.797088 | 3.846962 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442625 | 3.442625 | 3.442625 | 3.442625 | 4.000682 | 4.000683 | 3.442625 | 3.693751 | 3.442625 | 3.693751 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042964 | 3.042964 | 3.042964 | 3.042964 | 3.370175 | 3.370176 | 3.190209 | 3.190209 | 3.190209 | 3.190209 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697808 | 3.697808 | 3.697808 | 3.697808 | 2.503329 | 2.503334 | 3.697808 | 3.160295 | 3.697808 | 3.160295 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.76644 | 2.76644 | 2.76644 | 2.76644 | 3.370175 | 3.370176 | 2.76644 | 3.038121 | 2.76644 | 3.038121 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.04705 | 3.04705 | 3.04705 | 3.04705 | 2.743457 | 2.743458 | 3.04705 | 2.910434 | 3.04705 | 2.910434 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702047 | 2.702047 | 2.702047 | 2.702047 | 3.089558 | 3.089559 | 2.702047 | 2.876427 | 2.702047 | 2.876427 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787776 | 1.787776 | 1.787776 | 1.787776 | 2.805863 | 2.805869 | 1.787776 | 2.245918 | 1.787776 | 2.245918 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086728 | 2.086728 | 2.086728 | 2.086728 | 2.08286 | 2.082861 | 2.086728 | 2.084988 | 2.086728 | 2.084988 |
| Raphinha | BRA | 0.86 | 0.86 | 0.557185 | 0.557185 | 0.557185 | 0.557185 | 3.878879 | 3.87888 | 2.051947 | 2.051948 | 2.051947 | 2.051948 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950101 | 1.950101 | 1.950101 | 1.950101 | 2.100651 | 2.100655 | 1.950101 | 2.01785 | 1.950101 | 2.01785 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.32649 | 1.32649 | 1.32649 | 1.32649 | 1.086098 | 1.086099 | 1.218314 | 1.218314 | 1.218314 | 1.218314 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.129057 | 0.129057 | 0.677086 | 0.677086 | 0.677086 | 0.677086 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280042 | 0.280042 | 0.280042 | 0.280042 | 0.496867 | 0.496868 | 0.377613 | 0.377614 | 0.377613 | 0.377614 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.493776 | 0.493776 | 0.493776 | 0.493776 | 0.208965 | 0.208965 | 0.365611 | 0.365611 | 0.365611 | 0.365611 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.190165 | 0.190166 | 0.308375 | 0.308375 | 0.308375 | 0.308375 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.530551 | 0 | 0.530551 | 0 | 0.0 | 0 | 0.530551 | 0 | 0.530551 | 0 |
