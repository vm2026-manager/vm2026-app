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
- Price-quality formelmismatches foer: 838
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.494432
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1447.867
- Base-EV total efter: 1447.336
- Slut-EV total foer: 1448.077
- Slut-EV total efter: 1454.521
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260606_110808.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260606_110808.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 3.938417 | 3.938416 | 4.511834 | 4.511834 | 4.511834 | 4.511834 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.59947 | 4.59947 | 4.59947 | 4.59947 | 4.028472 | 4.028474 | 4.59947 | 4.342522 | 4.59947 | 4.342522 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344695 | 4.344695 | 4.344695 | 4.344695 | 3.938417 | 3.938416 | 4.344695 | 4.161869 | 4.344695 | 4.161869 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797086 | 3.797086 | 3.797086 | 3.797086 | 3.907067 | 3.907068 | 3.797086 | 3.846578 | 3.797086 | 3.846578 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442623 | 3.442623 | 3.442623 | 3.442623 | 3.999838 | 3.99984 | 3.442623 | 3.693371 | 3.442623 | 3.693371 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042964 | 3.042964 | 3.042964 | 3.042964 | 3.369898 | 3.3699 | 3.042964 | 3.190085 | 3.042964 | 3.190085 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697805 | 3.697805 | 3.697805 | 3.697805 | 2.505892 | 2.503328 | 3.161444 | 3.16029 | 3.161444 | 3.16029 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766439 | 2.766439 | 2.766439 | 2.766439 | 3.369898 | 3.3699 | 2.766439 | 3.037996 | 2.766439 | 3.037996 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047048 | 3.047048 | 3.047048 | 3.047048 | 2.743457 | 2.743456 | 3.047048 | 2.910432 | 3.047048 | 2.910432 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702047 | 2.702047 | 2.702047 | 2.702047 | 3.10557 | 3.089557 | 2.702047 | 2.876427 | 2.702047 | 2.876427 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787772 | 1.787772 | 1.787772 | 1.787772 | 2.808773 | 2.805862 | 1.787772 | 2.245913 | 1.787772 | 2.245913 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086726 | 2.086726 | 2.086726 | 2.086726 | 2.08286 | 2.082859 | 2.086726 | 2.084986 | 2.086726 | 2.084986 |
| Raphinha | BRA | 0.86 | 0.86 | 0.557184 | 0.557184 | 0.557184 | 0.557184 | 3.878145 | 3.878147 | 0.557184 | 2.051617 | 0.557184 | 2.051617 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.9501 | 1.9501 | 1.9501 | 1.9501 | 2.102802 | 2.10065 | 1.9501 | 2.017848 | 1.9501 | 2.017848 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326489 | 1.326489 | 1.326489 | 1.326489 | 1.091866 | 1.086098 | 1.220909 | 1.218313 | 1.220909 | 1.218313 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.129108 | 0.129108 | 0.677109 | 0.677109 | 0.677109 | 0.677109 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.531552 | 0.531552 | 0.531552 | 0.531552 | 0.497131 | 0.496867 | 0.531552 | 0.515944 | 0.531552 | 0.515944 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.493775 | 0.493775 | 0.493775 | 0.493775 | 0.208962 | 0.208962 | 0.493775 | 0.365609 | 0.493775 | 0.365609 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.190165 | 0.190165 | 0.308375 | 0.308375 | 0.308375 | 0.308375 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.530548 | 0 | 0.530548 | 0 | 0.0 | 0 | 0.530548 | 0 | 0.530548 | 0 |
