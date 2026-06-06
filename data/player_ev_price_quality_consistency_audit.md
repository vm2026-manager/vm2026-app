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
- Price-quality formelmismatches foer: 795
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.704859
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1445.307
- Base-EV total efter: 1444.777
- Slut-EV total foer: 1447.869
- Slut-EV total efter: 1454.169
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260606_144929.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260606_144929.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 4.038201 | 4.038206 | 4.556737 | 4.556739 | 4.556737 | 4.556739 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.599477 | 4.599477 | 4.599477 | 4.599477 | 4.03034 | 4.03034 | 4.343365 | 4.343365 | 4.343365 | 4.343365 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344699 | 4.344699 | 4.344699 | 4.344699 | 4.038201 | 4.038206 | 4.344699 | 4.206777 | 4.344699 | 4.206777 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797089 | 3.797089 | 3.797089 | 3.797089 | 3.908899 | 3.908898 | 3.797089 | 3.847403 | 3.797089 | 3.847403 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442629 | 3.442629 | 3.442629 | 3.442629 | 4.001651 | 4.001651 | 3.442629 | 3.694189 | 3.442629 | 3.694189 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042966 | 3.042966 | 3.042966 | 3.042966 | 3.370493 | 3.370493 | 3.190353 | 3.190353 | 3.190353 | 3.190353 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697811 | 3.697811 | 3.697811 | 3.697811 | 2.503335 | 2.503337 | 3.697811 | 3.160298 | 3.697811 | 3.160298 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766444 | 2.766444 | 2.766444 | 2.766444 | 3.370493 | 3.370493 | 2.766444 | 3.038266 | 2.766444 | 3.038266 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047052 | 3.047052 | 3.047052 | 3.047052 | 2.811835 | 2.811839 | 2.941205 | 2.941206 | 2.941205 | 2.941206 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702048 | 2.702048 | 2.702048 | 2.702048 | 3.089358 | 3.089362 | 2.702048 | 2.876339 | 2.702048 | 2.876339 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787778 | 1.787778 | 1.787778 | 1.787778 | 2.80587 | 2.805872 | 1.787778 | 2.24592 | 1.787778 | 2.24592 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086729 | 2.086729 | 2.086729 | 2.086729 | 2.135631 | 2.135634 | 2.086729 | 2.108736 | 2.086729 | 2.108736 |
| Raphinha | BRA | 0.86 | 0.86 | 0.55719 | 0.55719 | 0.55719 | 0.55719 | 3.879723 | 3.879723 | 0.55719 | 2.05233 | 0.55719 | 2.05233 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950103 | 1.950103 | 1.950103 | 1.950103 | 2.100655 | 2.100657 | 1.950103 | 2.017852 | 1.950103 | 2.017852 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326492 | 1.326492 | 1.326492 | 1.326492 | 1.085972 | 1.085974 | 1.326492 | 1.218259 | 1.326492 | 1.218259 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.128999 | 0.128999 | 0.67706 | 0.67706 | 0.67706 | 0.67706 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280046 | 0.280046 | 0.280046 | 0.280046 | 0.496867 | 0.496869 | 0.280046 | 0.377616 | 0.280046 | 0.377616 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.493777 | 0.493777 | 0.493777 | 0.493777 | 0.208968 | 0.208968 | 0.493777 | 0.365613 | 0.493777 | 0.365613 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.194949 | 0.194949 | 0.310528 | 0.310528 | 0.310528 | 0.310528 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.530553 | 0 | 0.530553 | 0 | 0.0 | 0 | 0.530553 | 0 | 0.530553 | 0 |
