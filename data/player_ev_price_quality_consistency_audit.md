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
- Price-quality formelmismatches foer: 884
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.495220
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1440.821
- Base-EV total efter: 1440.290
- Slut-EV total foer: 1443.279
- Slut-EV total efter: 1454.013
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260607_123003.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260607_123003.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9186, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.86 | 0.86 | 4.978706 | 4.978706 | 4.978706 | 4.978706 | 3.93929 | 3.963655 | 4.510969 | 4.521933 | 4.510969 | 4.521933 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.599483 | 4.599483 | 4.599483 | 4.599483 | 4.030554 | 3.992864 | 4.599483 | 4.326504 | 4.599483 | 4.326504 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344702 | 4.344702 | 4.344702 | 4.344702 | 3.93929 | 3.963655 | 4.344702 | 4.173231 | 4.344702 | 4.173231 |
| Kylian Mbappe | FRA | 0.6759 | 0.6759 | 3.797092 | 3.797092 | 3.797092 | 3.797092 | 3.905642 | 3.869108 | 3.797092 | 3.829499 | 3.797092 | 3.829499 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442635 | 3.442635 | 3.442635 | 3.442635 | 4.001859 | 3.96446 | 3.442635 | 3.677456 | 3.442635 | 3.677456 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697816 | 3.697816 | 3.697816 | 3.697816 | 2.589368 | 2.545882 | 3.697816 | 3.179445 | 3.697816 | 3.179445 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.04297 | 3.04297 | 3.04297 | 3.04297 | 3.370563 | 3.339573 | 3.04297 | 3.176441 | 3.04297 | 3.176441 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.76645 | 2.76645 | 2.76645 | 2.76645 | 3.370563 | 3.339573 | 2.76645 | 3.024355 | 2.76645 | 3.024355 |
| Alexander Schlager | AUT | 0.9212 | 0.9212 | 3.047606 | 3.047606 | 3.047606 | 3.047606 | 2.744057 | 2.760247 | 2.911009 | 2.918295 | 2.911009 | 2.918295 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702052 | 2.702052 | 2.702052 | 2.702052 | 3.105383 | 3.16988 | 2.702052 | 2.912575 | 2.702052 | 2.912575 |
| Kerim Alajbegovic | BIH | 0.8345 | 0.8345 | 2.611772 | 2.611772 | 2.611772 | 2.611772 | 2.17285 | 2.545882 | 2.611772 | 2.582121 | 2.611772 | 2.582121 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787782 | 1.787782 | 1.787782 | 1.787782 | 2.903525 | 2.854163 | 1.787782 | 2.267654 | 1.787782 | 2.267654 |
| Raphinha | BRA | 0.86 | 0.86 | 0.557194 | 0.557194 | 0.557194 | 0.557194 | 3.879904 | 3.843743 | 0.557194 | 2.036141 | 0.557194 | 2.036141 |
| Manuel Neuer | GER | 0.2645 | 0.2645 | 1.654787 | 1.654787 | 1.654787 | 1.654787 | 2.077694 | 1.497695 | 1.654787 | 1.584096 | 1.654787 | 1.584096 |
| Antonio Rüdiger | GER | 0.208 | 0.208 | 1.295756 | 1.295756 | 1.295756 | 1.295756 | 1.093257 | 1.071774 | 1.295756 | 1.194964 | 1.295756 | 1.194964 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.128986 | 0.127872 | 0.677054 | 0.676553 | 0.677054 | 0.676553 |
| Igor Thiago | BRA | 0.2955 | 0.2955 | 0.493783 | 0.493783 | 0.493783 | 0.493783 | 0.208969 | 0.740674 | 0.493783 | 0.604884 | 0.493783 | 0.604884 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280051 | 0.280051 | 0.280051 | 0.280051 | 0.505824 | 0.5013 | 0.280051 | 0.379613 | 0.280051 | 0.379613 |
| Joan Garcia | ESP | 0.0404 | 0.0404 | 0.405548 | 0.405548 | 0.405548 | 0.405548 | 0.190679 | 0.191835 | 0.308857 | 0.309377 | 0.308857 | 0.309377 |
| Christoph Baumgartner | AUT | 0.9186 | 0 | 0.530115 | 0 | 0.530115 | 0 | 0.0 | 0 | 0.530115 | 0 | 0.530115 | 0 |
