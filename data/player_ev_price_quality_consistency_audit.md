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
- Price-quality formelmismatches foer: 851
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.704463
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1427.904
- Base-EV total efter: 1427.374
- Slut-EV total foer: 1427.372
- Slut-EV total efter: 1439.829
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260605_182823.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260605_182823.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 3.919789 | 3.919792 | 4.503452 | 4.503453 | 4.503452 | 4.503453 |
| Harry Kane | ENG | 0.92 | 0.92 | 4.340567 | 4.340567 | 4.340567 | 4.340567 | 4.031632 | 4.031634 | 4.340567 | 4.201547 | 4.340567 | 4.201547 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344691 | 4.344691 | 4.344691 | 4.344691 | 3.919789 | 3.919792 | 4.153485 | 4.153487 | 4.153485 | 4.153487 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797083 | 3.797083 | 3.797083 | 3.797083 | 3.910166 | 3.910168 | 3.797083 | 3.847971 | 3.797083 | 3.847971 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442615 | 3.442615 | 3.442615 | 3.442615 | 4.002905 | 4.002906 | 3.442615 | 3.694746 | 3.442615 | 3.694746 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.04296 | 3.04296 | 3.04296 | 3.04296 | 3.370901 | 3.370902 | 3.04296 | 3.190534 | 3.04296 | 3.190534 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766432 | 2.766432 | 2.766432 | 2.766432 | 3.370901 | 3.370902 | 2.766432 | 3.038444 | 2.766432 | 3.038444 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047048 | 3.047048 | 3.047048 | 3.047048 | 2.730691 | 2.730693 | 3.047048 | 2.904688 | 3.047048 | 2.904688 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702042 | 2.702042 | 2.702042 | 2.702042 | 3.06411 | 3.064112 | 2.702042 | 2.864973 | 2.702042 | 2.864973 |
| Antonio Nusa | NOR | 0.6563 | 0.6563 | 2.966811 | 2.966811 | 2.966811 | 2.966811 | 2.347385 | 2.347385 | 2.966811 | 2.688069 | 2.966811 | 2.688069 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787764 | 1.787764 | 1.787764 | 1.787764 | 2.806744 | 2.806744 | 1.787764 | 2.246305 | 1.787764 | 2.246305 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086723 | 2.086723 | 2.086723 | 2.086723 | 2.073008 | 2.07301 | 2.086723 | 2.080552 | 2.086723 | 2.080552 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950097 | 1.950097 | 1.950097 | 1.950097 | 2.100951 | 2.100951 | 1.950097 | 2.017981 | 1.950097 | 2.017981 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326487 | 1.326487 | 1.326487 | 1.326487 | 1.076931 | 1.076932 | 1.326487 | 1.214187 | 1.326487 | 1.214187 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.128923 | 0.128923 | 0.677025 | 0.677025 | 0.677025 | 0.677025 |
| Raphinha | BRA | 0.6781 | 0.6781 | 0.488147 | 0.488147 | 0.488147 | 0.488147 | 0.732217 | 0.73222 | 0.488147 | 0.59798 | 0.488147 | 0.59798 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280038 | 0.280038 | 0.280038 | 0.280038 | 0.493758 | 0.493759 | 0.280038 | 0.376212 | 0.280038 | 0.376212 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.482279 | 0.482279 | 0.482279 | 0.482279 | 0.208972 | 0.208972 | 0.482279 | 0.359291 | 0.482279 | 0.359291 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.189272 | 0.189273 | 0.307973 | 0.307973 | 0.307973 | 0.307973 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.530548 | 0 | 0.530548 | 0 | 0.0 | 0 | 0.530548 | 0 | 0.530548 | 0 |
