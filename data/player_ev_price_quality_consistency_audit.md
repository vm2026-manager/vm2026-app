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
- Price-quality formelmismatches foer: 835
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.704462
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1435.248
- Base-EV total efter: 1434.717
- Slut-EV total foer: 1436.457
- Slut-EV total efter: 1450.078
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260605_230053.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260605_230053.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 3.919794 | 3.938417 | 4.503454 | 4.511835 | 4.503454 | 4.511835 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.59947 | 4.59947 | 4.59947 | 4.59947 | 4.031634 | 4.031636 | 4.59947 | 4.343944 | 4.59947 | 4.343944 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344694 | 4.344694 | 4.344694 | 4.344694 | 3.919794 | 3.938417 | 4.153489 | 4.16187 | 4.153489 | 4.16187 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797084 | 3.797084 | 3.797084 | 3.797084 | 3.910168 | 3.910169 | 3.797084 | 3.847972 | 3.797084 | 3.847972 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442622 | 3.442622 | 3.442622 | 3.442622 | 4.002906 | 4.002908 | 3.442622 | 3.694751 | 3.442622 | 3.694751 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042961 | 3.042961 | 3.042961 | 3.042961 | 3.370902 | 3.370904 | 3.042961 | 3.190535 | 3.042961 | 3.190535 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697805 | 3.697805 | 3.697805 | 3.697805 | 2.347387 | 2.503689 | 3.697805 | 3.160453 | 3.697805 | 3.160453 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766437 | 2.766437 | 2.766437 | 2.766437 | 3.370902 | 3.370904 | 2.766437 | 3.038447 | 2.766437 | 3.038447 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047049 | 3.047049 | 3.047049 | 3.047049 | 2.730694 | 2.743457 | 2.904689 | 2.910432 | 2.904689 | 2.910432 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702044 | 2.702044 | 2.702044 | 2.702044 | 3.064114 | 3.105569 | 2.702044 | 2.88363 | 2.702044 | 2.88363 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.78777 | 1.78777 | 1.78777 | 1.78777 | 2.806746 | 2.806746 | 1.78777 | 2.246309 | 1.78777 | 2.246309 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086725 | 2.086725 | 2.086725 | 2.086725 | 2.073011 | 2.08286 | 2.086725 | 2.084986 | 2.086725 | 2.084986 |
| Raphinha | BRA | 0.86 | 0.86 | 0.544547 | 0.544547 | 0.544547 | 0.544547 | 0.73222 | 3.880816 | 0.544547 | 2.045868 | 0.544547 | 2.045868 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950099 | 1.950099 | 1.950099 | 1.950099 | 2.100953 | 2.100953 | 1.950099 | 2.017983 | 1.950099 | 2.017983 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326488 | 1.326488 | 1.326488 | 1.326488 | 1.076933 | 1.091866 | 1.326488 | 1.220908 | 1.326488 | 1.220908 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.128923 | 0.128923 | 0.677025 | 0.677025 | 0.677025 | 0.677025 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280039 | 0.280039 | 0.280039 | 0.280039 | 0.49376 | 0.49376 | 0.280039 | 0.376213 | 0.280039 | 0.376213 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.482282 | 0.482282 | 0.482282 | 0.482282 | 0.208972 | 0.208973 | 0.482282 | 0.359293 | 0.482282 | 0.359293 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.189273 | 0.190165 | 0.307973 | 0.308375 | 0.307973 | 0.308375 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.53055 | 0 | 0.53055 | 0 | 0.0 | 0 | 0.53055 | 0 | 0.53055 | 0 |
