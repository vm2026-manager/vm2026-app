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
- Price-quality formelmismatches foer: 1104
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.626621
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1429.450
- Base-EV total efter: 1428.919
- Slut-EV total foer: 1432.109
- Slut-EV total efter: 1443.063
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260605_173342.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260605_173342.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.9192, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 3.188321 | 3.919791 | 4.980994 | 4.503453 | 4.980994 | 4.503453 |
| Harry Kane | ENG | 0.92 | 0.92 | 4.307649 | 4.307649 | 4.307649 | 4.307649 | 3.966195 | 4.093073 | 4.307649 | 4.21109 | 4.307649 | 4.21109 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344689 | 4.344689 | 4.344689 | 4.344689 | 3.188321 | 3.919791 | 4.344689 | 4.153485 | 4.344689 | 4.153485 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.79708 | 3.79708 | 3.79708 | 3.79708 | 3.98031 | 3.96977 | 3.79708 | 3.87479 | 3.79708 | 3.87479 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442623 | 3.442623 | 3.442623 | 3.442623 | 3.937965 | 4.063878 | 3.442623 | 3.722188 | 3.442623 | 3.722188 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042958 | 3.042958 | 3.042958 | 3.042958 | 2.266396 | 3.421595 | 3.042958 | 3.213344 | 3.042958 | 3.213344 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766437 | 2.766437 | 2.766437 | 2.766437 | 3.316909 | 3.421595 | 2.766437 | 3.061258 | 2.766437 | 3.061258 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047047 | 3.047047 | 3.047047 | 3.047047 | 2.272056 | 2.73069 | 3.047047 | 2.904686 | 3.047047 | 2.904686 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.70204 | 2.70204 | 2.70204 | 2.70204 | 3.505191 | 3.06411 | 2.70204 | 2.864971 | 2.70204 | 2.864971 |
| Antonio Nusa | NOR | 0.6563 | 0.6563 | 2.966819 | 2.966819 | 2.966819 | 2.966819 | 2.627366 | 2.347384 | 2.966819 | 2.688073 | 2.966819 | 2.688073 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787772 | 1.787772 | 1.787772 | 1.787772 | 2.503732 | 2.806743 | 1.787772 | 2.246309 | 1.787772 | 2.246309 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.08672 | 2.08672 | 2.08672 | 2.08672 | 1.082207 | 2.073009 | 2.08672 | 2.08055 | 2.08672 | 2.08055 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950095 | 1.950095 | 1.950095 | 1.950095 | 1.735261 | 2.10095 | 1.950095 | 2.01798 | 1.950095 | 2.01798 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326483 | 1.326483 | 1.326483 | 1.326483 | 3.840246 | 1.076932 | 1.326483 | 1.214185 | 1.326483 | 1.214185 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125472 | 1.125472 | 1.125472 | 1.125472 | 0.126956 | 0.130769 | 1.125472 | 0.677856 | 1.125472 | 0.677856 |
| Raphinha | BRA | 0.6781 | 0.6781 | 0.485575 | 0.485575 | 0.485575 | 0.485575 | 0.728361 | 0.728363 | 0.485575 | 0.594829 | 0.485575 | 0.594829 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280035 | 0.280035 | 0.280035 | 0.280035 | 0.420054 | 0.493755 | 0.280035 | 0.376209 | 0.280035 | 0.376209 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.47973 | 0.47973 | 0.47973 | 0.47973 | 0.205648 | 0.212094 | 0.47973 | 0.359294 | 0.47973 | 0.359294 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.127794 | 0.189272 | 0.405092 | 0.307973 | 0.405092 | 0.307973 |
| Christoph Baumgartner | AUT | 0.9192 | 0 | 0.530546 | 0 | 0.530546 | 0 | 0.0 | 0 | 0.530546 | 0 | 0.530546 | 0 |
