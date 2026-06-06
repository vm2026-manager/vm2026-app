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
- Price-quality formelmismatches foer: 803
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.704859
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1450.069
- Base-EV total efter: 1450.069
- Slut-EV total foer: 1456.097
- Slut-EV total efter: 1461.160
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260606_213234.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260606_213234.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.0, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | 4.980994 | 4.980994 | 4.980994 | 4.980994 | 4.038206 | 4.038204 | 4.556739 | 4.556739 | 4.556739 | 4.556739 |
| Harry Kane | ENG | 0.97 | 0.97 | 4.599477 | 4.599477 | 4.599477 | 4.599477 | 4.03034 | 4.03034 | 4.599477 | 4.343365 | 4.599477 | 4.343365 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 4.344698 | 4.344698 | 4.344698 | 4.344698 | 4.038206 | 4.038204 | 4.344698 | 4.206776 | 4.344698 | 4.206776 |
| Kylian Mbappe | FRA | 0.6765 | 0.6765 | 3.797089 | 3.797089 | 3.797089 | 3.797089 | 3.908898 | 3.908898 | 3.797089 | 3.847403 | 3.797089 | 3.847403 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 3.442629 | 3.442629 | 3.442629 | 3.442629 | 4.001651 | 4.001651 | 3.442629 | 3.694189 | 3.442629 | 3.694189 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 3.042967 | 3.042967 | 3.042967 | 3.042967 | 3.370493 | 3.370494 | 3.042967 | 3.190354 | 3.042967 | 3.190354 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 3.697813 | 3.697813 | 3.697813 | 3.697813 | 2.503337 | 2.528904 | 3.697813 | 3.171804 | 3.697813 | 3.171804 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 2.766444 | 2.766444 | 2.766444 | 2.766444 | 3.370493 | 3.370494 | 2.766444 | 3.038266 | 2.766444 | 3.038266 |
| Alexander Schlager | AUT | 0.921 | 0.921 | 3.047053 | 3.047053 | 3.047053 | 3.047053 | 2.811839 | 2.811838 | 3.047053 | 2.941206 | 3.047053 | 2.941206 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 2.702048 | 2.702048 | 2.702048 | 2.702048 | 3.089362 | 3.117812 | 2.702048 | 2.889142 | 2.702048 | 2.889142 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 1.787779 | 1.787779 | 1.787779 | 1.787779 | 2.805872 | 2.834893 | 1.787779 | 2.25898 | 1.787779 | 2.25898 |
| Manuel Neuer | GER | 0.3702 | 0.3702 | 2.086732 | 2.086732 | 2.086732 | 2.086732 | 2.135634 | 2.135633 | 2.086732 | 2.108738 | 2.086732 | 2.108738 |
| Raphinha | BRA | 0.86 | 0.86 | 0.55719 | 0.55719 | 0.55719 | 0.55719 | 3.879723 | 3.879723 | 0.55719 | 2.05233 | 0.55719 | 2.05233 |
| Kerim Alajbegovic | BIH | 0.5874 | 0.5874 | 1.950104 | 1.950104 | 1.950104 | 1.950104 | 2.100657 | 2.122112 | 1.950104 | 2.027507 | 1.950104 | 2.027507 |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | 1.326494 | 1.326494 | 1.326494 | 1.326494 | 1.085974 | 1.096222 | 1.326494 | 1.222871 | 1.326494 | 1.222871 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 1.125473 | 1.125473 | 1.125473 | 1.125473 | 0.128999 | 0.128999 | 0.67706 | 0.67706 | 0.67706 | 0.67706 |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | 0.280048 | 0.280048 | 0.280048 | 0.280048 | 0.496869 | 0.499529 | 0.280048 | 0.378814 | 0.280048 | 0.378814 |
| Igor Thiago | BRA | 0.047 | 0.047 | 0.493778 | 0.493778 | 0.493778 | 0.493778 | 0.208968 | 0.208968 | 0.493778 | 0.365614 | 0.493778 | 0.365614 |
| Joan Garcia | ESP | 0.0403 | 0.0403 | 0.405092 | 0.405092 | 0.405092 | 0.405092 | 0.194949 | 0.194949 | 0.310528 | 0.310528 | 0.310528 | 0.310528 |
| Christoph Baumgartner | AUT | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
