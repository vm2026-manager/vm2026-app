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

- Komplette komponentraekker efter: 1196
- Aggregate-only-raekker efter: 0
- No-EV-source-raekker efter: 47
- Out-of-tournament efter: 1
- Price-quality formelmismatches foer: 951
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 1.788514
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 1638.617
- Base-EV total efter: 1638.617
- Slut-EV total foer: 1736.218
- Slut-EV total efter: 2013.405
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260604_223409.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260604_223409.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.8952, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kylian Mbappe | FRA | 0.902 | 0.902 | 4.791598 | 4.791598 | 4.791598 | 4.791598 | 4.632679 | 4.632679 | 4.791598 | 4.720084 | 4.791598 | 4.720084 |
| Harry Kane | ENG | 0.8702 | 0.8702 | 4.066753 | 4.066753 | 4.066753 | 4.066753 | 4.615973 | 4.615973 | 4.066753 | 4.313902 | 4.066753 | 4.313902 |
| Erling Haaland | NOR | 0.8883 | 0.8883 | 3.373354 | 3.373354 | 3.373354 | 3.373354 | 4.582563 | 4.582563 | 3.373354 | 3.917498 | 3.373354 | 3.917498 |
| Manuel Neuer | GER | 0.7293 | 0.7293 | 3.554182 | 3.554182 | 3.554182 | 3.554182 | 2.854694 | 2.896965 | 3.554182 | 3.258434 | 3.554182 | 3.258434 |
| Antonio Nusa | NOR | 0.8157 | 0.8157 | 3.572754 | 3.572754 | 3.572754 | 3.572754 | 2.865202 | 2.865204 | 3.572754 | 3.254356 | 3.572754 | 3.254356 |
| Donyell Malen | NED | 0.7099 | 0.7099 | 2.694472 | 2.694472 | 2.694472 | 2.694472 | 3.847531 | 3.847531 | 3.213349 | 3.213349 | 3.213349 | 3.213349 |
| Alexander Sørloth | NOR | 0.8078 | 0.8078 | 2.526252 | 2.526252 | 2.526252 | 2.526252 | 3.847531 | 3.847531 | 2.526252 | 3.120828 | 2.526252 | 3.120828 |
| Jules Kounde | FRA | 0.7661 | 0.7661 | 2.848916 | 2.848916 | 2.848916 | 2.848916 | 3.112957 | 3.112963 | 2.848916 | 2.967737 | 2.848916 | 2.967737 |
| Kerim Alajbegovic | BIH | 0.873 | 0.873 | 2.714846 | 2.714846 | 2.714846 | 2.714846 | 2.865202 | 2.865204 | 2.714846 | 2.782507 | 2.714846 | 2.782507 |
| Martin Ødegaard | NOR | 0.8375 | 0.8375 | 1.686029 | 1.686029 | 1.686029 | 1.686029 | 3.222167 | 3.222169 | 1.686029 | 2.377292 | 1.686029 | 2.377292 |
| Raphinha | BRA | 0.8812 | 0.8812 | 0.485569 | 0.485569 | 0.485569 | 0.485569 | 4.440568 | 4.440568 | 0.485569 | 2.265319 | 0.485569 | 2.265319 |
| Christoph Baumgartner | AUT | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
