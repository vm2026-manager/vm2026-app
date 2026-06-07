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

- Komplette komponentraekker efter: 1235
- Aggregate-only-raekker efter: 0
- No-EV-source-raekker efter: 12
- Out-of-tournament efter: 272
- Price-quality formelmismatches foer: 1175
- Price-quality formelmismatches efter: 0
- Maks formelafvigelse foer: 32.263916
- Maks formelafvigelse efter: 0.000001
- Base-EV total foer: 11112.212
- Base-EV total efter: 11112.212
- Slut-EV total foer: 11120.596
- Slut-EV total efter: 9976.979
- EV-backup: `data\player_ev_group_stage_v1.backup_before_price_quality_consistency_20260607_165337.csv`
- Player-pool-backup: `data\player_pool_v1.backup_before_baumgartner_out_20260607_165337.json`

## Christoph Baumgartner

- Foer: holdet_is_out=True, start_prob=0.0, EV=0.0
- Efter: holdet_is_out=True, start_prob=0.0, EV=0.0

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | component_sum_before | component_sum_after | base_before | base_after | price_quality_before | price_quality_after | final_before | final_after | optimizer_before | optimizer_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.86 | 0.86 | 71.521132 | 71.521132 | 71.521132 | 71.521132 | 4.084942 | 18.556564 | 71.521132 | 47.687076 | 71.521132 | 47.687076 |
| Victor Munoz | ESP | 0.0424 | 0.0424 | 68.634525 | 68.634525 | 68.634525 | 68.634525 | 0.144517 | 0.66175 | 68.634525 | 38.046776 | 68.634525 | 38.046776 |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 17.250604 | 17.250604 | 17.250604 | 17.250604 | 4.084942 | 18.556564 | 17.250604 | 17.838286 | 17.250604 | 17.838286 |
| Kylian Mbappe | FRA | 0.6759 | 0.6759 | 16.918035 | 16.918035 | 16.918035 | 16.918035 | 4.558137 | 18.250598 | 16.918035 | 17.517688 | 16.918035 | 17.517688 |
| Raphinha | BRA | 0.86 | 0.86 | 14.236737 | 14.236737 | 14.236737 | 14.236737 | 4.515701 | 18.209068 | 14.236737 | 16.024286 | 14.236737 | 16.024286 |
| Donyell Malen | NED | 0.7971 | 0.7971 | 15.94523 | 15.94523 | 15.94523 | 15.94523 | 3.882995 | 16.072041 | 15.94523 | 16.002295 | 15.94523 | 16.002295 |
| Harry Kane | ENG | 0.97 | 0.97 | 13.020765 | 13.020765 | 13.020765 | 13.020765 | 4.70284 | 18.841146 | 13.020765 | 15.639937 | 13.020765 | 15.639937 |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 14.11902 | 14.11902 | 14.11902 | 14.11902 | 3.247556 | 16.839601 | 14.11902 | 15.343282 | 14.11902 | 15.343282 |
| Alexander Schlager | AUT | 0.9212 | 0.9212 | 13.732106 | 13.732106 | 13.732106 | 13.732106 | 2.739621 | 13.424901 | 13.732106 | 13.593864 | 13.732106 | 13.593864 |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 8.129447 | 8.129447 | 8.129447 | 8.129447 | 4.667194 | 18.72075 | 8.129447 | 12.895534 | 8.129447 | 12.895534 |
| Manuel Neuer | GER | 0.2771 | 0.2771 | 16.33198 | 16.33198 | 16.33198 | 16.33198 | 1.617053 | 7.345748 | 16.33198 | 12.288176 | 16.33198 | 12.288176 |
| Antonio Nusa | NOR | 0.82 | 0.82 | 8.685508 | 8.685508 | 8.685508 | 8.685508 | 2.946328 | 15.914976 | 8.685508 | 11.938769 | 8.685508 | 11.938769 |
| Joan Garcia | ESP | 0.0404 | 0.0404 | 19.738036 | 19.738036 | 19.738036 | 19.738036 | 0.194026 | 0.913753 | 19.738036 | 11.267108 | 19.738036 | 11.267108 |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 6.136805 | 6.136805 | 6.136805 | 6.136805 | 3.315816 | 17.475665 | 6.136805 | 11.239292 | 6.136805 | 11.239292 |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 7.218725 | 7.218725 | 7.218725 | 7.218725 | 3.882995 | 16.072041 | 7.218725 | 11.202717 | 7.218725 | 11.202717 |
| Antonio Rüdiger | GER | 0.208 | 0.208 | 15.338368 | 15.338368 | 15.338368 | 15.338368 | 1.097161 | 5.658376 | 15.338368 | 10.982372 | 15.338368 | 10.982372 |
| Igor Thiago | BRA | 0.2955 | 0.2955 | 12.93362 | 12.93362 | 12.93362 | 12.93362 | 0.740682 | 6.327279 | 12.93362 | 9.960767 | 12.93362 | 9.960767 |
| Kerim Alajbegovic | BIH | 0.8345 | 0.8345 | 4.309422 | 4.309422 | 4.309422 | 4.309422 | 2.946328 | 15.914976 | 4.309422 | 9.531921 | 4.309422 | 9.531921 |
| Yousef Qashi | JOR | 0.0 | 0.0 | 0.280055 | 0.280055 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| Christoph Baumgartner | AUT | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
