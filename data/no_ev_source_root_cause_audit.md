# No-EV Source Root-Cause Audit

## Scope

Denne read-only audit analyserer de 103 spillere fra `no_ev_source_recovery_audit.csv`. Den skelner mellem en egentlig historisk spiller-EV-kilde og senere team-/appearance-afledte komponenter. Ingen EV-, player-pool-, Transfermarkt-, start-, optimizer- eller strategifiler er ?ndret eller genk?rt.

Alle 103 spillere har tre gyldige gruppefixtures og eksisterende team-/kampkontekst. Fixturelaget er derfor ikke den upstream rod?rsag.

## Rod?rsager

- `ev_master_row_missing_player_stat_allocation`: 75
- `missing_from_ev_master_at_holdet_rebase`: 12
- `historical_aggregate_only_missing_match_components`: 12
- `historical_components_lost_in_component_rebuild`: 4

### Pr?cis datak?de

- `rebase_player_ev_to_holdet_master.py` kan kun kopiere EV og komponenter fra en allerede matchet EV-r?kke. Ved `unmatched` oprettes en Holdet-r?kke med startchance/minute-share, men uden spiller-EV-kilde.
- `repair_player_ev_fixture_data_gaps.py` kan restaurere eksisterende aggregate EV og fordele den over fixtures, men dokumenterer selv, at den ikke opfinder ny spiller-EV.
- `build_player_ev_group_stage.py` multiplicerer eksisterende goal/assist/clean-sheet-komponenter. Nul eller tom input forbliver nul.
- `repair_ev_components_after_start_prob_repair.py` kan bygge team-, appearance- og clean-sheet-komponenter fra fixturekontekst. Den kan ikke skabe spillerens goal/assist/SOT-rater, n?r `goal_share_norm`, `assist_share_norm` og `sot_share_norm` mangler.

## Reparationsklasser

- Deterministisk fuld komponentrestaurering: 4 spillere.
- Deterministisk aggregate-restaurering, men komponentmetode mangler: 12 spillere.
- Kr?ver ny generel spiller-EV-metode: 75 spillere.
- Kr?ver f?rst identitetskontrol, derefter generel metode: 12 spillere.
- Har siden f?et team-/appearance-afledt komponentbase: 93 spillere; dette er ikke bevis p? en oprindelig spillerstatkilde.
- St?r fortsat med `no_ev_source` i nuv?rende EV-fil: 10 spillere.

Ingen af de 103 b?r permanent st? uden EV alene p? grund af manglende fixtures eller holdkontekst. De 12 rebase-unmatched spillere b?r dog ikke repareres automatisk, f?r identiteten mod den oprindelige EV-kilde er afklaret.

## Deterministisk reparerbare

| player_name | team_id | position | historical_recovery_status | max_backup_aggregate_ev | repair_class |
| --- | --- | --- | --- | --- | --- |
| Micky van de Ven | NED | DEF | recoverable_from_backup_components | 0.8492 | deterministic_full |
| Oston Urunov | UZB | MID | recoverable_aggregate_only | 0.5201 | deterministic_aggregate_only |
| Nikola Katic | BIH | DEF | recoverable_aggregate_only | 0.772 | deterministic_aggregate_only |
| Victor Lindelöf | SWE | DEF | recoverable_aggregate_only | 0.6436 | deterministic_aggregate_only |
| Nathan Ake | NED | DEF | recoverable_from_backup_components | 0.7781 | deterministic_full |
| Daniel Svensson | SWE | DEF | recoverable_aggregate_only | 1.1426 | deterministic_aggregate_only |
| Jurrien Timber | NED | DEF | recoverable_from_backup_components | 0.3967 | deterministic_full |
| Gustaf Lagerbielke | SWE | DEF | recoverable_aggregate_only | 0.6856 | deterministic_aggregate_only |
| Hyeon-Woo Jo | KOR | GK | recoverable_aggregate_only | 0.513 | deterministic_aggregate_only |
| Carl Starfelt | SWE | DEF | recoverable_aggregate_only | 0.3929 | deterministic_aggregate_only |
| Nihad Mujakic | BIH | DEF | recoverable_aggregate_only | 0.6185 | deterministic_aggregate_only |
| Fredrik Andre Bjørkan | NOR | DEF | recoverable_aggregate_only | 0.2343 | deterministic_aggregate_only |
| Frans Dhia Putros | IRQ | DEF | recoverable_from_backup_components | 0.038 | deterministic_full |
| Mark Flekken | NED | GK | recoverable_aggregate_only | 0.6446 | deterministic_aggregate_only |
| Rocky Bushiri | COD | DEF | recoverable_aggregate_only | 0.1243 | deterministic_aggregate_only |
| Bum-Keun Song | KOR | GK | recoverable_aggregate_only | 0.1179 | deterministic_aggregate_only |

## Kr?ver identitetskontrol

| player_name | team_id | position | price | start_prob | ev_match_method |
| --- | --- | --- | --- | --- | --- |
| Victor Munoz | ESP | FWD | 3000000 | 0.873 | unmatched |
| Joan Garcia | ESP | GK | 4000000 | 0.8655 | unmatched |
| Ibrahim Sangare | CIV | MID | 3000000 | 0.7036 | unmatched |
| Oumar Diakite | CIV | FWD | 2500000 | 0.6966 | unmatched |
| Yerry Mina | COL | DEF | 3000000 | 0.6156 | unmatched |
| Eric Garcia | ESP | DEF | 3000000 | 0.5243 | unmatched |
| Juan 'Cucho' Hernandez | COL | FWD | 4000000 | 0.25 | unmatched |
| Pablo Gavi | ESP | MID | 4000000 | 0.25 | unmatched |
| Carlos Andres Gomez | COL | MID | 2000000 | 0.4417 | unmatched |
| Ange-Yoan Bonny | CIV | FWD | 2500000 | 0.25 | unmatched |
| Willer Ditta | COL | DEF | 2500000 | 0.3311 | unmatched |
| Marc Pubill | ESP | DEF | 3000000 | 0.25 | unmatched |

## Top 25 gaps

Prioriteringen kombinerer pris, startchance og positionsbaseret fantasy-relevans.

| player_name | team_id | position | price | start_prob | root_cause | recommended_repair_method |
| --- | --- | --- | --- | --- | --- | --- |
| Neymar Jr. | BRA | FWD | 5500000 | 0.8919 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Lucas Paqueta | BRA | MID | 5500000 | 0.7869 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Romelu Lukaku | BEL | FWD | 4500000 | 0.7965 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Manuel Neuer | GER | GK | 5000000 | 0.8454 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Mateo Kovacic | CRO | MID | 4000000 | 0.839 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Bruno Guimaraes | BRA | MID | 4500000 | 0.7749 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Victor Munoz | ESP | FWD | 3000000 | 0.873 | missing_from_ev_master_at_holdet_rebase | verify canonical identity against the pre-rebase source, then generate EV with the new general player-EV method |
| Bradley Barcola | FRA | MID | 4000000 | 0.7583 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Joan Garcia | ESP | GK | 4000000 | 0.8655 | missing_from_ev_master_at_holdet_rebase | verify canonical identity against the pre-rebase source, then generate EV with the new general player-EV method |
| Ben Doak | SCO | MID | 3000000 | 0.8529 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| In-beom Hwang | KOR | MID | 2500000 | 0.897 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Matthew Garbett | NZL | MID | 2500000 | 0.8803 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Maxence Lacroix | FRA | DEF | 3000000 | 0.873 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Ahmed Zizo | EGY | MID | 3000000 | 0.7866 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Hamza Abdelkarim | EGY | FWD | 2000000 | 0.8655 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Micky van de Ven | NED | DEF | 4000000 | 0.7361 | historical_components_lost_in_component_rebuild | restore validated match components from named backup, then rerun existing consistency calculations |
| Mehdi Ghayedi | IRN | MID | 3000000 | 0.7519 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Jules Kounde | FRA | DEF | 3500000 | 0.7661 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Lachlan Bayliss | NZL | MID | 2000000 | 0.8655 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Mostafa Ziko | EGY | MID | 2000000 | 0.8655 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Pape Alassane Gueye | SEN | MID | 3000000 | 0.7287 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Assane Diao Diaoune | SEN | FWD | 2500000 | 0.7493 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Oston Urunov | UZB | MID | 2500000 | 0.7918 | historical_aggregate_only_missing_match_components | restore validated aggregate EV; rebuild match components only through a separate general component-generation method |
| Issa Diop | MAR | DEF | 2500000 | 0.8568 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |
| Fredrik Aursnes | NOR | MID | 3000000 | 0.7078 | ev_master_row_missing_player_stat_allocation | generate player shares/rates from a new general position-and-usage EV method; do not infer them from team context alone |

## Anbefalet n?ste kodeopgave

Implement?r ?n afgr?nset, generel `player_component_source`-generator for r?kker uden historiske shares. Den skal tage position, start/appearance, team-position-baselines og fixturekontekst som input, skrive eksplicit kilde/proveniens og f?rst k?re i audit-mode. Hold backup-restaurering og de 12 rebase-unmatched identitetskontroller uden for denne opgave.
