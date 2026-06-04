# Player Pool vs EV Start Probability Audit

Audit og målrettet repair af EV-filens startfelter. Goal EV, weighted match EV og strategi-output er ikke genberegnet.

## Rodårsag

`tools/rebase_player_ev_to_holdet_master.py` brugte EV-rækkens eksisterende `start_prob` som førstevalg og player-pool signalet som fallback. Dermed kunne legacy/fallback-kilder som `team_minute_rank`, `name+team` og `holdet_official_unmatched_default` overskrive nyere dokumenterede Transfermarkt-/manual-/lineup-signaler.

## Kildeprioritet efter rettelse

1. confirmed_lineup / expected_lineup / manual / transfermarkt_availability_split / context_override
2. andre dokumenterede ikke-fallback-kilder
3. team_minute_rank / name+team / holdet_official_unmatched_default / legacy / fallback

## Mismatch counts

- Alvorlige mismatches før: 1
- Alvorlige mismatches efter: 0
- team_minute_rank før/efter: 0 / 0
- holdet_official_unmatched_default før/efter: 0 / 0
- name+team før/efter: 0 / 0
- Rækker påvirket: 32
- Backup: `data\player_ev_group_stage_v1.backup_before_start_prob_source_repair_20260604_223321.csv`

## Sanity-spillere

| player_name | team_id | old_start_prob | new_start_prob | old_source | new_source | minute_share | goal_ev_changed | round_ev_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | 0.8702 | 0.8702 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.079109 | no | no |
| Erling Haaland | NOR | 0.8883 | 0.8883 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.080755 | no | no |
| Antonio Nusa | NOR | 0.8157 | 0.8157 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.074155 | no | no |
| Alexander Sørloth | NOR | 0.8078 | 0.8078 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.073436 | no | no |
| Jules Kounde | FRA | 0.7661 | 0.7661 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.069645 | no | no |
| Manuel Neuer | GER | 0.5761 | 0.7293 | transfermarkt_availability_split_2026_06_04+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_04+gk_hierarchy_normalized | 0.0663 | no | no |
| Martin Ødegaard | NOR | 0.8375 | 0.8375 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.076136 | no | no |
| Thibaut Courtois | BEL | 0.4432 | 0.493 | transfermarkt_availability_split_2026_06_04+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_04+gk_hierarchy_normalized | 0.044818 | no | no |
| Ladislav Krejci | CZE | 0.8635 | 0.8635 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.0785 | no | no |
| Raphinha | BRA | 0.8812 | 0.8812 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.080109 | no | no |
| Vladimir Coufal | CZE | 0.7537 | 0.7537 | transfermarkt_availability_split_2026_06_01 | transfermarkt_availability_split_2026_06_01 | 0.068518 | no | no |

## Noter

- EV-komponenter som `match_1_goal_ev` og `match_1_weighted_match_ev` blev ikke ændret af denne repair.
- `minute_share` blev kun synkroniseret til den nye dokumenterede startchance for de rækker, hvor startsignalet blev promoveret.
- Optimizer og strategi-output blev ikke genkørt.
