# Player Pool vs EV Start Probability Audit

Autoritativ synkronisering af EV-filens startfelter fra player pool ved exact `player_id`. Komponenter genbygges i næste pipeline-trin.

## Rodårsag

Repairen anvendte `source_priority()` og opdaterede kun, når poolkilden havde højere prioritet end EV-kilden. Nye og gamle Transfermarkt-kilder lå i samme bucket, fald blev ofte afvist, og `count_serious()` talte kun store positive differencer. Derfor kunne scriptet rapportere 0 alvorlige mismatches, selv om over 1.000 exact-ID-rækker var ude af sync.

## Kildeprioritet efter rettelse

1. Player pool er autoritativ for alle exact-player_id matches.
2. Context-overrides bevares, fordi de allerede er indarbejdet i player pool før sync.
3. Rækker uden exact player_id-match blokeres og rapporteres; der bruges ikke fuzzy overskrivning.

## Mismatch counts

- Start_prob mismatches > 0.001 før: 8
- Start_prob mismatches > 0.001 efter: 0
- Start_prob_source mismatches før: 0
- Start_prob_source mismatches efter: 0
- Blokerede identitetsmatches: 0
- team_minute_rank før/efter: 0 / 0
- holdet_official_unmatched_default før/efter: 0 / 0
- name+team før/efter: 0 / 0
- Rækker påvirket: 1244
- Backup: `data\player_ev_group_stage_v1.backup_before_start_prob_source_repair_20260606_213231.csv`

## Sanity-spillere

| player_name | team_id | old_start_prob | new_start_prob | old_source | new_source | minute_share | goal_ev_changed | round_ev_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unai Simon | ESP | 0.8605 | 0.8605 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.078227 | no | no |
| Harry Kane | ENG | 0.97 | 0.97 | transfermarkt_availability_split_2026_06_06+context_override | transfermarkt_availability_split_2026_06_06+context_override | 0.088182 | no | no |
| Mike Maignan | FRA | 0.9407 | 0.9407 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.085518 | no | no |
| Erling Haaland | NOR | 0.9058 | 0.9058 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.082345 | no | no |
| Thibaut Courtois | BEL | 0.8305 | 0.8305 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.0755 | no | no |
| Antonio Nusa | NOR | 0.82 | 0.82 | transfermarkt_availability_split_2026_06_06+context_override | transfermarkt_availability_split_2026_06_06+context_override | 0.074545 | no | no |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.080045 | no | no |
| Alexander Schlager | AUT | 0.921 | 0.921 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.083727 | no | no |
| Jules Kounde | FRA | 0.6814 | 0.6814 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.061945 | no | no |
| Vladimir Coufal | CZE | 0.9098 | 0.9098 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.082709 | no | no |
| Ladislav Krejci | CZE | 0.9051 | 0.9051 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.082282 | no | no |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.080373 | no | no |
| Manuel Neuer | GER | 0.3702 | 0.3702 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.033655 | no | no |
| Raphinha | BRA | 0.86 | 0.86 | transfermarkt_availability_split_2026_06_06+context_override | transfermarkt_availability_split_2026_06_06+context_override | 0.078182 | no | no |
| Antonio Rüdiger | GER | 0.2165 | 0.2165 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.019682 | no | no |
| Victor Munoz | ESP | 0.0424 | 0.0424 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.003855 | no | no |
| Yousef Qashi | JOR | 0.8559 | 0.8559 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.077809 | no | no |
| Igor Thiago | BRA | 0.047 | 0.047 | transfermarkt_availability_split_2026_06_06 | transfermarkt_availability_split_2026_06_06 | 0.004273 | no | no |
| Joan Garcia | ESP | 0.0403 | 0.0403 | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | transfermarkt_availability_split_2026_06_06+gk_hierarchy_normalized | 0.003664 | no | no |

## Noter

- EV-komponenter som `match_1_goal_ev` og `match_1_weighted_match_ev` blev ikke ændret af denne repair.
- `minute_share` blev kun synkroniseret til den nye dokumenterede startchance for de rækker, hvor startsignalet blev promoveret.
- Optimizer og strategi-output blev ikke genkørt.
