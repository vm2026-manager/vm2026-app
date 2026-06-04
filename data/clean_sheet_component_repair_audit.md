# Clean Sheet Component Repair Audit

Clean sheet-komponenten er genberegnet for GK/DEF fra `fixture_strength_multipliers.csv` clean sheet-probability og en 60-minutters eligibility-proxy: `start_prob * min(match_n_minutes_if_start / 60, 1)`.

MID/FWD holdes paa 0, fordi den eksplicitte Holdet clean sheet-regel her kun bruges for GK/DEF.

## Foer/efter

- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 foer: 0
- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 efter: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob foer: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob efter: 0
- Clean sheet repair rows: 105
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260604_223354.csv`

## Sanity-spillere

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after | match_1_weighted_before | match_1_weighted_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.8865 | SEN | 0.4624 | 0.379168 | 0.409918 | 1.264306 | 1.348641 |
| Manuel Neuer | GER | GK | 0.7293 | CUW | 0.6881 | 0.396414 | 0.501831 | 1.387083 | 1.682251 |
| Jules Kounde | FRA | DEF | 0.7661 | SEN | 0.4624 | 0.354245 | 0.354245 | 0.965315 | 0.965314 |
| Stefan Posch | AUT | DEF | 0.7042 | JOR | 0.4865 | 0.342593 | 0.342593 | 1.410595 | 1.410596 |
| Philipp Lienhart | AUT | DEF | 0.6814 | JOR | 0.4865 | 0.331501 | 0.331501 | 1.408012 | 1.408012 |
| Alexander Schlager | AUT | GK | 0.789 | JOR | 0.4865 | 0.43785 | 0.383848 | 1.441849 | 1.293483 |
| Gregor Kobel | SUI | GK | 0.5051 | QAT | 0.5585 | 0.222841 | 0.282098 | 0.768548 | 0.931573 |
| Kevin Danso | AUT | DEF | 0.7136 | JOR | 0.4865 | 0.347166 | 0.347166 | 1.11503 | 1.115031 |
| Maximilian Wöber | AUT | DEF | 0.6391 | JOR | 0.4865 | 0.310922 | 0.310922 | 1.007502 | 1.007502 |
| Patrick Pentz | AUT | GK | 0.1599 | JOR | 0.4865 | 0.077791 | 0.077791 | 0.35883 | 0.35883 |

## Schlager vs Pentz vs Posch vs Lienhart

| player_name | position | start_prob | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- |
| Stefan Posch | DEF | 0.7042 | 0.4865 | 0.342593 | 0.342593 |
| Philipp Lienhart | DEF | 0.6814 | 0.4865 | 0.331501 | 0.331501 |
| Alexander Schlager | GK | 0.789 | 0.4865 | 0.43785 | 0.383848 |
| Patrick Pentz | GK | 0.1599 | 0.4865 | 0.077791 | 0.077791 |

## Maignan vs Schlager

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.8865 | SEN | 0.4624 | 0.379168 | 0.409918 |
| Alexander Schlager | AUT | GK | 0.789 | JOR | 0.4865 | 0.43785 | 0.383848 |
