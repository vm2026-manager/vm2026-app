# Clean Sheet Component Repair Audit

Clean sheet-komponenten er genberegnet for GK/DEF fra `fixture_strength_multipliers.csv` clean sheet-probability og en 60-minutters eligibility-proxy: `start_prob * min(match_n_minutes_if_start / 60, 1)`.

MID/FWD holdes paa 0, fordi den eksplicitte Holdet clean sheet-regel her kun bruges for GK/DEF.

## Foer/efter

- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 foer: 0
- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 efter: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob foer: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob efter: 0
- Clean sheet repair rows: 42
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260605_230052.csv`

## Sanity-spillere

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after | match_1_weighted_before | match_1_weighted_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.9407 | SEN | 0.4624 | 0.43498 | 0.43498 | 1.442054 | 1.442054 |
| Gregor Kobel | SUI | GK | 0.9174 | QAT | 0.5585 | 0.268415 | 0.512368 | 0.89393 | 1.668806 |
| Alexander Schlager | AUT | GK | 0.921 | JOR | 0.4865 | 0.448067 | 0.448067 | 1.4956 | 1.4956 |
| Jules Kounde | FRA | DEF | 0.6814 | SEN | 0.4624 | 0.315079 | 0.315079 | 0.907738 | 0.907738 |
| Philipp Lienhart | AUT | DEF | 0.6578 | JOR | 0.4865 | 0.32002 | 0.32002 | 1.377354 | 1.377354 |
| Stefan Posch | AUT | DEF | 0.636 | JOR | 0.4865 | 0.309414 | 0.309414 | 1.294632 | 1.294632 |
| Manuel Neuer | GER | GK | 0.3702 | CUW | 0.6881 | 0.254735 | 0.254735 | 0.990382 | 0.990382 |
| Maximilian Wöber | AUT | DEF | 0.5612 | JOR | 0.4865 | 0.273024 | 0.273024 | 0.903613 | 0.903613 |
| Kevin Danso | AUT | DEF | 0.3362 | JOR | 0.4865 | 0.163561 | 0.163561 | 0.620027 | 0.620027 |
| Patrick Pentz | AUT | GK | 0.054 | JOR | 0.4865 | 0.026271 | 0.026271 | 0.217369 | 0.217369 |

## Schlager vs Pentz vs Posch vs Lienhart

| player_name | position | start_prob | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- |
| Alexander Schlager | GK | 0.921 | 0.4865 | 0.448067 | 0.448067 |
| Philipp Lienhart | DEF | 0.6578 | 0.4865 | 0.32002 | 0.32002 |
| Stefan Posch | DEF | 0.636 | 0.4865 | 0.309414 | 0.309414 |
| Patrick Pentz | GK | 0.054 | 0.4865 | 0.026271 | 0.026271 |

## Maignan vs Schlager

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.9407 | SEN | 0.4624 | 0.43498 | 0.43498 |
| Alexander Schlager | AUT | GK | 0.921 | JOR | 0.4865 | 0.448067 | 0.448067 |
