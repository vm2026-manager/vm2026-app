# Clean Sheet Component Repair Audit

Clean sheet-komponenten er genberegnet for GK/DEF fra `fixture_strength_multipliers.csv` clean sheet-probability og en 60-minutters eligibility-proxy: `start_prob * min(match_n_minutes_if_start / 60, 1)`.

MID/FWD holdes paa 0, fordi den eksplicitte Holdet clean sheet-regel her kun bruges for GK/DEF.

## Foer/efter

- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 foer: 0
- GK/DEF med clean_sheet_prob > 0 og start_prob >= 0.70 men clean_sheet_ev = 0 efter: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob foer: 0
- GK/DEF med NaN clean_sheet_ev trods clean_sheet_prob efter: 0
- Clean sheet repair rows: 537
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260607_123003.csv`

## Sanity-spillere

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after | match_1_weighted_before | match_1_weighted_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.9407 | SEN | 0.4624 | 0.43498 | 0.43498 | 1.442053 | 1.442053 |
| Gregor Kobel | SUI | GK | 0.9192 | QAT | 0.5585 | 0.512535 | 0.513373 | 1.669274 | 1.671619 |
| Philipp Lienhart | AUT | DEF | 0.84 | JOR | 0.4865 | 0.40866 | 0.40866 | 1.705033 | 1.705033 |
| Stefan Posch | AUT | DEF | 0.84 | JOR | 0.4865 | 0.40866 | 0.40866 | 1.647953 | 1.647953 |
| Alexander Schlager | AUT | GK | 0.9212 | JOR | 0.4865 | 0.448164 | 0.448164 | 1.49587 | 1.49587 |
| Jules Kounde | FRA | DEF | 0.6814 | SEN | 0.4624 | 0.315079 | 0.315079 | 0.907738 | 0.907738 |
| Manuel Neuer | GER | GK | 0.2645 | CUW | 0.6881 | 0.254047 | 0.182002 | 0.988455 | 0.786729 |
| Maximilian Wöber | AUT | DEF | 0.559 | JOR | 0.4865 | 0.271954 | 0.271954 | 0.901258 | 0.901258 |
| Kevin Danso | AUT | DEF | 0.3346 | JOR | 0.4865 | 0.162783 | 0.162783 | 0.618315 | 0.618315 |
| Patrick Pentz | AUT | GK | 0.0538 | JOR | 0.4865 | 0.026174 | 0.026174 | 0.217097 | 0.217097 |

## Schlager vs Pentz vs Posch vs Lienhart

| player_name | position | start_prob | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- |
| Philipp Lienhart | DEF | 0.84 | 0.4865 | 0.40866 | 0.40866 |
| Stefan Posch | DEF | 0.84 | 0.4865 | 0.40866 | 0.40866 |
| Alexander Schlager | GK | 0.9212 | 0.4865 | 0.448164 | 0.448164 |
| Patrick Pentz | GK | 0.0538 | 0.4865 | 0.026174 | 0.026174 |

## Maignan vs Schlager

| player_name | team_id | position | start_prob | match_1_opponent | match_1_clean_sheet_prob | match_1_clean_sheet_ev_before | match_1_clean_sheet_ev_after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | GK | 0.9407 | SEN | 0.4624 | 0.43498 | 0.43498 |
| Alexander Schlager | AUT | GK | 0.9212 | JOR | 0.4865 | 0.448164 | 0.448164 |
