# Player EV Component Dependency Audit

Audit og maalrettet rebuild efter start_prob-repair. Optimizer, strategi-output og frontend er ikke genkoert.

## Rodarsag

`start_prob` og `minute_share` blev repareret fra player pool, men eksisterende kampkomponenter blev liggende fra den gamle startbasis. Den gamle basis kan ses direkte som `match_n_start_minutes_ev / match_n_minutes_if_start`. For Haaland var basis ca. 0.163, mens ny dokumenteret `start_prob` er 0.8883.

## Rebuild-regel

- Kun spillere med eksisterende per-kamp-komponenter og udledelig gammel startbasis blev genberegnet.
- Startafhaengige komponenter blev skaleret med `ny_start_prob / gammel_komponent_startbasis`: goal, assist, shots_on_target, clean_sheet, card og on_pitch.
- `match_n_start_minutes_ev` blev sat til `start_prob * match_n_minutes_if_start`.
- `match_n_total_ev_next_match` og `match_n_weighted_match_ev` blev genberegnet med eksisterende pointformel fra `build_player_ev_group_stage.py`.
- `weighted_group_stage_ev` og `optimizer_ev` bevarer den eksisterende aggregerings-/price-quality-skala som multiplikator; price/value-laget er ikke kalibreret om.
- Spillere uden basekomponenter fik ikke opfundet maal/assist/SOT-komponenter.

## Counts

- Stale komponenter foer: 9
- Stale komponenter efter: 0
- Rækker genberegnet: 964
- Team/match/on_pitch high-start spreads > 0.05 foer: 0
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.031182
- Stoerste high-start on_pitch spread efter: 0.031182
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 1
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260604_223354.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | 0.8702 | 0.8702 | 0.213065 | 0.213065 | 57.891795 | 57.891795 | 1.343987 | 1.343987 | 4.313901 | 4.066753 | ok |
| Mike Maignan | FRA | 0.8865 | 0.8865 | 0.0 | 0 | 66.12316 | 71.485587 | 1.264306 | 1.348641 | 3.349581 | 4.015626 | ok |
| Antonio Nusa | NOR | 0.8157 | 0.8157 | 0.237187 | 0.237187 | 54.169821 | 54.169821 | 1.650005 | 1.650005 | 3.25435 | 3.572754 | ok |
| Manuel Neuer | GER | 0.7293 | 0.7293 |  |  |  |  | 1.387083 | 1.682251 | 2.895083 | 3.554182 | ok |
| Erling Haaland | NOR | 0.8883 | 0.8883 | 0.258297 | 0.258297 | 58.991115 | 58.991115 | 1.543012 | 1.543012 | 3.917492 | 3.373354 | ok |
| Jules Kounde | FRA | 0.7661 | 0.7661 |  |  |  |  | 0.965315 | 0.965314 | 2.967735 | 2.848916 | ok |
| Stefan Posch | AUT | 0.7042 | 0.7042 | 0.057918 | 0.057918 | 53.205831 | 53.205831 | 1.410595 | 1.410596 | 2.709993 | 2.694152 | ok |
| Philipp Lienhart | AUT | 0.6814 | 0.6814 | 0.071815 | 0.071815 | 51.35303 | 51.35303 | 1.408012 | 1.408012 | 2.699683 | 2.675404 | ok |
| Alexander Schlager | AUT | 0.789 | 0.789 | 0.0 | 0 | 71.5905 | 62.761005 | 1.441849 | 1.293483 | 2.550906 | 2.634172 | ok |
| Alexander Sørloth | NOR | 0.8078 | 0.8078 | 0.171648 | 0.171648 | 53.64519 | 53.64519 | 1.145683 | 1.145683 | 3.120824 | 2.526252 | ok |
| Gregor Kobel | SUI | 0.5051 | 0.5051 | 0.0 | 0 | 32.068827 | 40.596402 | 0.768548 | 0.931573 | 2.112957 | 2.222928 | ok |
| Kevin Danso | AUT | 0.7136 | 0.7136 | 0.026251 | 0.026251 | 53.063296 | 53.063296 | 1.11503 | 1.115031 | 2.073809 | 2.165154 | ok |
| Maximilian Wöber | AUT | 0.6391 | 0.6391 | 0.025212 | 0.025212 | 47.64171 | 47.64171 | 1.007502 | 1.007502 | 1.951163 | 1.942158 | ok |
| Martin Ødegaard | NOR | 0.8375 | 0.8375 | 0.07493 | 0.07493 | 58.967538 | 58.967538 | 0.777292 | 0.777292 | 2.377287 | 1.686029 | ok |
| Patrick Pentz | AUT | 0.1599 | 0.1599 | 0.0 | 0.0 | 12.806551 | 12.806551 | 0.35883 | 0.35883 | 1.11532 | 0.731649 | ok |
| Raphinha | BRA | 0.8812 | 0.8812 |  |  |  |  | 0.204084 | 0.204083 | 2.265319 | 0.485569 | ok |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Oscar Bobb | 0.9101 | 0.156927 | 0.070127 | -0.058307 | 0.061312 |
| Erling Haaland | 0.8883 | 0.150278 | 0.067155 | -0.055836 | 0.056596 |
| Martin Ødegaard | 0.8375 | 0.141684 | 0.063315 | -0.052643 | 0.0505 |
| Antonio Nusa | 0.8157 | 0.137996 | 0.061667 | -0.051273 | 0.047884 |
| Alexander Sørloth | 0.8078 | 0.13666 | 0.061069 | -0.050776 | 0.046936 |

## Saerlige rodarsager

- Erling Haaland: havde korrekte nye startfelter, men komponenterne var stadig baseret paa gammel `team_minute_rank`-basis ca. 0.163. Genberegnet fra eksisterende komponenter.
- Harry Kane: komponentbasis var gammel `name+team`-basis ca. 0.456. Genberegnet fra eksisterende komponenter.
- Raphinha: har samlet/fordelt runde-EV, men mangler basekomponenter som goal/assist/SOT/start_minutes. Ikke genberegnet, fordi det ville kraeve at opfinde komponentfordeling.
- Jules Kounde: mangler fortsat EV-kilde og komponenter. Ikke genberegnet.
- Manuel Neuer: mangler fortsat EV-kilde og komponenter. Ikke genberegnet.

## Eksempler paa samlet EV men manglende komponenter

| player_name | team_id | position | weighted_group_stage_ev | suspected_issue |
| --- | --- | --- | --- | --- |

## Eksempler paa spillere uden EV-kilde

| player_name | team_id | position | weighted_group_stage_ev | suspected_issue |
| --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | MID | 0.0 | no_player_ev_source |
