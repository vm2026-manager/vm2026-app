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

- Stale komponenter foer: 519
- Stale komponenter efter: 0
- Rækker genberegnet: 1137
- Team/match/on_pitch high-start spreads > 0.05 foer: 3
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.067600
- Stoerste high-start on_pitch spread efter: 0.027252
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 0
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260605_173327.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 0.0 | 0 | 71.485587 | 75.856167 | 1.380956 | 1.442053 | 3.720248 | 4.344689 | ok |
| Harry Kane | ENG | 0.92 | 0.92 | 0.208144 | 0.225259 | 56.554603 | 61.20484 | 1.317357 | 1.423762 | 3.976067 | 4.307649 | ok |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 0.258297 | 0.263386 | 58.991115 | 60.153272 | 1.543012 | 1.574396 | 3.627429 | 3.442623 | ok |
| Alexander Schlager | AUT | 0.921 | 0.921 | 0.0 | 0 | 66.778028 | 73.260945 | 1.386664 | 1.4956 | 2.57799 | 3.047047 | ok |
| Antonio Nusa | NOR | 0.6563 | 0.6563 | 0.201334 | 0.190837 | 45.981592 | 43.584227 | 1.43026 | 1.365923 | 2.88954 | 2.966819 | ok |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 0.159876 | 0.187096 | 49.966132 | 58.473125 | 1.080406 | 1.253291 | 2.804337 | 2.766437 | ok |
| Jules Kounde | FRA | 0.6814 | 0.6814 |  |  |  |  | 1.069486 | 0.907738 | 3.33847 | 2.70204 | ok |
| Philipp Lienhart | AUT | 0.6578 | 0.6578 | 0.086728 | 0.069328 | 62.017036 | 49.574439 | 1.669303 | 1.377354 | 3.126866 | 2.623064 | ok |
| Stefan Posch | AUT | 0.636 | 0.636 | 0.055138 | 0.052309 | 50.652072 | 48.05298 | 1.353124 | 1.294632 | 2.74975 | 2.477577 | ok |
| Gregor Kobel | SUI | 0.4806 | 0.4806 | 0.0 | 0 | 46.246624 | 38.627264 | 1.039593 | 0.89393 | 2.361878 | 2.13449 | ok |
| Manuel Neuer | GER | 0.3702 | 0.3702 |  |  |  |  | 0.734904 | 0.990382 | 1.33666 | 2.08672 | ok |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 0.073874 | 0.079099 | 58.136711 | 62.248597 | 0.769198 | 0.823324 | 2.044715 | 1.787772 | ok |
| Maximilian Wöber | AUT | 0.5612 | 0.5612 | 0.023733 | 0.022139 | 44.846272 | 41.834654 | 0.957492 | 0.903613 | 1.877249 | 1.745107 | ok |
| Kevin Danso | AUT | 0.3362 | 0.3362 | 0.01942 | 0.012368 | 39.254644 | 24.999832 | 0.871466 | 0.620027 | 1.691063 | 1.221051 | ok |
| Raphinha | BRA | 0.6781 | 0.6781 |  |  |  |  | 0.204083 | 0.204083 | 0.594828 | 0.485575 | ok |
| Patrick Pentz | AUT | 0.054 | 0.054 | 0.0 | 0 | 8.609782 | 4.324914 | 0.288835 | 0.217369 | 0.441704 | 0.447661 | ok |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.068479 | -0.056937 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066839 | -0.055573 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066567 | -0.055347 | 0.05566 |
| Antonio Nusa | 0.6563 | 0.137996 | 0.061667 | -0.051273 | 0.047884 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070127 | -0.058307 | 0.061312 |

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
