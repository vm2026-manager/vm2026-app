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

- Stale komponenter foer: 0
- Stale komponenter efter: 0
- Rækker genberegnet: 871
- Team/match/on_pitch high-start spreads > 0.05 foer: 3
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.067600
- Stoerste high-start on_pitch spread efter: 0.027252
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 0
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260606_120529.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | 0.97 | 0.97 | 0.237501 | 0.237501 | 64.53119 | 64.53119 | 1.51232 | 1.512321 | 4.342915 | 4.599476 | ok |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 0.0 | 0.0 | 75.856167 | 75.856167 | 1.442054 | 1.442054 | 4.16187 | 4.344697 | ok |
| Gregor Kobel | SUI | 0.9174 | 0.9174 | 0.0 | 0.0 | 73.73419 | 73.73419 | 1.668806 | 1.668805 | 3.678119 | 3.984935 | ok |
| Antonio Nusa | NOR | 0.82 | 0.82 | 0.238437 | 0.238437 | 54.45538 | 54.45538 | 1.701233 | 1.701234 | 3.160292 | 3.697808 | ok |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 0.263386 | 0.263386 | 60.153272 | 60.153272 | 1.574396 | 1.574396 | 3.69375 | 3.442625 | ok |
| Alexander Schlager | AUT | 0.921 | 0.921 | 0.0 | 0.0 | 73.260945 | 73.260945 | 1.495599 | 1.495599 | 2.910433 | 3.04705 | ok |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 0.187096 | 0.187096 | 58.473125 | 58.473125 | 1.25329 | 1.253291 | 3.03812 | 2.76644 | ok |
| Jules Kounde | FRA | 0.6814 | 0.6814 |  |  |  |  | 0.907738 | 0.907738 | 2.876427 | 2.702047 | ok |
| Philipp Lienhart | AUT | 0.6578 | 0.6578 | 0.069328 | 0.069328 | 49.574439 | 49.574439 | 1.377353 | 1.377353 | 2.603202 | 2.623067 | ok |
| Stefan Posch | AUT | 0.636 | 0.636 | 0.052309 | 0.052309 | 48.05298 | 48.05298 | 1.29463 | 1.29463 | 2.484724 | 2.47758 | ok |
| Manuel Neuer | GER | 0.3702 | 0.3702 |  |  |  |  | 0.990382 | 0.990382 | 2.084986 | 2.086728 | ok |
| Raphinha | BRA | 0.86 | 0.86 |  |  |  |  | 0.228093 | 0.228093 | 2.051947 | 2.051947 | ok |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 0.079099 | 0.079099 | 62.248597 | 62.248597 | 0.823323 | 0.823324 | 2.245913 | 1.787776 | ok |
| Maximilian Wöber | AUT | 0.5612 | 0.5612 | 0.022139 | 0.022139 | 41.834654 | 41.834654 | 0.903612 | 0.903612 | 1.639984 | 1.745109 | ok |
| Kevin Danso | AUT | 0.3362 | 0.3362 | 0.012368 | 0.012368 | 24.999832 | 24.999832 | 0.620026 | 0.620026 | 1.079053 | 1.221053 | ok |
| Patrick Pentz | AUT | 0.054 | 0.054 | 0.0 | 0.0 | 4.324914 | 4.324914 | 0.217368 | 0.217368 | 0.314146 | 0.447662 | ok |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.068482 | -0.05694 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066842 | -0.055576 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.06657 | -0.05535 | 0.05566 |
| Antonio Nusa | 0.82 | 0.162408 | 0.07258 | -0.060347 | 0.0652 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.07013 | -0.05831 | 0.061312 |

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
