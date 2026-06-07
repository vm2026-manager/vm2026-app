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

- Stale komponenter foer: 87
- Stale komponenter efter: 0
- Rækker genberegnet: 904
- Team/match/on_pitch high-start spreads > 0.05 foer: 6
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.067600
- Stoerste high-start on_pitch spread efter: 0.027252
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 0
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260607_123003.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | 0.97 | 0.97 | 0.237501 | 0.237501 | 64.53119 | 64.53119 | 1.51232 | 1.512321 | 4.343463 | 4.599483 | ok |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 0.0 | 0.0 | 75.856167 | 75.856167 | 1.442053 | 1.442053 | 4.162266 | 4.344702 | ok |
| Gregor Kobel | SUI | 0.9192 | 0.9192 | 0.0 | 0.0 | 73.73419 | 73.73419 | 1.669274 | 1.671619 | 3.679057 | 3.991579 | ok |
| Antonio Nusa | NOR | 0.82 | 0.82 | 0.238437 | 0.238437 | 54.45538 | 54.45538 | 1.701233 | 1.701233 | 3.199014 | 3.697816 | ok |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 0.263386 | 0.263386 | 60.153272 | 60.153272 | 1.574396 | 1.574396 | 3.694285 | 3.442635 | ok |
| Philipp Lienhart | AUT | 0.84 | 0.84 | 0.088531 | 0.088531 | 63.30576 | 63.30576 | 1.705033 | 1.705033 | 3.02023 | 3.02023 | ok |
| Stefan Posch | AUT | 0.84 | 0.84 | 0.069087 | 0.069087 | 63.4662 | 63.4662 | 1.647953 | 1.647953 | 2.968204 | 2.968204 | ok |
| Alexander Schlager | AUT | 0.9212 | 0.9212 | 0.0 | 0.0 | 73.260945 | 73.260945 | 1.49587 | 1.49587 | 2.911009 | 2.911009 | ok |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 0.187096 | 0.187096 | 58.473125 | 58.473125 | 1.25329 | 1.253291 | 3.0383 | 2.76645 | ok |
| Jules Kounde | FRA | 0.6814 | 0.6814 |  |  |  |  | 0.907738 | 0.907738 | 2.88355 | 2.702052 | ok |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 0.079099 | 0.079099 | 62.248597 | 62.248597 | 0.823323 | 0.823323 | 2.289866 | 1.787782 | ok |
| Manuel Neuer | GER | 0.2645 | 0.2645 |  |  |  |  | 0.988455 | 0.786729 | 2.080418 | 1.654787 | ok |
| Maximilian Wöber | AUT | 0.559 | 0.559 | 0.022139 | 0.022139 | 41.834654 | 41.834654 | 0.901258 | 0.901258 | 1.6378 | 1.6378 | ok |
| Kevin Danso | AUT | 0.3346 | 0.3346 | 0.012368 | 0.012368 | 24.999832 | 24.999832 | 0.618315 | 0.618315 | 1.077068 | 1.217594 | ok |
| Raphinha | BRA | 0.86 | 0.86 |  |  |  |  | 0.228093 | 0.228093 | 2.052412 | 0.557194 | ok |
| Patrick Pentz | AUT | 0.0538 | 0.0538 | 0.0 | 0.0 | 4.324914 | 4.324914 | 0.217097 | 0.217097 | 0.31361 | 0.447118 | ok |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.068484 | -0.056942 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066844 | -0.055579 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066572 | -0.055352 | 0.05566 |
| Antonio Nusa | 0.82 | 0.162408 | 0.072582 | -0.06035 | 0.0652 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070133 | -0.058313 | 0.061312 |

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
