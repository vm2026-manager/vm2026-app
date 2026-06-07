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

- Stale komponenter foer: 35
- Stale komponenter efter: 34
- Rækker genberegnet: 1177
- Team/match/on_pitch high-start spreads > 0.05 foer: 0
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.027252
- Stoerste high-start on_pitch spread efter: 0.027252
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 20
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260607_165336.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 0.0 | 0.0 | 75.856167 | 75.856167 | 1.442053 | 1.432882 | 4.227811 | 17.250604 | ok |
| Manuel Neuer | GER | 0.2771 | 0.2771 | 0.0 | 0.0 |  |  | 0.811008 | 0.800258 | 1.666128 | 16.33198 | ok |
| Raphinha | BRA | 0.86 | 0.86 | 0.0 | 0.0 |  |  | 0.228093 | 0.228316 | 2.338525 | 14.236737 | ok |
| Jules Kounde | FRA | 0.6814 | 0.6814 | 0.0 | 0.0 |  |  | 0.907738 | 0.899625 | 2.94753 | 14.11902 | ok |
| Alexander Schlager | AUT | 0.9212 | 0.9212 | 0.0 | 0.0 | 73.260945 | 73.260945 | 1.495869 | 1.484124 | 2.909014 | 13.732106 | ok |
| Harry Kane | ENG | 0.97 | 0.97 | 0.293884 | 0.293884 | 64.53119 | 64.53119 | 1.776307 | 1.772914 | 5.248523 | 13.020765 | ok |
| Philipp Lienhart | AUT | 0.84 | 0.84 | 0.119517 | 0.119517 | 63.30576 | 63.30576 | 1.917802 | 1.907734 | 3.196748 | 12.606667 | ok |
| Stefan Posch | AUT | 0.84 | 0.84 | 0.093267 | 0.093267 | 63.4662 | 63.4662 | 1.834427 | 1.824359 | 3.12944 | 12.484289 | ok |
| Gregor Kobel | SUI | 0.9145 | 0.9145 | 0.0 | 0.0 | 73.73419 | 73.73419 | 1.664269 | 1.667305 | 3.698663 | 9.71441 | ok |
| Kevin Danso | AUT | 0.3346 | 0.3346 | 0.016697 | 0.016697 | 24.999832 | 24.999832 | 0.652242 | 0.643689 | 1.112871 | 9.031111 | ok |
| Antonio Nusa | NOR | 0.82 | 0.82 | 0.32189 | 0.32189 | 54.45538 | 54.45538 | 2.165674 | 2.172641 | 3.63366 | 8.685508 | ok |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 0.355571 | 0.355571 | 60.153272 | 60.153272 | 1.995246 | 2.00182 | 4.241624 | 8.129447 | ok |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 0.25258 | 0.25258 | 58.473125 | 58.473125 | 1.565882 | 1.572272 | 3.452912 | 7.218725 | ok |
| Patrick Pentz | AUT | 0.0538 | 0.0538 | 0.0 | 0.0 | 4.324914 | 4.324914 | 0.217096 | 0.209807 | 0.313331 | 7.078323 | ok |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 0.106784 | 0.106784 | 62.248597 | 62.248597 | 1.025325 | 1.031741 | 2.593366 | 6.136805 | ok |
| Maximilian Wöber | AUT | 0.0 | 0.0 | 0.029888 | 0.029888 | 41.834654 | 41.834654 | 0.961987 | 0.961987 | 0.0 | 0.0 | stale_start_dependent_components |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.09058 | -0.072464 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.08841 | -0.070728 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.08805 | -0.07044 | 0.05566 |
| Antonio Nusa | 0.82 | 0.162408 | 0.096 | -0.0768 | 0.0652 |
| Oscar Bobb | 0.58 | 0.156927 | 0.09276 | -0.074208 | 0.061312 |

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
| Viljar Myhra | NOR | GK | 0.0 | no_player_ev_source |
| Bruno Varela | CPV | GK | 0.0 | no_player_ev_source |
| Scott Bain | SCO | GK | 0.0 | no_player_ev_source |
| Noureddine Farhati | TUN | GK | 0.0 | no_player_ev_source |
| Bechir Ben Said | TUN | GK | 0.0 | no_player_ev_source |
| Maarten Vandevoordt | BEL | GK | 0.0 | no_player_ev_source |
| Matz Sels | BEL | GK | 0.0 | no_player_ev_source |
| Solomon Agbasi | GHA | GK | 0.0 | no_player_ev_source |
| Paul Reverson | GHA | GK | 0.0 | no_player_ev_source |
| Bento Krepski | BRA | GK | 0.0 | no_player_ev_source |
| Melker Ellborg | SWE | GK | 0.0 | no_player_ev_source |
