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
- Rækker genberegnet: 864
- Team/match/on_pitch high-start spreads > 0.05 foer: 3
- Team/match/on_pitch high-start spreads > 0.05 efter: 0
- Negative on_pitch_ev for start_prob >= 0.70 foer: 0
- Negative on_pitch_ev for start_prob >= 0.70 efter: 0
- Stoerste high-start on_pitch spread foer: 0.067600
- Stoerste high-start on_pitch spread efter: 0.027252
- Samlet EV men manglende basekomponenter efter: 0
- Uden EV-kilde efter: 0
- Backup: `data\player_ev_group_stage_v1.backup_before_component_rebuild_20260605_182822.csv`

## Sanity-spillere

| player_name | team_id | start_prob_before | start_prob_after | match_1_goal_ev_before | match_1_goal_ev_after | match_1_start_minutes_ev_before | match_1_start_minutes_ev_after | match_1_weighted_match_ev_before | match_1_weighted_match_ev_after | weighted_group_stage_ev_before | weighted_group_stage_ev_after | issue_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | ENG | 0.92 | 0.92 | 0.225259 | 0.225259 | 61.20484 | 61.20484 | 1.426476 | 1.426476 | 4.201545 | 4.340567 | ok |
| Mike Maignan | FRA | 0.9407 | 0.9407 | 0.0 | 0.0 | 75.856167 | 75.856167 | 1.442053 | 1.442053 | 4.153485 | 4.153485 | ok |
| Erling Haaland | NOR | 0.9058 | 0.9058 | 0.263386 | 0.263386 | 60.153272 | 60.153272 | 1.574396 | 1.574396 | 3.694744 | 3.442615 | ok |
| Alexander Schlager | AUT | 0.921 | 0.921 | 0.0 | 0.0 | 73.260945 | 73.260945 | 1.4956 | 1.4956 | 2.904687 | 3.047048 | ok |
| Antonio Nusa | NOR | 0.6563 | 0.6563 | 0.190837 | 0.190837 | 43.584227 | 43.584227 | 1.365923 | 1.365923 | 2.68807 | 2.966811 | ok |
| Alexander Sørloth | NOR | 0.8805 | 0.8805 | 0.187096 | 0.187096 | 58.473125 | 58.473125 | 1.253291 | 1.253291 | 3.038442 | 2.766432 | ok |
| Jules Kounde | FRA | 0.6814 | 0.6814 |  |  |  |  | 0.907738 | 0.907738 | 2.864972 | 2.702042 | ok |
| Philipp Lienhart | AUT | 0.6578 | 0.6578 | 0.069328 | 0.069328 | 49.574439 | 49.574439 | 1.377354 | 1.377354 | 2.594027 | 2.623064 | ok |
| Stefan Posch | AUT | 0.636 | 0.636 | 0.052309 | 0.052309 | 48.05298 | 48.05298 | 1.294632 | 1.294632 | 2.475854 | 2.475854 | ok |
| Gregor Kobel | SUI | 0.4806 | 0.4806 | 0.0 | 0.0 | 38.627264 | 38.627264 | 0.89393 | 0.89393 | 2.189701 | 2.134492 | ok |
| Manuel Neuer | GER | 0.3702 | 0.3702 |  |  |  |  | 0.990382 | 0.990382 | 2.08055 | 2.086723 | ok |
| Martin Ødegaard | NOR | 0.8841 | 0.8841 | 0.079099 | 0.079099 | 62.248597 | 62.248597 | 0.823324 | 0.823323 | 2.246306 | 1.787764 | ok |
| Maximilian Wöber | AUT | 0.5612 | 0.5612 | 0.022139 | 0.022139 | 41.834654 | 41.834654 | 0.903613 | 0.903613 | 1.635364 | 1.635364 | ok |
| Kevin Danso | AUT | 0.3362 | 0.3362 | 0.012368 | 0.012368 | 24.999832 | 24.999832 | 0.620027 | 0.620027 | 1.076285 | 1.076285 | ok |
| Raphinha | BRA | 0.6781 | 0.6781 |  |  |  |  | 0.204298 | 0.204298 | 0.597978 | 0.488147 | ok |
| Patrick Pentz | AUT | 0.054 | 0.054 | 0.0 | 0.0 | 4.324914 | 4.324914 | 0.217369 | 0.217369 | 0.313837 | 0.447661 | ok |

## NOR vs IRQ efter

| player_name | start_prob | match_1_result_ev | match_1_team_scores_ev | match_1_opponent_scores_ev | match_1_on_pitch_ev |
| --- | --- | --- | --- | --- | --- |
| Erling Haaland | 0.9058 | 0.153239 | 0.06848 | -0.056938 | 0.058696 |
| Martin Ødegaard | 0.8841 | 0.149568 | 0.066839 | -0.055574 | 0.056092 |
| Alexander Sørloth | 0.8805 | 0.148959 | 0.066567 | -0.055347 | 0.05566 |
| Antonio Nusa | 0.6563 | 0.137996 | 0.061668 | -0.051274 | 0.047884 |
| Oscar Bobb | 0.6439 | 0.156927 | 0.070128 | -0.058308 | 0.061312 |

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
