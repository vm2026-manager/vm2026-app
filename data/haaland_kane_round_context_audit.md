# Haaland/Kane Round Context Audit

Ren audit af eksisterende runde- og fixturefelter. Ingen produktionsoutput er skrevet.

## Fokus

| player_name | round | opponent | win_probability | goal_multiplier | goal_share_norm | start_prob | minutes_if_start | match_goal_ev | round_ev | fixture_mapping_status | plausibility_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harry Kane | 1 | CRO | 0.5431 | 1.2374 | 0.082 | 0.8702 | 66.527 | 0.213065 | 1.343986 | ok | Round context is present; values are moderate, not missing, but may understate premium ceiling. |
| Harry Kane | 2 | GHA | 0.7092 | 1.35 | 0.082 | 0.8702 | 66.527 | 0.267503 | 1.523111 | ok | Round context is present; values are moderate, not missing, but may understate premium ceiling. |
| Harry Kane | 3 | PAN | 0.7628 | 1.35 | 0.082 | 0.8702 | 66.527 | 0.225814 | 1.19965 | ok | Round context is present; values are moderate, not missing, but may understate premium ceiling. |
| Erling Haaland | 1 | IRQ | 0.7579 | 1.35 | 0.097 | 0.8883 | 66.409 | 0.258297 | 1.543013 | ok | NOR-IRQ mapping is present and fixture is favorable, but match_goal_ev=0.047 is very low for a premium FWD; low value is not explained by fixture mapping. |
| Erling Haaland | 2 | SEN | 0.4575 | 1.1364 | 0.097 | 0.8883 | 66.409 | 0.204147 | 1.200635 | ok | Round context present. |
| Erling Haaland | 3 | FRA | 0.1969 | 0.7527 | 0.097 | 0.8883 | 66.409 | 0.086475 | 0.629696 | ok | Round context present. |

## Konklusion

- Haaland er korrekt mappet som NOR mod IRQ i runde 1 med win probability 0,7579 og goal multiplier 1,35.
- Hans lave next-round value skyldes derfor ikke åbenlys team/fixture-mappingfejl. Den skyldes især meget lav `match_1_goal_ev` på 0,047385 og lav samlet `match_1_total_ev_next_match` på 0,237368.
- Det virker svagt for Haaland mod Irak ud fra de tilgængelige outputfelter, men det bør testes i EV/round-context-modellen, ikke rettes manuelt her.
- Kane har reel fixture-specifik kontekst og moderate rundeværdier; hans problem er mere ceiling/round-weighting end missing data.
