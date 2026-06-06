# Offensive Fallback Optimizer Dry-Run

| strategy | formation_before | formation_after | price_before | price_after | ev_before | ev_after | score_before | score_after | players_in | players_out | fallback_player_count | fallback_players | high_risk_before | high_risk_after | dominance_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_round | 3-4-3 | 3-4-3 | 49500000 | 49500000 | 43.3972 | 43.3996 | 94.3646 | 94.3665 |  |  | 0 |  | 0 | 0 |  |
| round1_2 | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 44.5742 | 44.5749 | 113.1555 | 113.156 |  |  | 0 |  | 0 | 0 |  |
| group_stage | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 44.8837 | 44.8845 | 127.2461 | 127.2466 |  |  | 0 |  | 0 | 0 |  |
| long_run | 4-3-3 | 4-3-3 | 50000000 | 50000000 | 38.8461 | 38.5008 | 70.9912 | 71.0763 | Fabian Ruiz | Declan Rice | 1 | Fabian Ruiz | 0 | 0 |  |

## Sikkerhed

- Ingen strategi har fire eller flere fallback-spillere.
- Kun `long_run` skifter spiller i denne dry-run, og holdet indeholder én fallbackspiller.

Dry-run-optimeringen bruger uændrede formationer, budgetregler, maksimum fire pr. land og strategiscores.
