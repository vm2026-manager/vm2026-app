# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 820 -> 0
- `optimizer_ev` forskel > 0.10: 131 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260607_123004.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 1096 | 0 |
| weighted_group_stage_ev | 1096 | 0 |
| weighted_group_stage_ev_before_price_quality | 739 | 0 |
| price_quality_ev | 1094 | 0 |
| model_ev_before_price_quality | 739 | 0 |
| optimizer_ev_before_price_quality | 739 | 0 |
| price_quality_raw_ev | 1189 | 0 |
| price_quality_appearance_scaled_ev | 1189 | 0 |
| price_quality_base_capped_ev | 858 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 40 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.88355 | 2.912575 | 2.912575 | updated |
| Leo Pereira | BRA | 1.890835 | 1.906037 | 1.906037 | updated |
| Unai Simon | ESP | 4.510969 | 4.521933 | 4.521933 | updated |
| Joan Garcia | ESP | 0.308857 | 0.309377 | 0.309377 | updated |
| Patrick Agyemang | USA | 0.930044 | 0.92823 | 0.92823 | updated |
| Noni Madueke | ENG | 1.600553 | 1.594603 | 1.594603 | updated |
