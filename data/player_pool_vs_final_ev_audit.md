# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 8 -> 0
- `optimizer_ev` forskel > 0.10: 0 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260605_182824.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 676 | 0 |
| weighted_group_stage_ev | 676 | 0 |
| weighted_group_stage_ev_before_price_quality | 635 | 0 |
| price_quality_ev | 598 | 0 |
| model_ev_before_price_quality | 635 | 0 |
| optimizer_ev_before_price_quality | 635 | 0 |
| price_quality_raw_ev | 973 | 0 |
| price_quality_appearance_scaled_ev | 584 | 0 |
| price_quality_base_capped_ev | 720 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 0 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.864972 | 2.864973 | 2.864973 | updated |
| Leo Pereira | BRA | 2.793129 | 2.79313 | 2.79313 | updated |
| Unai Simon | ESP | 4.503452 | 4.503453 | 4.503453 | updated |
| Joan Garcia | ESP | 0.307973 | 0.307973 | 0.307973 | already_equal |
| Patrick Agyemang | USA | 0.929935 | 0.929935 | 0.929935 | already_equal |
| Noni Madueke | ENG | 1.600451 | 1.600452 | 1.600452 | updated |
