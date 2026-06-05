# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 397 -> 0
- `optimizer_ev` forskel > 0.10: 8 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260605_230054.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 837 | 0 |
| weighted_group_stage_ev | 837 | 0 |
| weighted_group_stage_ev_before_price_quality | 625 | 0 |
| price_quality_ev | 759 | 0 |
| model_ev_before_price_quality | 625 | 0 |
| optimizer_ev_before_price_quality | 625 | 0 |
| price_quality_raw_ev | 905 | 0 |
| price_quality_appearance_scaled_ev | 769 | 0 |
| price_quality_base_capped_ev | 703 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 7 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.864975 | 2.88363 | 2.88363 | updated |
| Leo Pereira | BRA | 2.793132 | 2.809037 | 2.809037 | updated |
| Unai Simon | ESP | 4.503454 | 4.511835 | 4.511835 | updated |
| Joan Garcia | ESP | 0.307973 | 0.308375 | 0.308375 | updated |
| Patrick Agyemang | USA | 0.929936 | 0.929936 | 0.929936 | already_equal |
| Noni Madueke | ENG | 1.600452 | 1.600453 | 1.600453 | updated |
