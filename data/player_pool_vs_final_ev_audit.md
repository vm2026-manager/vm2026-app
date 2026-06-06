# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 0 -> 0
- `optimizer_ev` forskel > 0.10: 0 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260606_093238.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 0 | 0 |
| weighted_group_stage_ev | 0 | 0 |
| weighted_group_stage_ev_before_price_quality | 0 | 0 |
| price_quality_ev | 0 | 0 |
| model_ev_before_price_quality | 0 | 0 |
| optimizer_ev_before_price_quality | 0 | 0 |
| price_quality_raw_ev | 0 | 0 |
| price_quality_appearance_scaled_ev | 0 | 0 |
| price_quality_base_capped_ev | 0 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 0 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.88363 | 2.88363 | 2.88363 | already_equal |
| Leo Pereira | BRA | 2.809037 | 2.809037 | 2.809037 | already_equal |
| Unai Simon | ESP | 4.511835 | 4.511835 | 4.511835 | already_equal |
| Joan Garcia | ESP | 0.308375 | 0.308375 | 0.308375 | already_equal |
| Patrick Agyemang | USA | 0.930268 | 0.930268 | 0.930268 | already_equal |
| Noni Madueke | ENG | 1.600831 | 1.600831 | 1.600831 | already_equal |
