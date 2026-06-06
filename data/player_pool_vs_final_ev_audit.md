# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 346 -> 0
- `optimizer_ev` forskel > 0.10: 24 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260606_110808.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 933 | 0 |
| weighted_group_stage_ev | 933 | 0 |
| weighted_group_stage_ev_before_price_quality | 662 | 0 |
| price_quality_ev | 903 | 0 |
| model_ev_before_price_quality | 662 | 0 |
| optimizer_ev_before_price_quality | 662 | 0 |
| price_quality_raw_ev | 1050 | 0 |
| price_quality_appearance_scaled_ev | 968 | 0 |
| price_quality_base_capped_ev | 797 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 2 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.883631 | 2.876427 | 2.876427 | updated |
| Leo Pereira | BRA | 2.809039 | 1.886863 | 1.886863 | updated |
| Unai Simon | ESP | 4.511834 | 4.511834 | 4.511834 | already_equal |
| Joan Garcia | ESP | 0.308375 | 0.308375 | 0.308375 | already_equal |
| Patrick Agyemang | USA | 0.930238 | 0.930238 | 0.930238 | already_equal |
| Noni Madueke | ENG | 1.600729 | 1.600729 | 1.600729 | already_equal |
