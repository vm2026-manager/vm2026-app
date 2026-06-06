# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 1 -> 0
- `optimizer_ev` forskel > 0.10: 1 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260606_144930.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 739 | 0 |
| weighted_group_stage_ev | 739 | 0 |
| weighted_group_stage_ev_before_price_quality | 620 | 0 |
| price_quality_ev | 702 | 0 |
| model_ev_before_price_quality | 620 | 0 |
| optimizer_ev_before_price_quality | 620 | 0 |
| price_quality_raw_ev | 925 | 0 |
| price_quality_appearance_scaled_ev | 743 | 0 |
| price_quality_base_capped_ev | 727 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 1 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.876335 | 2.876339 | 2.876339 | updated |
| Leo Pereira | BRA | 1.88688 | 1.886881 | 1.886881 | updated |
| Unai Simon | ESP | 4.556737 | 4.556739 | 4.556739 | updated |
| Joan Garcia | ESP | 0.310528 | 0.310528 | 0.310528 | already_equal |
| Patrick Agyemang | USA | 0.930061 | 0.930062 | 0.930062 | updated |
| Noni Madueke | ENG | 1.60057 | 1.60057 | 1.60057 | already_equal |
