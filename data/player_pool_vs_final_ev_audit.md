# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 517 -> 0
- `optimizer_ev` forskel > 0.10: 8 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260606_213235.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 886 | 0 |
| weighted_group_stage_ev | 886 | 0 |
| weighted_group_stage_ev_before_price_quality | 628 | 0 |
| price_quality_ev | 857 | 0 |
| model_ev_before_price_quality | 628 | 0 |
| optimizer_ev_before_price_quality | 628 | 0 |
| price_quality_raw_ev | 1085 | 0 |
| price_quality_appearance_scaled_ev | 922 | 0 |
| price_quality_base_capped_ev | 766 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 7 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.876339 | 2.889142 | 2.889142 | updated |
| Leo Pereira | BRA | 1.886881 | 1.893899 | 1.893899 | updated |
| Unai Simon | ESP | 4.556739 | 4.556739 | 4.556739 | already_equal |
| Joan Garcia | ESP | 0.310528 | 0.310528 | 0.310528 | already_equal |
| Patrick Agyemang | USA | 0.930062 | 0.930062 | 0.930062 | already_equal |
| Noni Madueke | ENG | 1.60057 | 1.600571 | 1.600571 | updated |
