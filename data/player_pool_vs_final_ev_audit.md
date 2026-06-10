# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 908 -> 0
- `optimizer_ev` forskel > 0.10: 513 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 0
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260609_220205.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 916 | 0 |
| weighted_group_stage_ev | 916 | 0 |
| weighted_group_stage_ev_before_price_quality | 2 | 0 |
| price_quality_ev | 1 | 0 |
| model_ev_before_price_quality | 4 | 0 |
| optimizer_ev_before_price_quality | 2 | 0 |
| price_quality_raw_ev | 3 | 0 |
| price_quality_appearance_scaled_ev | 4 | 0 |
| price_quality_base_capped_ev | 3 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 1 | 0 |
| price_quality_method | 2 | 0 |
| base_ev_source | 4 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.94753 | 3.593876 | 3.593876 | updated |
| Leo Pereira | BRA | 1.928714 | 2.374221 | 2.374221 | updated |
| Unai Simon | ESP | 4.576512 | 5.884328 | 5.884328 | updated |
| Joan Garcia | ESP | 1.1523378712871288 | 0.361513 | 0.361513 | updated |
| Patrick Agyemang | USA | 0.0 | 0.0 | 0.0 | already_equal |
| Noni Madueke | ENG | 1.879299 | 2.297708 | 2.297708 | updated |
