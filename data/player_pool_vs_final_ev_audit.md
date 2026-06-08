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
- Pool-rækker uden exact EV-match: 0
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260608_185627.json`

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
| Jules Kounde | FRA | 2.94753 | 2.94753 | 2.94753 | already_equal |
| Leo Pereira | BRA | 1.928714 | 1.928714 | 1.928714 | already_equal |
| Unai Simon | ESP | 4.576512 | 4.576512 | 4.576512 | already_equal |
| Joan Garcia | ESP | 1.1523378712871288 | 1.1523378712871288 | 1.1523378712871288 | already_equal |
| Patrick Agyemang | USA | 0.0 | 0.0 | 0.0 | already_equal |
| Noni Madueke | ENG | 1.879299 | 1.879299 | 1.879299 | already_equal |
