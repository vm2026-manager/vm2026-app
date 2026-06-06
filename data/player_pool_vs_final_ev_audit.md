# Player pool vs final EV audit

## Autoritet og retning

- Start-, availability-, Holdet-, pris-, hold- og positionsfelter forbliver autoritative i player pool og synkroniseres ikke her.
- Færdig EV og price-quality-proveniens synkroniseres kun `player_ev_group_stage_v1.csv -> player_pool_v1.json` via exact `player_id`.
- Navne-, hold- og rækkefølgefallback bruges ikke. Dublerede IDs blokeres.

## Før/efter

- `optimizer_ev` forskel > 0.001: 24 -> 0
- `optimizer_ev` forskel > 0.10: 23 -> 0
- Dublerede player_id i pool: 0
- Dublerede player_id i EV: 0
- Pool-rækker uden exact EV-match: 48
- EV-rækker uden exact pool-match: 0
- Backup: `data\player_pool_v1.backup_before_final_ev_sync_20260606_105810.json`

## Feltmismatches

| Felt | Før | Efter |
| --- | ---: | ---: |
| optimizer_ev | 671 | 0 |
| weighted_group_stage_ev | 671 | 0 |
| weighted_group_stage_ev_before_price_quality | 586 | 0 |
| price_quality_ev | 619 | 0 |
| model_ev_before_price_quality | 586 | 0 |
| optimizer_ev_before_price_quality | 586 | 0 |
| price_quality_raw_ev | 794 | 0 |
| price_quality_appearance_scaled_ev | 648 | 0 |
| price_quality_base_capped_ev | 642 | 0 |
| price_quality_weight | 0 | 0 |
| price_quality_spread_multiplier | 0 | 0 |
| price_quality_applied | 0 | 0 |
| price_quality_method | 2 | 0 |
| base_ev_source | 0 | 0 |

## Sanity

| Spiller | Hold | Pool før | EV | Pool efter | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Jules Kounde | FRA | 2.88363 | 2.883631 | 2.883631 | updated |
| Leo Pereira | BRA | 2.809037 | 2.809039 | 2.809039 | updated |
| Unai Simon | ESP | 4.511835 | 4.511834 | 4.511834 | updated |
| Joan Garcia | ESP | 0.308375 | 0.308375 | 0.308375 | already_equal |
| Patrick Agyemang | USA | 0.930268 | 0.930238 | 0.930238 | updated |
| Noni Madueke | ENG | 1.600831 | 1.600729 | 1.600729 | updated |
