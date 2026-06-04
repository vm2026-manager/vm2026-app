# Player EV Fixture Data Gap Report

Repairen canonicaliserer kun sikre Holdet-team aliases og fordeler eksisterende optimizer/weighted EV til runde-context. Den opfinder ikke ny spiller-EV.

## Counts

- Gaps foer: 539
- Gaps efter: 87
- Placeholder IDs/teams foer: 24
- Placeholder IDs/teams efter: 0
- Missing round context for existing EV foer: 405
- Missing round context for existing EV efter: 0
- Player-pool identity changes: 24
- EV identity changes: 24
- EV restored from existing price-quality diagnostics: 51
- Pool EV restored from existing price-quality diagnostics: 86
- Round contexts filled from existing optimizer EV: 539

## Remaining no-EV-source examples

- Nathaniel Brown | GER | DEF | 2500000
- Maxence Lacroix | FRA | DEF | 3000000
- Jules Kounde | FRA | DEF | 3500000
- Marc Pubill | ESP | DEF | 3000000
- Eric Garcia | ESP | DEF | 3000000
- Djed Spence | ENG | DEF | 2500000
- Jarell Quansah | ENG | DEF | 2500000
- Ko Itakura | JPN | DEF | 2000000
- Takehiro Tomiyasu | JPN | DEF | 2000000
- Gi-hyuk Lee | KOR | DEF | 2000000
- Francis de Vries | NZL | DEF | 2500000
- Michael Boxall | NZL | DEF | 2000000
- Tommy Smith | NZL | DEF | 2000000
- Nando Pijnaker | NZL | DEF | 2000000
- Yerry Mina | COL | DEF | 3000000
- Willer Ditta | COL | DEF | 2500000
- Karim Hafez | EGY | DEF | 2000000
- Issa Diop | MAR | DEF | 2500000
- Raed Chikhaoui | TUN | DEF | 2000000
- Moustapha Mbow | SEN | DEF | 2000000
- Aaron Hickey | SCO | DEF | 2500000
- Logan Costa | CPV | DEF | 2500000
- Sondre Langås | NOR | DEF | 2000000
- Armando Obispo | CUW | DEF | 2000000
- Deveron Fonville | CUW | DEF | 2000000
- ... plus 62 flere i CSV.

## Remaining round-context examples


## Notes

- Kounde og Neuer har stadig ingen EV-kilde i eksisterende mellemdata; deres fixtures kan findes, men spiller-EV maa genbygges upstream.
- Raphinha, Wesley Franca og Mahmoud Trezeguet havde optimizer_ev uden runde-context; runde-context er udfyldt fra eksisterende EV og fixture-multipliers.
- HOLDET_584 er canonicaliseret til CZE i player_pool og EV-output.
