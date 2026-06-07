# Player EV Fixture Data Gap Report

Repairen canonicaliserer kun sikre Holdet-team aliases og fordeler eksisterende optimizer/weighted EV til runde-context. Den opfinder ikke ny spiller-EV.

## Counts

- Gaps foer: 280
- Gaps efter: 280
- Placeholder IDs/teams foer: 0
- Placeholder IDs/teams efter: 0
- Missing round context for existing EV foer: 0
- Missing round context for existing EV efter: 0
- Player-pool identity changes: 0
- EV identity changes: 0
- EV restored from existing price-quality diagnostics: 1
- Pool EV restored from existing price-quality diagnostics: 0
- Round contexts filled from existing optimizer EV: 279

## Remaining no-EV-source examples

- Christoph Baumgartner | AUT | MID | 3500000
- Becir Omeragic | SUI | DEF | 2000000
- Cedric Zesiger | SUI | DEF | 2000000
- Isaac Schmidt | SUI | DEF | 2000000
- Adrian Bajrami | SUI | DEF | 2000000
- Lucas Blondel | SUI | DEF | 2000000
- David Zima | CZE | DEF | 2000000
- Nuno Tavares | POR | DEF | 3500000
- Antonio Silva | POR | DEF | 3500000
- Caglar Soyuncu | TUR | DEF | 2500000
- David Affengruber | AUT | DEF | 2500000
- Michael Svoboda | AUT | DEF | 2500000
- Ivan Smolcic | CRO | DEF | 2000000
- Maximilian Mittelstädt | GER | DEF | 3500000
- Robin Koch | GER | DEF | 3500000
- Pierre Kalulu | FRA | DEF | 3000000
- Dean Donny Huijsen | ESP | DEF | 4000000
- Robin Le Normand | ESP | DEF | 4500000
- Daniel Vivian | ESP | DEF | 3500000
- Victor Eriksson | SWE | DEF | 2000000
- Mohamed Amine Tougai | ALG | DEF | 2000000
- Jaouen Hadjam | ALG | DEF | 2500000
- Samir Chergui | ALG | DEF | 2000000
- Lisandro Martinez | ARG | DEF | 3000000
- Facundo Medina | ARG | DEF | 2500000
- ... plus 255 flere i CSV.

## Remaining round-context examples


## Notes

- Kounde og Neuer har stadig ingen EV-kilde i eksisterende mellemdata; deres fixtures kan findes, men spiller-EV maa genbygges upstream.
- Raphinha, Wesley Franca og Mahmoud Trezeguet havde optimizer_ev uden runde-context; runde-context er udfyldt fra eksisterende EV og fixture-multipliers.
- HOLDET_584 er canonicaliseret til CZE i player_pool og EV-output.
