# User Strategy Detail Report

Ingen modeldata er ændret af denne rapport. Baumgartner-skadeinfo er kun markeret som manuel note.

## Aktuel Strategikontekst

- Target round: 1 (runde 1)
- Display: Næste runde (runde 1)

## Strategioversigt

| Strategi | Formation | Pris | EV | Score | Avg cond | High risk | Kaptajn | Kaptajn vækst | Vurdering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Næste runde (runde 1) | 3-4-3 | 49.5 | 42.438 | 90.126 | 0.8441 | 1 | Cristiano Ronaldo | 2 | rimelig starter-sikkerhed; 1 high_risk |
| 1. + 2. runde | 3-4-3 | 49.0 | 43.676 | 107.522 | 0.8488 | 1 | Cristiano Ronaldo | 2 | rimelig starter-sikkerhed; 1 high_risk |
| Gruppespil | 3-4-3 | 48.5 | 44.223 | 120.934 | 0.8412 | 2 | Cristiano Ronaldo | 2 | rimelig starter-sikkerhed; 2 high_risk; underudnytter budget lidt; inkluderer runde 3-rotation |
| Lang sigt | 3-5-2 | 50.0 | 34.264 | 65.949 | 0.878 | 0 | Mikel Oyarzabal | 1.348 | høj starter-sikkerhed; orienteret mod stærkere turneringsnationer |

## Kaptajn-tjek

Kaptajn beregnes nu med separat kaptajnscore: forventet rundevaekst, start-sikkerhed, high_risk-penalty, kampfavorit og manuel doedbold/straffeprofil. `manual_captain_status=avoid` blokerer kun kaptajnvalg, ikke spillerudtagelse.

TODO: Tilfoej national_goal_rate, recent_goal_rate og et egentligt set_piece_takers-lag, saa kaptajnscore ikke behoever at bruge positions-/rolleproxy.



| Spiller | Hold | Pos | Total EV | R1 captain growth | R1 weighted EV | Manuel note |
| --- | --- | --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | MID | 4.574 | 3.152 | 3.152 | BRUGERNOTE 2026-06-03: meldt skadet/ikke længere udtaget til VM; bør undgås indtil data er opdateret. |
| Roberto Alvarado | MEX | MID | 4.126 | 2.551 | 2.551 |  |
| Raul Jimenez | MEX | FWD | 4.993 | 2.421 | 2.421 |  |
| Cristiano Ronaldo | POR | FWD | 4.947 | 2 | 2 |  |
| Diogo Costa | POR | GK | 3.56 | 1.769 | 1.769 |  |
| Vinicius Junior | BRA | FWD | 5.045 | 1.692 | 1.692 |  |
| Kerem Akturkoglu | TUR | MID | 4.078 | 1.672 | 1.672 |  |

## Næste runde (runde 1)

| Spiller | Hold | Pos | Pris | EV | Score | Start | Cond | Risk | Runder/modstandere | Win prob | CS prob | Goal/assist | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.974216 | 8.104507 | 0.7657 | 0.8431 | medium_risk | R1:COD | R1:0.734 | R1:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Stefan Posch | AUT | DEF | 3000000 | 2.655883 | 7.216075 | 0.7042 | 0.7586 | medium_risk | R1:JOR | R1:0.708 | R1:0.486 |  | favoritkamp i relevant horisont |
| Ruben Dias | POR | DEF | 4000000 | 2.812349 | 5.760249 | 0.8077 | 0.8506 | low_risk | R1:COD | R1:0.734 | R1:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.947107 | 9.548111 | 0.908 | 0.97 | medium_risk | R1:COD | R1:0.734 |  | R1: goal 1.35 / assist 1.25 | stærk startsikkerhed; høj EV; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Raul Jimenez | MEX | FWD | 4500000 | 4.992947 | 9.30221 | 0.7012 | 0.8154 | high_risk | R1:RSA | R1:0.645 |  | R1: goal 1.35 / assist 1.25 | manuel tjek: high_risk; høj EV; godt offensivt kampmiljø |
| Lautaro Martinez | ARG | FWD | 8000000 | 4.692971 | 8.45776 | 0.8498 | 0.9059 | medium_risk | R1:ALG | R1:0.656 |  | R1: goal 1.35 / assist 1.25 | stærk startsikkerhed; høj EV; godt offensivt kampmiljø |
| Diogo Costa | POR | GK | 5000000 | 3.560318 | 7.489855 | 0.7534 | 0.7925 | low_risk | R1:COD | R1:0.734 | R1:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Roberto Alvarado | MEX | MID | 2500000 | 4.125843 | 9.187823 | 0.6394 | 0.7246 | medium_risk | R1:RSA | R1:0.645 |  | R1: goal 1.35 / assist 1.25 | captain_avoid: maa ikke anbefales som kaptajn; manuel tjek: lav conditional start; godt offensivt kampmiljø |
| Oscar Bobb | NOR | MID | 3000000 | 3.563792 | 8.656539 | 0.9101 | 0.97 | medium_risk | R1:IRQ | R1:0.758 |  | R1: goal 1.35 / assist 1.25 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.967413 | 8.571763 | 0.6607 | 0.7143 | medium_risk | R1:IRQ | R1:0.758 |  | R1: goal 1.35 / assist 1.25 | manuel tjek: lav conditional start; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Scott McTominay | SCO | MID | 4500000 | 3.145203 | 7.830847 | 0.8738 | 0.9403 | medium_risk | R1:HAI | R1:0.661 |  | R1: goal 1.35 / assist 1.25 | stærk startsikkerhed; godt offensivt kampmiljø |

## 1. + 2. runde

| Spiller | Hold | Pos | Pris | EV | Score | Start | Cond | Risk | Runder/modstandere | Win prob | CS prob | Goal/assist | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.974216 | 10.191963 | 0.7657 | 0.8431 | medium_risk | R1:COD; R2:UZB | R1:0.734; R2:0.743 | R1:0.5; R2:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Chris Richards | USA | DEF | 3000000 | 3.127363 | 7.523092 | 0.6712 | 0.7805 | high_risk | R1:PAR; R2:AUS | R1:0.476; R2:0.545 | R1:0.411; R2:0.423 |  | manuel tjek: high_risk |
| Ruben Dias | POR | DEF | 4000000 | 2.812349 | 6.867158 | 0.8077 | 0.8506 | low_risk | R1:COD; R2:UZB | R1:0.734; R2:0.743 | R1:0.5; R2:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Jonathan David | CAN | FWD | 4500000 | 4.474384 | 11.605076 | 0.9 | 0.97 | medium_risk | R1:BIH; R2:QAT | R1:0.522; R2:0.671 |  | R1: goal 1.208 / assist 1.149; R2: goal 1.35 / assist 1.25 | stærk startsikkerhed; godt offensivt kampmiljø |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.947107 | 11.526874 | 0.908 | 0.97 | medium_risk | R1:COD; R2:UZB | R1:0.734; R2:0.743 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25 | stærk startsikkerhed; høj EV; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Vinicius Junior | BRA | FWD | 7500000 | 5.044969 | 10.51014 | 0.7444 | 0.8103 | medium_risk | R1:MAR; R2:HAI | R1:0.586; R2:0.897 |  | R1: goal 1.288 / assist 1.206; R2: goal 1.35 / assist 1.25 | høj EV; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Diogo Costa | POR | GK | 5000000 | 3.560318 | 9.422139 | 0.7534 | 0.7925 | low_risk | R1:COD; R2:UZB | R1:0.734; R2:0.743 | R1:0.5; R2:0.5 |  | favoritkamp i relevant horisont; stærk clean sheet-profil |
| Roberto Alvarado | MEX | MID | 2500000 | 4.125843 | 11.113235 | 0.6394 | 0.7246 | medium_risk | R1:RSA; R2:KOR | R1:0.645; R2:0.529 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.228 / assist 1.163 | captain_avoid: maa ikke anbefales som kaptajn; manuel tjek: lav conditional start; godt offensivt kampmiljø |
| Oscar Bobb | NOR | MID | 3000000 | 3.563792 | 9.780859 | 0.9101 | 0.97 | medium_risk | R1:IRQ; R2:SEN | R1:0.758; R2:0.458 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.136 / assist 1.097 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.967413 | 9.773094 | 0.6607 | 0.7143 | medium_risk | R1:IRQ; R2:SEN | R1:0.758; R2:0.458 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.136 / assist 1.097 | manuel tjek: lav conditional start; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Kerem Akturkoglu | TUR | MID | 4500000 | 4.077992 | 9.208221 | 0.8753 | 0.9107 | low_risk | R1:AUS; R2:PAR | R1:0.518; R2:0.422 |  | R1: goal 1.211 / assist 1.151; R2: goal 1.102 / assist 1.073 | stærk startsikkerhed |

## Gruppespil

| Spiller | Hold | Pos | Pris | EV | Score | Start | Cond | Risk | Runder/modstandere | Win prob | CS prob | Goal/assist | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.974216 | 10.954704 | 0.7657 | 0.8431 | medium_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 | R1:0.5; R2:0.5; R3:0.355 |  | favoritkamp i relevant horisont; stærk clean sheet-profil; runde 3-rotation proxy p6=0.545 |
| Chris Richards | USA | DEF | 3000000 | 3.127363 | 8.568773 | 0.6712 | 0.7805 | high_risk | R1:PAR; R2:AUS; R3:TUR | R1:0.476; R2:0.545; R3:0.371 | R1:0.411; R2:0.423; R3:0.277 |  | manuel tjek: high_risk |
| Josko Gvardiol | CRO | DEF | 3500000 | 2.840621 | 8.288236 | 0.8562 | 0.9216 | medium_risk | R1:ENG; R2:PAN; R3:GHA | R1:0.204; R2:0.619; R3:0.546 | R1:0.208; R2:0.486; R3:0.442 |  | stærk startsikkerhed |
| Raul Jimenez | MEX | FWD | 4500000 | 4.992947 | 12.796656 | 0.7012 | 0.8154 | high_risk | R1:RSA; R2:KOR; R3:CZE | R1:0.645; R2:0.529; R3:0.484 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.228 / assist 1.163; R3: goal 1.172 / assist 1.123 | manuel tjek: high_risk; høj EV; godt offensivt kampmiljø |
| Cristiano Ronaldo | POR | FWD | 7000000 | 4.947107 | 12.470983 | 0.908 | 0.97 | medium_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25; R3: goal 1.115 / assist 1.082 | stærk startsikkerhed; høj EV; favoritkamp i relevant horisont; godt offensivt kampmiljø; runde 3-rotation proxy p6=0.545 |
| Vinicius Junior | BRA | FWD | 7500000 | 5.044969 | 12.237349 | 0.7444 | 0.8103 | medium_risk | R1:MAR; R2:HAI; R3:SCO | R1:0.586; R2:0.897; R3:0.669 |  | R1: goal 1.288 / assist 1.206; R2: goal 1.35 / assist 1.25; R3: goal 1.35 / assist 1.25 | høj EV; favoritkamp i relevant horisont; godt offensivt kampmiljø; runde 3-rotation proxy p6=0.526 |
| Diogo Costa | POR | GK | 5000000 | 3.560318 | 10.086241 | 0.7534 | 0.7925 | low_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 | R1:0.5; R2:0.5; R3:0.355 |  | favoritkamp i relevant horisont; stærk clean sheet-profil; runde 3-rotation proxy p6=0.545 |
| Roberto Alvarado | MEX | MID | 2500000 | 4.125843 | 13.0122 | 0.6394 | 0.7246 | medium_risk | R1:RSA; R2:KOR; R3:CZE | R1:0.645; R2:0.529; R3:0.484 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.228 / assist 1.163; R3: goal 1.172 / assist 1.123 | captain_avoid: maa ikke anbefales som kaptajn; manuel tjek: lav conditional start; godt offensivt kampmiljø |
| Kerem Akturkoglu | TUR | MID | 4500000 | 4.077992 | 11.181498 | 0.8753 | 0.9107 | low_risk | R1:AUS; R2:PAR; R3:USA | R1:0.518; R2:0.422; R3:0.357 |  | R1: goal 1.211 / assist 1.151; R2: goal 1.102 / assist 1.073; R3: goal 0.99 / assist 0.993 | stærk startsikkerhed |
| Andreas Schjelderup | NOR | MID | 3500000 | 3.967413 | 10.712564 | 0.6607 | 0.7143 | medium_risk | R1:IRQ; R2:SEN; R3:FRA | R1:0.758; R2:0.458; R3:0.197 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.136 / assist 1.097; R3: goal 0.753 / assist 0.823 | manuel tjek: lav conditional start; favoritkamp i relevant horisont; godt offensivt kampmiljø |
| Oscar Bobb | NOR | MID | 3000000 | 3.563792 | 10.624446 | 0.9101 | 0.97 | medium_risk | R1:IRQ; R2:SEN; R3:FRA | R1:0.758; R2:0.458; R3:0.197 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.136 / assist 1.097; R3: goal 0.753 / assist 0.823 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø |

## Lang sigt

| Spiller | Hold | Pos | Pris | EV | Score | Start | Cond | Risk | Runder/modstandere | Win prob | CS prob | Goal/assist | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nuno Mendes | POR | DEF | 4500000 | 3.974216 | 5.92276 | 0.7657 | 0.8431 | medium_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 | R1:0.5; R2:0.5; R3:0.355 |  | favoritkamp i relevant horisont; stærk clean sheet-profil; valgt i lang sigt-kontekst |
| Aymeric Laporte | ESP | DEF | 4500000 | 1.647224 | 5.247334 | 0.822 | 0.902 | medium_risk | R1:CPV; R2:KSA; R3:URU | R1:0.87; R2:0.848; R3:0.577 | R1:0.625; R2:0.663; R3:0.423 |  | stærk startsikkerhed; favoritkamp i relevant horisont; stærk clean sheet-profil; valgt i lang sigt-kontekst |
| Ruben Dias | POR | DEF | 4000000 | 2.812349 | 5.221486 | 0.8077 | 0.8506 | low_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 | R1:0.5; R2:0.5; R3:0.355 |  | favoritkamp i relevant horisont; stærk clean sheet-profil; valgt i lang sigt-kontekst |
| Mikel Oyarzabal | ESP | FWD | 7500000 | 4.133042 | 7.608861 | 0.8478 | 0.9432 | medium_risk | R1:CPV; R2:KSA; R3:URU | R1:0.87; R2:0.848; R3:0.577 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25; R3: goal 1.273 / assist 1.195 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Nico Williams | ESP | FWD | 4500000 | 3.744479 | 7.239726 | 0.8158 | 0.9091 | medium_risk | R1:CPV; R2:KSA; R3:URU | R1:0.87; R2:0.848; R3:0.577 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25; R3: goal 1.273 / assist 1.195 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Diogo Costa | POR | GK | 5000000 | 3.560318 | 5.621557 | 0.7534 | 0.7925 | low_risk | R1:COD; R2:UZB; R3:COL | R1:0.734; R2:0.743; R3:0.436 | R1:0.5; R2:0.5; R3:0.355 |  | favoritkamp i relevant horisont; stærk clean sheet-profil; valgt i lang sigt-kontekst |
| Declan Rice | ENG | MID | 4500000 | 2.764994 | 5.896083 | 0.8465 | 0.8706 | low_risk | R1:CRO; R2:GHA; R3:PAN | R1:0.543; R2:0.709; R3:0.763 |  | R1: goal 1.237 / assist 1.17; R2: goal 1.35 / assist 1.25; R3: goal 1.35 / assist 1.25 | favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Giovani Lo Celso | ARG | MID | 4000000 | 3.082398 | 5.838152 | 0.7765 | 0.8553 | medium_risk | R1:ALG; R2:AUT; R3:JOR | R1:0.656; R2:0.571; R3:0.801 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.276 / assist 1.197; R3: goal 1.35 / assist 1.25 | favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Rodrigo de Paul | ARG | MID | 4500000 | 2.957481 | 5.811481 | 0.9318 | 0.9545 | low_risk | R1:ALG; R2:AUT; R3:JOR | R1:0.656; R2:0.571; R3:0.801 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.276 / assist 1.197; R3: goal 1.35 / assist 1.25 | stærk startsikkerhed; favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Manu Koné | FRA | MID | 3500000 | 2.799984 | 5.776565 | 0.7771 | 0.8571 | medium_risk | R1:SEN; R2:IRQ; R3:NOR | R1:0.652; R2:0.842; R3:0.55 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25; R3: goal 1.247 / assist 1.177 | favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |
| Aurelien Tchouameni | FRA | MID | 3500000 | 2.787585 | 5.764786 | 0.8272 | 0.88 | medium_risk | R1:SEN; R2:IRQ; R3:NOR | R1:0.652; R2:0.842; R3:0.55 |  | R1: goal 1.35 / assist 1.25; R2: goal 1.35 / assist 1.25; R3: goal 1.247 / assist 1.177 | favoritkamp i relevant horisont; godt offensivt kampmiljø; valgt i lang sigt-kontekst |

## Spillere Der Bør Tjekkes Manuelt

- Raul Jimenez (MEX, FWD): manuel tjek: high_risk; høj EV; godt offensivt kampmiljø
- Roberto Alvarado (MEX, MID): captain_avoid: maa ikke anbefales som kaptajn; manuel tjek: lav conditional start; godt offensivt kampmiljø
- Andreas Schjelderup (NOR, MID): manuel tjek: lav conditional start; favoritkamp i relevant horisont; godt offensivt kampmiljø
- Chris Richards (USA, DEF): manuel tjek: high_risk

## Inputfiler

- `data\optimal_squads_by_strategy.json`
- `data\strategy_comparison_report.csv`
- `data\strategy_sanity_report.md`
- `data\player_ev_group_stage_v1.csv`
- `data\player_pool_v1.json`
- `data\fixture_strength_multipliers.csv`
- `data\match_odds_probs.csv`
- `data\current_strategy_context.json`
