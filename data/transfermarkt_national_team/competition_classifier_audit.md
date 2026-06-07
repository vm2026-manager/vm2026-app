# Transfermarkt Competition Classifier Audit

## Rodårsag

Det nye CEAPI-format leverer korte turneringskoder i `competition`. Den tidligere classifier søgte kun engelske turneringsnavne i rækketeksten, så alle koder faldt gennem til `Ukendt`.

Derudover blev historiske to-cifrede årstal som `62` fortolket som 2062. Det skubbede reference-datoen ud i fremtiden og gjorde 2026-kampe kunstigt gamle.

## Før/Efter

| kategori | før | efter |
| --- | --- | --- |
| Kvalifikation | 0 | 29267 |
| Nations League | 0 | 6491 |
| Nations League-slutrunde | 0 | 1604 |
| Slutrunde | 0 | 17445 |
| Ukendt | 82245 | 0 |
| Ungdomskvalifikation | 0 | 19 |
| Ungdomsslutrunde | 0 | 94 |
| Venskabskamp | 0 | 27766 |
| Øvrig turnering | 0 | 30 |

- Ukendt før: 82245/82716 (99.43%)
- Ukendt efter: 0/82716 (0.00%)

## Konkurrencekoder

| kode | rækker | kategori | mapping |
| --- | --- | --- | --- |
| FS | 27766 | Venskabskamp | International friendly |
| WMQ6 | 5764 | Kvalifikation | World Cup qualification, UEFA |
| WMQ1 | 5211 | Kvalifikation | World Cup qualification, AFC |
| WMQ4 | 4620 | Kvalifikation | World Cup qualification, CONMEBOL |
| EMQ | 4095 | Kvalifikation | European Championship qualification |
| UNLA | 3402 | Nations League | UEFA Nations League A |
| WMQ2 | 3175 | Kvalifikation | World Cup qualification, CAF |
| AFCQ | 2985 | Kvalifikation | Africa Cup of Nations qualification |
| FIWC | 2748 | Slutrunde | FIFA World Cup |
| AFCN | 2700 | Slutrunde | Africa Cup of Nations |
| EURO | 2435 | Slutrunde | UEFA European Championship |
| WMQ3 | 2213 | Kvalifikation | World Cup qualification, CONCACAF |
| GOCU | 2040 | Slutrunde | CONCACAF Gold Cup |
| COPA | 1926 | Slutrunde | Copa America |
| UNLB | 1466 | Nations League | UEFA Nations League B |
| AFAC | 1344 | Slutrunde | AFC Asian Cup |
| ARCP | 1218 | Slutrunde | FIFA Arab Cup |
| CHAN | 1094 | Slutrunde | African Nations Championship |
| CNNF | 820 | Nations League-slutrunde | CONCACAF Nations League finals |
| UNFI | 784 | Nations League-slutrunde | UEFA Nations League finals |
| CNLA | 764 | Nations League | CONCACAF Nations League A |
| AGUC | 537 | Slutrunde | Regional senior Gold Cup |
| POWM | 511 | Kvalifikation | World Cup qualification play-off |
| EAFC | 419 | Slutrunde | EAFF Championship |
| WMQ5 | 331 | Kvalifikation | World Cup qualification, OFC |
| UNLC | 315 | Nations League | UEFA Nations League C |
| CAFA | 279 | Slutrunde | CAFA Nations Cup |
| CNLB | 276 | Nations League | CONCACAF Nations League B |
| UNPO | 192 | Nations League | UEFA Nations League play-off |
| ACQU | 176 | Kvalifikation | Asian Cup qualification |
| CONC | 168 | Slutrunde | FIFA Confederations Cup |
| CA16 | 163 | Slutrunde | Copa America Centenario |
| WAF1 | 110 | Slutrunde | WAFF Championship |
| CENC | 104 | Slutrunde | Central American Championship |
| OFCN | 96 | Slutrunde | OFC Nations Cup |
| CNLQ | 76 | Nations League | CONCACAF Nations League qualification |
| POEM | 71 | Kvalifikation | European qualification play-off |
| AFT | 64 | Slutrunde | Regional senior national-team tournament |
| CARQ | 43 | Kvalifikation | Caribbean / CONCACAF qualification |
| GCQU | 35 | Kvalifikation | Gold Cup qualification |
| TRIN | 30 | Øvrig turnering | Tri-nation / invitational tournament |
| 20WC | 21 | Ungdomsslutrunde | FIFA U20 World Cup |
| 2SAM | 21 | Ungdomsslutrunde | South American U20 Championship |
| CA17 | 18 | Ungdomsslutrunde | U17 continental championship |
| U21Q | 16 | Ungdomskvalifikation | UEFA U21 qualification |
| FARQ | 12 | Kvalifikation | Regional championship qualification |
| GCQ5 | 12 | Kvalifikation | Gold Cup qualification |
| C220 | 11 | Ungdomsslutrunde | U20 continental championship |
| CACP | 10 | Kvalifikation | Copa America Centenario qualification play-off |
| 23AF | 8 | Ungdomsslutrunde | CAF U23 Championship |
| 21EU | 6 | Ungdomsslutrunde | UEFA U21 Championship |
| A23Q | 3 | Ungdomskvalifikation | AFC U23 qualification |
| W23C | 3 | Ungdomsslutrunde | U23 World championship |
| CCPL | 3 | Kvalifikation | CONCACAF Cup qualification play-off |
| SAM2 | 3 | Ungdomsslutrunde | South American U20 Championship |
| 20AC | 3 | Ungdomsslutrunde | AFC U20 Championship |

## Top ukendte koder

(ingen)

## Antonio Rüdiger

- Recency/startsignal før: 0.5
- Recency/startsignal efter: 0.2008
- Fuld recency-historik efter: 0.6695
- Tre seneste tilgængelige observationer: 0.0
- Endeligt recency/startsignal = 70% seneste tre tilgængelige observationer + 30% fuld recency-/konkurrencehistorik.
- `injured` er udelukket fra udvælgelsesnævneren. `in squad` uden minutter er en tilgængelig nul-start-observation.

| date | competition_raw | competition_category_label | participation_state | minutes_estimate | started_estimate_clean | recency_weight |
| --- | --- | --- | --- | --- | --- | --- |
| 06/06/26 | FS | Venskabskamp | in squad |  | False | 1 |
| 31/05/26 | FS | Venskabskamp | in squad |  | False | 1 |
| 30/03/26 | FS | Venskabskamp | played | 45 | False | 0.92 |
| 27/03/26 | FS | Venskabskamp | in squad |  | False | 0.92 |
| 17/11/25 | WMQ6 | Kvalifikation | injured | 0 | False | 0.68 |
| 14/11/25 | WMQ6 | Kvalifikation | injured | 0 | False | 0.68 |
| 13/10/25 | WMQ6 | Kvalifikation | injured | 0 | False | 0.68 |
| 10/10/25 | WMQ6 | Kvalifikation | injured | 0 | False | 0.68 |
| 07/09/25 | WMQ6 | Kvalifikation | played | 82 | True | 0.68 |
| 04/09/25 | WMQ6 | Kvalifikation | played | 90 | True | 0.68 |

## Juni 2026 sanity

| spiller | dato | kode | kategori | status | start | recency_score |
| --- | --- | --- | --- | --- | --- | --- |
| Giovani Lo Celso | 07/06/26 | FS | Venskabskamp | played | True | 0.8387 |
| Juan Musso | 07/06/26 | FS | Venskabskamp | played | True | 0.256 |
| Geronimo Rulli | 07/06/26 | FS | Venskabskamp | in squad | False | 0.021 |
| Julian Alvarez | 07/06/26 | FS | Venskabskamp | not in squad | False | 0.6269 |
| Nicolas Otamendi | 07/06/26 | FS | Venskabskamp | played | True | 0.6617 |
