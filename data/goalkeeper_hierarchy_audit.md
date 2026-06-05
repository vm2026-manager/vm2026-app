# Goalkeeper Hierarchy Audit

GK-startchancer normaliseres pr. land efter alle individuelle Transfermarkt-signaler og context-overrides. Kun aktive keepere indgaar i sum-normaliseringen; `holdet_is_out=True` nulstilles.

Metode: raw score = recency-weighted competitive start score, competitive start score, conditional start probability, existing start_prob, recent start rate og availability. Scores skarpes kubisk og normaliseres til cirka 1.00 pr. land med en lille reservefloor. Context-overrides loeftes som prioriteret input foer normalisering.

## Foer/efter

- Lande hvor GK start_prob-sum > 1.10 foer: 0
- Lande hvor GK start_prob-sum > 1.10 efter: 0
- Lande med mindst to GK start_prob >= 0.60 foer: 0
- Lande med mindst to GK start_prob >= 0.60 efter: 0
- Lande med mindst to GK Sandsynlig/Klar starter foer: 0
- Lande med mindst to GK Sandsynlig/Klar starter efter: 0
- Maksimal GK start_prob-sum foer: 1.0001
- Maksimal GK start_prob-sum efter: 1.0001

## Sanity-hold

| team_id | team_gk_rank | player_name | raw_start_prob | raw_start_prob_source | competitive_starts | recent_starts | availability_prob | context_override | normalized_gk_start_prob | normalized_prob_sum_team |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALG | 1 | Luca Zidane | 0.6019 | transfermarkt_availability_split_2026_06_05 | 7.0 | 6 | 0.6729 |  | 0.8224 | 1.0 |
| ALG | 2 | Alexis Guendouz | 0.3543 | transfermarkt_availability_split_2026_06_05 | 14.0 | 1 | 0.6 |  | 0.136 | 1.0 |
| ALG | 3 | Anthony Mandrea | 0.1349 | transfermarkt_availability_split_2026_06_05 | 21.0 | 1 | 0.6 |  | 0.0416 | 1.0 |
| ARG | 1 | Emiliano Martinez | 0.9156 | transfermarkt_availability_split_2026_06_05 | 59.0 | 10 | 0.8399 |  | 0.9526 | 1.0 |
| ARG | 2 | Geronimo Rulli | 0.2398 | transfermarkt_availability_split_2026_06_05 | 7.0 | 1 | 0.7117 |  | 0.0172 | 1.0 |
| ARG | 3 | Walter Benitez | 0.0439 | transfermarkt_availability_split_2026_06_05 | 1.0 | 0 | 0.6524 |  | 0.0151 | 1.0 |
| ARG | 4 | Juan Musso | 0.043 | transfermarkt_availability_split_2026_06_05 | 1.0 |  | 0.6 |  | 0.0151 | 1.0 |
| AUT | 1 | Alexander Schlager | 0.9 | transfermarkt_availability_split_2026_06_05+context_override | 26.0 | 8 | 0.95 | yes | 0.921 | 1.0001 |
| AUT | 2 | Patrick Pentz | 0.3462 | transfermarkt_availability_split_2026_06_05 | 17.0 | 4 | 0.9017 |  | 0.054 | 1.0001 |
| AUT | 3 | Nikolas Polster | 0.0473 | transfermarkt_availability_split_2026_06_05 | 0.0 |  | 0.8438 |  | 0.0251 | 1.0001 |
| ESP | 1 | Unai Simon | 0.4608 | transfermarkt_availability_split_2026_06_05 | 57.0 | 11 | 0.9067 |  | 0.8605 | 0.9999 |
| ESP | 2 | David Raya | 0.2926 | transfermarkt_availability_split_2026_06_05 | 11.0 | 1 | 0.8901 |  | 0.0991 | 0.9999 |
| ESP | 3 | Joan Garcia | 0.15 | transfermarkt_availability_split_2026_06_05+context_override | 1.0 |  | 0.9 | yes | 0.0403 | 0.9999 |
| FRA | 1 | Mike Maignan | 0.82 | transfermarkt_availability_split_2026_06_05+context_override | 38.0 | 10 | 0.9 | yes | 0.9407 | 1.0 |
| FRA | 2 | Brice Samba | 0.2581 | transfermarkt_availability_split_2026_06_05 | 4.0 | 1 | 0.9557 |  | 0.0342 | 1.0 |
| FRA | 3 | Robin Risser | 0.0456 | transfermarkt_availability_split_2026_06_05 | 0.0 |  | 0.75 |  | 0.0251 | 1.0 |
| GER | 1 | Oliver Baumann | 0.5797 | transfermarkt_availability_split_2026_06_05 | 12.0 | 9 | 0.9554 |  | 0.5443 | 1.0001 |
| GER | 2 | Manuel Neuer | 0.8349 | transfermarkt_availability_split_2026_06_05 | 123.0 | 0 | 0.6034 |  | 0.3702 | 1.0001 |
| GER | 3 | Alexander Nübel | 0.2719 | transfermarkt_availability_split_2026_06_05 | 3.0 | 1 | 0.9359 |  | 0.0554 | 1.0001 |
| GER | 4 | Finn Dahmen | 0.048 | transfermarkt_availability_split_2026_06_05 | 0.0 |  | 0.8879 |  | 0.0151 | 1.0001 |
| GER | 5 | Jonas Urbig | 0.0456 | transfermarkt_availability_split_2026_06_05 | 0.0 |  | 0.75 |  | 0.0151 | 1.0001 |
| SUI | 1 | Gregor Kobel | 0.92 | transfermarkt_availability_split_2026_06_05+context_override | 20.0 | 10 | 0.97 | yes | 0.9174 | 1.0 |
| SUI | 2 | Yvon Mvogo | 0.4983 | transfermarkt_availability_split_2026_06_05 | 12.0 | 2 | 0.8504 |  | 0.0524 | 1.0 |
| SUI | 3 | Marvin Keller | 0.0485 | transfermarkt_availability_split_2026_06_05 | 0.0 | 0 | 0.915 |  | 0.0151 | 1.0 |
| SUI | 4 | Pascal Loretz | 0.0442 | transfermarkt_availability_split_2026_06_05 | 0.0 |  | 0.6704 |  | 0.0151 | 1.0 |
