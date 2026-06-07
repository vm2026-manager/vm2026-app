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
| ALG | 1 | Luca Zidane | 0.6036 | transfermarkt_availability_split_2026_06_07 | 7.0 | 6 | 0.6803 |  | 0.8226 | 1.0 |
| ALG | 2 | Alexis Guendouz | 0.3573 | transfermarkt_availability_split_2026_06_07 | 14.0 | 1 | 0.6 |  | 0.1358 | 1.0 |
| ALG | 3 | Anthony Mandrea | 0.1367 | transfermarkt_availability_split_2026_06_07 | 21.0 | 1 | 0.6 |  | 0.0416 | 1.0 |
| ARG | 1 | Emiliano Martinez | 0.9059 | transfermarkt_availability_split_2026_06_07 | 59.0 | 10 | 0.8113 |  | 0.9429 | 1.0 |
| ARG | 2 | Juan Musso | 0.2356 | transfermarkt_availability_split_2026_06_07 | 2.0 |  | 0.6 |  | 0.0269 | 1.0 |
| ARG | 3 | Geronimo Rulli | 0.0452 | transfermarkt_availability_split_2026_06_07 | 7.0 | 1 | 0.723 |  | 0.0151 | 1.0 |
| ARG | 4 | Walter Benitez | 0.0439 | transfermarkt_availability_split_2026_06_07 | 1.0 | 0 | 0.6524 |  | 0.0151 | 1.0 |
| AUT | 1 | Alexander Schlager | 0.9 | transfermarkt_availability_split_2026_06_07+context_override | 26.0 | 8 | 0.95 | yes | 0.9212 | 1.0001 |
| AUT | 2 | Patrick Pentz | 0.3448 | transfermarkt_availability_split_2026_06_07 | 17.0 | 4 | 0.901 |  | 0.0538 | 1.0001 |
| AUT | 3 | Nikolas Polster | 0.0473 | transfermarkt_availability_split_2026_06_07 | 0.0 |  | 0.8438 |  | 0.0251 | 1.0001 |
| ESP | 1 | Unai Simon | 0.4599 | transfermarkt_availability_split_2026_06_07 | 57.0 | 11 | 0.9057 |  | 0.86 | 1.0 |
| ESP | 2 | David Raya | 0.2932 | transfermarkt_availability_split_2026_06_07 | 11.0 | 1 | 0.8892 |  | 0.0996 | 1.0 |
| ESP | 3 | Joan Garcia | 0.15 | transfermarkt_availability_split_2026_06_07+context_override | 1.0 |  | 0.9 | yes | 0.0404 | 1.0 |
| FRA | 1 | Mike Maignan | 0.82 | transfermarkt_availability_split_2026_06_07+context_override | 38.0 | 10 | 0.9 | yes | 0.9407 | 1.0001 |
| FRA | 2 | Brice Samba | 0.2584 | transfermarkt_availability_split_2026_06_07 | 4.0 | 1 | 0.9553 |  | 0.0343 | 1.0001 |
| FRA | 3 | Robin Risser | 0.0456 | transfermarkt_availability_split_2026_06_07 | 0.0 |  | 0.75 |  | 0.0251 | 1.0001 |
| GER | 1 | Oliver Baumann | 0.5908 | transfermarkt_availability_split_2026_06_07 | 13.0 | 9 | 0.9567 |  | 0.6484 | 1.0001 |
| GER | 2 | Manuel Neuer | 0.6251 | transfermarkt_availability_split_2026_06_07 | 123.0 | 0 | 0.6164 |  | 0.2645 | 1.0001 |
| GER | 3 | Alexander Nübel | 0.27 | transfermarkt_availability_split_2026_06_07 | 3.0 | 1 | 0.9384 |  | 0.057 | 1.0001 |
| GER | 4 | Finn Dahmen | 0.048 | transfermarkt_availability_split_2026_06_07 | 0.0 |  | 0.8879 |  | 0.0151 | 1.0001 |
| GER | 5 | Jonas Urbig | 0.0456 | transfermarkt_availability_split_2026_06_07 | 0.0 |  | 0.75 |  | 0.0151 | 1.0001 |
| SUI | 1 | Gregor Kobel | 0.92 | transfermarkt_availability_split_2026_06_07+context_override | 21.0 | 10 | 0.97 | yes | 0.9192 | 1.0001 |
| SUI | 2 | Yvon Mvogo | 0.4953 | transfermarkt_availability_split_2026_06_07 | 12.0 | 2 | 0.8544 |  | 0.0507 | 1.0001 |
| SUI | 3 | Marvin Keller | 0.0486 | transfermarkt_availability_split_2026_06_07 | 0.0 | 0 | 0.9206 |  | 0.0151 | 1.0001 |
| SUI | 4 | Pascal Loretz | 0.0441 | transfermarkt_availability_split_2026_06_07 | 0.0 |  | 0.6629 |  | 0.0151 | 1.0001 |
