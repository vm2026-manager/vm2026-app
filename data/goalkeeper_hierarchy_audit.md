# Goalkeeper Hierarchy Audit

Foer-data kommer fra `data\player_pool_v1.backup_before_tm_competitive_merge_20260604_222625.json`; efter-data er aktuel `player_pool_v1.json`.

Metode: raw score bruger recency-weighted competitive start score, competitive start score, conditional start probability, existing start_prob, recent start rate og availability. Scores skarpes kubisk og normaliseres til cirka 1.00 pr. land med en lille reservefloor. Positive context-overrides kan loefte en keeper; lave context-overrides capper raw score ned foer normalisering.

## Foer/efter

- Lande hvor GK start_prob-sum > 1.10 foer: 30
- Lande hvor GK start_prob-sum > 1.10 efter: 0
- Lande med mindst to GK start_prob >= 0.60 foer: 6
- Lande med mindst to GK start_prob >= 0.60 efter: 0
- Lande med mindst to GK Sandsynlig/Klar starter foer: 3
- Lande med mindst to GK Sandsynlig/Klar starter efter: 0
- Maksimal GK start_prob-sum foer: 2.3325
- Maksimal GK start_prob-sum efter: 1.0001

## Sanity-hold

| team_id | team_gk_rank | player_name | raw_start_prob | raw_start_prob_source | competitive_starts | recent_starts | availability_prob | context_override | normalized_gk_start_prob | normalized_prob_sum_team |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALG | 1 | Luca Zidane | 0.5802 | transfermarkt_availability_split_2026_06_04 | 11.0 | 6 | 0.6296 |  | 0.399 | 1.0 |
| ALG | 2 | Anthony Mandrea | 0.5826 | transfermarkt_availability_split_2026_06_04 | 41.0 | 1 | 0.6 |  | 0.3123 | 1.0 |
| ALG | 3 | Alexis Guendouz | 0.5733 | transfermarkt_availability_split_2026_06_04 | 33.0 | 1 | 0.6 |  | 0.2887 | 1.0 |
| ARG | 1 | Emiliano Martinez | 0.7354 | transfermarkt_availability_split_2026_06_04 | 101.0 | 10 | 0.6 |  | 0.7332 | 1.0 |
| ARG | 2 | Juan Musso | 0.48 | holdet_new_player_import_default |  |  | 1.0 |  | 0.1537 | 1.0 |
| ARG | 3 | Geronimo Rulli | 0.0987 | transfermarkt_availability_split_2026_06_04 | 33.0 | 1 | 0.6 |  | 0.0597 | 1.0 |
| ARG | 4 | Walter Benitez | 0.0433 | transfermarkt_availability_split_2026_06_04 | 7.0 | 0 | 0.617 |  | 0.0534 | 1.0 |
| AUT | 1 | Alexander Schlager | 0.9 | transfermarkt_availability_split_2026_06_04+context_override | 50.0 | 8 | 0.95 | yes | 0.789 | 1.0 |
| AUT | 2 | Patrick Pentz | 0.3832 | transfermarkt_availability_split_2026_06_04 | 31.0 | 4 | 0.8475 |  | 0.1599 | 1.0 |
| AUT | 3 | Nikolas Polster | 0.0433 | transfermarkt_availability_split_2026_06_04 | 2.0 |  | 0.619 |  | 0.0511 | 1.0 |
| ESP | 1 | Unai Simon | 0.8011 | transfermarkt_availability_split_2026_06_04 | 81.0 | 11 | 0.8333 |  | 0.7984 | 1.0 |
| ESP | 2 | David Raya | 0.2257 | transfermarkt_availability_split_2026_06_04 | 25.0 | 1 | 0.9375 |  | 0.1725 | 1.0 |
| ESP | 3 | Joan Garcia | 0.15 | transfermarkt_availability_split_2026_06_04+context_override | 2.0 |  | 0.9 | yes | 0.0291 | 1.0 |
| FRA | 1 | Mike Maignan | 0.82 | transfermarkt_availability_split_2026_06_04+context_override | 65.0 | 10 | 0.9 | yes | 0.8865 | 1.0001 |
| FRA | 2 | Brice Samba | 0.1023 | transfermarkt_availability_split_2026_06_04 | 15.0 | 1 | 0.92 |  | 0.0885 | 1.0001 |
| FRA | 3 | Robin Risser | 0.09 | holdet_official_unmatched_default+gk_slot_normalized_2026_05_25 |  |  |  |  | 0.0251 | 1.0001 |
| GER | 1 | Manuel Neuer | 0.8454 | transfermarkt_availability_split_2026_06_04 | 219.0 | 0 | 0.8302 |  | 0.7293 | 0.9999 |
| GER | 2 | Oliver Baumann | 0.342 | transfermarkt_availability_split_2026_06_04 | 38.0 | 9 | 0.6628 |  | 0.1776 | 0.9999 |
| GER | 3 | Alexander Nübel | 0.143 | transfermarkt_availability_split_2026_06_04 | 9.0 | 1 | 0.7297 |  | 0.0596 | 0.9999 |
| GER | 4 | Finn Dahmen | 0.48 | holdet_new_player_import_default |  |  | 1.0 |  | 0.0167 | 0.9999 |
| GER | 5 | Jonas Urbig | 0.48 | holdet_new_player_import_default |  |  | 1.0 |  | 0.0167 | 0.9999 |
| SUI | 1 | Gregor Kobel | 0.43 | transfermarkt_availability_split_2026_06_04 | 64.0 | 10 | 0.6 |  | 0.5051 | 1.0 |
| SUI | 2 | Yvon Mvogo | 0.2232 | transfermarkt_availability_split_2026_06_04 | 65.0 | 2 | 0.7465 |  | 0.2688 | 1.0 |
| SUI | 3 | Marvin Keller | 0.2314 | transfermarkt_availability_split_2026_06_04 | 10.0 | 0 | 0.6216 |  | 0.1706 | 1.0 |
| SUI | 4 | Pascal Loretz | 0.48 | holdet_new_player_import_default |  |  | 1.0 |  | 0.0555 | 1.0 |
