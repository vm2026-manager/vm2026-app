# Missing Player Component Source Experiment

## Metode

- Referencepopulation: 682 spillere med komplette komponenter og dokumenterede offensive shares.
- Shares estimeres med shrinkage mellem samme hold/position og samme position/startniveau.
- Dokumenterede shares anvendes direkte, når de findes.
- Offensive komponentrater estimeres robust fra samme holds komplette spillere med positionsfallback.
- Clean sheet, resultat, team-score, opponent-score og on-pitch følger de eksisterende generelle modelregler.
- Samlede nye shares skaleres mod et konservativt holdloft på 0.90. Pris bruges kun til prioritering, aldrig som performance-input.
- De 12 usikre Holdet-rebase-identiteter får ingen estimeret EV.

## Leave-One-Out Validation

- Spillere testet: 682
- Median absolut fejl: 0.1729
- Gennemsnitlig absolut fejl: 0.3468
- Median relativ fejl: 13.8%
- Afvigelse over 25%: 195
- Afvigelse over 50%: 56
- Afvigelse over 100%: 17

### Fejl pr. position

| position | players | median_absolute_error | mean_absolute_error | median_signed_error | median_relative_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEF | 239 | 0.0916 | 0.1445 | -0.0442 | 0.0522 | 20 | 2 | 0 |
| FWD | 160 | 0.3723 | 0.5434 | -0.0297 | 0.1821 | 62 | 24 | 8 |
| GK | 37 | 0.1221 | 0.1353 | -0.0978 | 0.133 | 8 | 0 | 0 |
| MID | 246 | 0.3073 | 0.4472 | -0.101 | 0.2153 | 105 | 30 | 9 |

### Fejl pr. startniveau

| start_band | players | median_absolute_error | mean_absolute_error | median_signed_error | median_relative_error | over_25_pct | over_50_pct | over_100_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contender | 239 | 0.1355 | 0.2844 | -0.0304 | 0.0879 | 60 | 17 | 5 |
| likely_starter | 252 | 0.2049 | 0.4078 | -0.0675 | 0.1302 | 69 | 20 | 5 |
| reserve | 24 | 0.1259 | 0.1285 | -0.1223 | 0.3223 | 17 | 3 | 1 |
| rotation | 79 | 0.1258 | 0.181 | -0.0767 | 0.1439 | 22 | 8 | 3 |
| strong_starter | 88 | 0.3743 | 0.5497 | -0.2306 | 0.162 | 27 | 8 | 3 |

Positiv median signed error betyder systematisk overvurdering; negativ betyder undervurdering.

Positionsbias: DEF under 0.044, FWD under 0.030, GK under 0.098, MID under 0.101.

Startniveau-bias: contender under 0.030, likely_starter under 0.068, reserve under 0.122, rotation under 0.077, strong_starter under 0.231. Metoden undervurderer især `strong_starter` i absolut EV og har den højeste relative fejl for `reserve`.

## Sikkerhed

- `high`: 69
- `low`: 14
- `blocked`: 12
- `medium`: 8

Hyppigste advarsler:

- `small_team_position_reference`: 16
- `unsafe_rebase_identity`: 12
- `no_team_position_reference`: 6
- `assist_team_share_cap_applied`: 6
- `goal_team_share_cap_applied`: 3
- `negative_base_ev_requires_zero_floor`: 3

### Høj sikkerhed

Neymar Jr., Romelu Lukaku, Mateo Kovacic, Bradley Barcola, Ben Doak, In-beom Hwang, Matthew Garbett, Maxence Lacroix, Ahmed Zizo, Micky van de Ven, Mehdi Ghayedi, Jules Kounde, Lachlan Bayliss, Mostafa Ziko, Pape Alassane Gueye, Assane Diao Diaoune, Oston Urunov, Issa Diop, Fredrik Aursnes, Nikola Katic, Jean-Philippe Mateta, Victor Lindelöf, Kobbie Mainoo, Goncalo Guedes, Habib Diarra, Armando Obispo, Jürgen Locadia, Aaron Hickey, Nathan Ake, Bamba Dieng, Samu Costa, Gaël Kakuta, Daniel Svensson, Nadiem Amiri, Takehiro Tomiyasu, Jamie Leweling, Gilson Benchimol Tavares, Jurrien Timber, Eberechi Eze, Maximilian Beier, Jens Castrop, Michael Boxall, Francis de Vries, Angelo Stiller, Ivan Toney, Gustaf Lagerbielke, Ko Itakura, Nando Pijnaker, Djed Spence, Dong-gyeong Lee, Thelo Aasgaard, Nathaniel Brown, Carl Starfelt, Nihad Mujakic, Jarell Quansah, Fredrik Andre Bjørkan, Frans Dhia Putros, Matias Fernandez-Pardo, Deveron Fonville, Lenny Joseph, Nabil 'Dunga' Emad, Bara Sapoko Ndiaye, Kevin Lenini Pina, Rocky Bushiri, Steve Kapuadi, Sondre Langås, Raed Chikhaoui, Tommy Smith, Moustapha Mbow

### Medium sikkerhed

Lucas Paqueta, Bruno Guimaraes, Hamza Abdelkarim, Logan Costa, Karim Hafez, Dominique Simon, Gi-hyuk Lee, Aqtay Abdallah

### Lav sikkerhed

Manuel Neuer, Mike Penders, Lawrence Shankland, Craig Gordon, Ross Stewart, Hyeon-Woo Jo, Robin Risser, Mark Flekken, Weverton, CJ Dos Santos, Mostafa Oufa Shobeir, Bum-Keun Song, Sander Tangvik, El Mahdi Soliman

## 12 Holdet-Rebase Identitetsproblemer

| player_name | team_id | position | start_prob | reference_method | warning_flags |
| --- | --- | --- | --- | --- | --- |
| Victor Munoz | ESP | FWD | 0.873 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Joan Garcia | ESP | GK | 0.0291 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Ibrahim Sangare | CIV | MID | 0.7036 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Oumar Diakite | CIV | FWD | 0.6966 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Yerry Mina | COL | DEF | 0.6156 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Eric Garcia | ESP | DEF | 0.5243 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Juan 'Cucho' Hernandez | COL | FWD | 0.25 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Pablo Gavi | ESP | MID | 0.25 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Carlos Andres Gomez | COL | MID | 0.4417 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Ange-Yoan Bonny | CIV | FWD | 0.25 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Willer Ditta | COL | DEF | 0.3311 | identity_review_required_no_ev_generated | unsafe_rebase_identity |
| Marc Pubill | ESP | DEF | 0.25 | identity_review_required_no_ev_generated | unsafe_rebase_identity |

## Top 25 Manglende Spillere

| player_name | team_id | position | start_prob | root_cause | reference_population_size | confidence | estimated_base_ev | warning_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Neymar Jr. | BRA | FWD | 0.8919 | ev_master_row_missing_player_stat_allocation | 4 | high | 4.6445 |  |
| Lucas Paqueta | BRA | MID | 0.7869 | ev_master_row_missing_player_stat_allocation | 1 | medium | 3.0747 | small_team_position_reference |
| Romelu Lukaku | BEL | FWD | 0.7965 | ev_master_row_missing_player_stat_allocation | 4 | high | 3.241 |  |
| Manuel Neuer | GER | GK | 0.7293 | ev_master_row_missing_player_stat_allocation | 0 | low | 3.5022 | no_team_position_reference |
| Mateo Kovacic | CRO | MID | 0.839 | ev_master_row_missing_player_stat_allocation | 7 | high | 2.3172 |  |
| Bruno Guimaraes | BRA | MID | 0.7749 | ev_master_row_missing_player_stat_allocation | 1 | medium | 3.0053 | small_team_position_reference |
| Bradley Barcola | FRA | MID | 0.7583 | ev_master_row_missing_player_stat_allocation | 5 | high | 2.3294 |  |
| Ben Doak | SCO | MID | 0.8529 | ev_master_row_missing_player_stat_allocation | 5 | high | 2.2939 |  |
| In-beom Hwang | KOR | MID | 0.897 | ev_master_row_missing_player_stat_allocation | 5 | high | 2.1843 |  |
| Matthew Garbett | NZL | MID | 0.8803 | ev_master_row_missing_player_stat_allocation | 7 | high | 0.9191 |  |
| Maxence Lacroix | FRA | DEF | 0.873 | ev_master_row_missing_player_stat_allocation | 7 | high | 3.8363 |  |
| Ahmed Zizo | EGY | MID | 0.7866 | ev_master_row_missing_player_stat_allocation | 5 | high | 1.4231 |  |
| Hamza Abdelkarim | EGY | FWD | 0.8655 | ev_master_row_missing_player_stat_allocation | 2 | medium | 3.3465 | small_team_position_reference |
| Micky van de Ven | NED | DEF | 0.7361 | historical_components_lost_in_component_rebuild | 9 | high | 2.5128 |  |
| Mehdi Ghayedi | IRN | MID | 0.7519 | ev_master_row_missing_player_stat_allocation | 4 | high | 1.4926 |  |
| Jules Kounde | FRA | DEF | 0.7661 | ev_master_row_missing_player_stat_allocation | 7 | high | 3.3521 |  |
| Lachlan Bayliss | NZL | MID | 0.8655 | ev_master_row_missing_player_stat_allocation | 7 | high | 0.8935 |  |
| Mostafa Ziko | EGY | MID | 0.8655 | ev_master_row_missing_player_stat_allocation | 5 | high | 1.603 |  |
| Pape Alassane Gueye | SEN | MID | 0.7287 | ev_master_row_missing_player_stat_allocation | 7 | high | 1.1211 | assist_team_share_cap_applied |
| Assane Diao Diaoune | SEN | FWD | 0.7493 | ev_master_row_missing_player_stat_allocation | 3 | high | 2.2214 | assist_team_share_cap_applied |
| Oston Urunov | UZB | MID | 0.7918 | historical_aggregate_only_missing_match_components | 4 | high | 0.7375 |  |
| Issa Diop | MAR | DEF | 0.8568 | ev_master_row_missing_player_stat_allocation | 5 | high | 2.8567 |  |
| Fredrik Aursnes | NOR | MID | 0.7078 | ev_master_row_missing_player_stat_allocation | 7 | high | 1.4398 |  |
| Nikola Katic | BIH | DEF | 0.7847 | historical_aggregate_only_missing_match_components | 5 | high | 1.8575 |  |
| Jean-Philippe Mateta | FRA | FWD | 0.5843 | ev_master_row_missing_player_stat_allocation | 6 | high | 2.297 |  |

## Konklusion

**egnet kun med konservativ cap/floor**

Konklusionen er baseret på leave-one-out-fejlen, ikke på om de estimerede værdier ser plausible ud enkeltvis.
