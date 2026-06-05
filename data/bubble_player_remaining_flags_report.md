# Remaining Bubble Flags Report

Ren klassifikation af de resterende bubble-flags. Ingen optimizer-, EV- eller frontend-output er genberegnet.

## Fordeling

| primary_flag_category | count |
|---|---:|
| missing_or_weak_round_context | 10 |
| uncertain_start | 3 |
| availability_risk | 1 |
| likely_overweighted_by_model | 1 |
| likely_underweighted_by_model | 1 |
| missing_ev_source | 1 |

## Fokusspillere

| player_name | team | position | primary_flag_category | start_prob | optimizer_ev | recommended_next_action |
| --- | --- | --- | --- | --- | --- | --- |
| Scott McTominay | SCO | MID | likely_overweighted_by_model | 0.8818 | 3.193 | Sammenlign marginal budgetbrug mod premium FWD-upside foer kalibrering. |
| Mahmoud Trezeguet | EGY | MID | missing_or_weak_round_context | 0.8438 | 1.2443 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Jurrien Timber | NED | DEF | missing_or_weak_round_context | 0.6333 | 2.0551 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Wesley Franca | BRA | DEF | missing_or_weak_round_context | 0.6273 | 2.7262 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Raphinha | BRA | FWD | missing_or_weak_round_context | 0.7752 | 2.2639 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Deniz Undav | GER | FWD | missing_or_weak_round_context | 0.3483 | 2.0966 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Michael Olise | FRA | FWD | likely_underweighted_by_model | 0.7361 | 3.305 | Brug positional budget-auditten foer eventuel senere modelaendring. |
| Jamal Musiala | GER | MID | missing_or_weak_round_context | 0.6787 | 2.9434 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |
| Manuel Neuer | GER | GK | missing_or_weak_round_context | 0.2376 | 2.2844 | Audit round EV og fixturefordeling foer eventuel modelkalibrering. |

## Alle flags

| player_name | team | position | primary_flag_category | suspected_issue |
| --- | --- | --- | --- | --- |
| Christoph Baumgartner | AUT | MID | missing_ev_source | Reel datamangel: spilleren har ingen brugbar EV-/fixturevaerdi i modeloutputtet. |
| Antonio Nusa | NOR | MID | uncertain_start | Reel spillerusikkerhed: startchance er ikke sikker nok til at behandle spilleren som fast starter. |
| Scott McTominay | SCO | MID | likely_overweighted_by_model | Mulig modelvaegtning: sikkerhed/value eller central MID/DEF-score kan fylde for meget. |
| Mahmoud Trezeguet | EGY | MID | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Jurrien Timber | NED | DEF | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Wesley Franca | BRA | DEF | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Ismael Saibari | MAR | MID | availability_risk | Reel spillerusikkerhed: availability/injury-risk er selve problemet, ikke primaert scoremodellen. |
| Raphinha | BRA | FWD | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Gregor Kobel | SUI | GK | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Ismaila Sarr | SEN | MID | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Aleksandar Pavlovic | GER | MID | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Patrick Wimmer | AUT | MID | uncertain_start | Reel spillerusikkerhed: startchance er ikke sikker nok til at behandle spilleren som fast starter. |
| Deniz Undav | GER | FWD | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Michael Olise | FRA | FWD | likely_underweighted_by_model | Mulig modelvaegtning: premium/offensiv upside kan vaere for lavt vaegtet ift. maal, flere maal og captain-ceiling. |
| Jamal Musiala | GER | MID | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Manuel Neuer | GER | GK | missing_or_weak_round_context | Reel eller delvis datamangel: aggregate/modelscore findes, men rundekonteksten er svag eller ikke i samme skala. |
| Andreas Schjelderup | NOR | MID | uncertain_start | Reel spillerusikkerhed: startchance er ikke sikker nok til at behandle spilleren som fast starter. |
