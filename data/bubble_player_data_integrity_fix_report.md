# Bubble Player Data Integrity Fix Report

Ren data-/auditoprydning. Ingen optimizer, EV-kalibrering, strategi-output eller frontend er aendret.

## Summary

- Model error flags foer: 30
- Model error flags efter: 30
- Matchede spillere efter: 49/49
- Hoj-sikkerhedsrettelser udfoert: audit-input for Harry Kane og Deniz Undav, canonical Holdet team alias for Tjekkiet i auditlaget, og mere praecis nul-EV/fixture-diagnose.
- Ikke rettet her: reelle mangler i eksisterende EV-/fixture-output. De kraever upstream data-/EV-genbygning, ikke manuel EV-redigering.

## Problemgennemgang

| spiller | oprindeligt problem | rodaarsag | aendret fil/script | resultat efter rettelse | resterende usikkerhed |
| --- | --- | --- | --- | --- | --- |
| Ladislav Krejci | Audit matchede HOLDET_584 i stedet for CZE | Player/EV-output bruger stadig Holdet-team-id, mens eksisterende Holdet-diff/import-scripts har alias HOLDET_584 -> CZE | tools/write_bubble_player_audit.py | Audit viser nu team_id CZE via eksisterende TEAM_ALIASES | matched_player_id er stadig ladislav_krejci__holdet_584 og EV/fixture er 0 i nuvaerende modeldata; boer rettes upstream og genbygges senere |
| Vladimir Coufal | Audit matchede HOLDET_584 i stedet for CZE | Samme som Krejci | tools/write_bubble_player_audit.py | Audit viser nu team_id CZE via eksisterende TEAM_ALIASES | matched_player_id er stadig vladimir_coufal__holdet_584 og EV/fixture er 0 i nuvaerende modeldata; boer rettes upstream og genbygges senere |
| Harry Kane | Audit-input sagde GER, men pool/EV siger ENG | Fejl i boble-audit input, ikke i player_pool | tools/write_bubble_player_audit.py | Team mismatch fjernet; Kane matches stabilt som ENG | Stadig flagget for lav runde 1-score relativt til brugerens runde 1-note |
| Jules Kounde | Nul EV og nul fixture-vaerdier | Viser sig at ligge i eksisterende player_ev_group_stage_v1.csv/modeldata: optimizer_ev og alle match_EV-felter er 0/tomme | Ingen datarettelse; kun auditdiagnose praeciseret i tools/write_bubble_player_audit.py | Flagget er nu ev_and_fixture_values_missing_in_model_data | Kraever upstream EV/fixture-match rettelse og senere genkoersel; EV er ikke manuelt aendret |
| Manuel Neuer | Nul EV og nul fixture-vaerdier | Viser sig at ligge i eksisterende player_ev_group_stage_v1.csv/modeldata: optimizer_ev og alle match_EV-felter er 0/tomme | Ingen datarettelse; kun auditdiagnose praeciseret i tools/write_bubble_player_audit.py | Flagget er nu ev_and_fixture_values_missing_in_model_data | Kraever upstream EV/fixture-match rettelse og senere genkoersel; EV er ikke manuelt aendret |
| Jurrien Timber | Runde 2-3 note, men lav relevant runde-EV | Modeldata har faktisk meget lave per-runde EV-felter; ikke audit-matchfejl | tools/write_bubble_player_audit.py | Fortsat flagget, men som lav/ikke-understoettet runde-EV og startcheck | Kan vaere forventet hvis spilleren er rotation/lav rolle; kraever manuel startcheck |
| Wesley Franca | Runde 2-3 note, men nul per-runde EV | optimizer_ev findes, men round fixture context mangler i EV-output | tools/write_bubble_player_audit.py | Flagget er nu round_fixture_context_missing_but_optimizer_ev_present | Kraever upstream fixture/round-context rettelse hvis spilleren skal vurderes per runde |
| Raphinha | Runde 2-3 note, men nul per-runde EV | optimizer_ev findes, men round fixture context mangler i EV-output | tools/write_bubble_player_audit.py | Flagget er nu round_fixture_context_missing_but_optimizer_ev_present | Kraever upstream fixture/round-context rettelse; ikke en rankingkalibrering endnu |
| Mahmoud Trezeguet | Runde 2 note, men nul per-runde EV | optimizer_ev findes, men round fixture context mangler i EV-output | tools/write_bubble_player_audit.py | Flagget er nu round_fixture_context_missing_but_optimizer_ev_present | Kraever upstream fixture/round-context rettelse hvis runde 2-vurdering skal bruges |
| Deniz Undav | Prismismatch 3.0m vs 3.5m | Boble-audit input brugte 3.0m; autoritativ Holdet/player_pool pris er 3.5m | tools/write_bubble_player_audit.py | Prismismatch fjernet; pris er nu 3.500.000 | Stadig flagget for usikker start og lav next_round_score ift. runde 1-note |

## Efter rettelse: plausibilitet

- Kounde: Ikke plausibel endnu; reelt manglende EV/fixture-output.
- Neuer: Ikke plausibel endnu; reelt manglende EV/fixture-output.
- Krejci: Land/hold vises plausibelt som CZE i audit, men underliggende player_id/EV-output er stadig ikke-kanonisk.
- Coufal: Land/hold vises plausibelt som CZE i audit, men underliggende player_id/EV-output er stadig ikke-kanonisk.
- Kane: Match er nu plausibelt som ENG; lav runde 1-score er et separat modelspoergsmaal.
- Timber: Ser plausibel som usikker/lav EV ud fra eksisterende data, men kraever manuel startkontrol.
- Wesley Franca: Ikke fuldt plausibel per runde, fordi round fixture context mangler selv om optimizer_ev findes.
- Raphinha: Ikke fuldt plausibel per runde, fordi round fixture context mangler selv om optimizer_ev findes.
- Trezeguet: Ikke fuldt plausibel per runde, fordi round fixture context mangler selv om optimizer_ev findes.
- Undav: Pris er nu plausibel; start/round-score flag er stadig et reelt reviewpunkt.

## Stadig undervurderede / overvurderede efter datarensning

Undervurderede ifoelge auditregler:

- Erling Haaland
- Luis Diaz
- Michael Olise
- Jamal Musiala

Overvurderede ifoelge auditregler:

- Konrad Laimer
- Scott McTominay

## Filer aendret

- tools/write_bubble_player_audit.py
- data/bubble_player_audit.csv
- data/bubble_player_audit.md
- data/bubble_player_data_integrity_fix_report.md

## Ikke aendret

- index.html
- data/player_pool_v1.json
- optimizerlogik
- strategi-output
- EV/modelkalibrering
