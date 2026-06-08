# Strategy Cleanup Report

## Brugerrettede Strategier

- next_round: Næste runde (runde 1)
- round1_2: 1. + 2. runde
- group_stage: Gruppespil
- practical_start: 1. + 2. runde
- long_run: Lang sigt

## Mapping

- round1_safe_favorite er erstattet af next_round.
- safe_starters er ikke længere en brugerrettet hovedstrategi; starter-sikkerhed indgår i alle strategier.
- fixture_attack og clean_sheet_stack indgår som komponenter via kamp-multipliers og clean sheet-data.
- balanced/debug-output er ikke længere primært strategi-output.

## Dynamisk Næste Runde

- target_round: 1
- display: Næste runde (runde 1)
- target_round beregnes som laveste grupperunde med mindst én kamp, der endnu ikke er startet.

## Kaptajn

- Kaptajn vælges pr. strategi som spilleren med højeste forventede vækst i target_round.
- Kaptajn-output skrives i strategy_comparison_report.csv og optimal_squads_by_strategy.json.

## Forberedte Inputlag

- data/confirmed_lineups.csv er oprettet som struktur til bekræftede lineups.
- data/current_squad.csv er oprettet som struktur til transfergebyr efter runde 1.
- data/manual_player_overrides.csv er oprettet som struktur til manuelle locks/check/avoid.

## TODO

- UI-visning af strategiknapper og kaptajnmarkering er ikke ændret i denne opgave.
- Transfergebyr efter runde 1 indgår i next_round, når data/current_squad.csv indeholder brugerens nuværende hold.
- confirmed_lineups påvirker endnu ikke optimizer direkte; strukturen er klar til integration.
