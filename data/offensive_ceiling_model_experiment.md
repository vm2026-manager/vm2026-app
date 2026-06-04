# Offensive Ceiling Model Experiment

Kontrolleret eksperiment baseret på eksisterende modeloutput. Produktionsfiler for optimizer, strategi-output, EV, player pool og frontend er ikke overskrevet.

## Variantformler

- `baseline`: uændret `optimizer.load_players()` og eksisterende scoreformler.
- `ceiling_components_only`: baseline plus beregnelige matchwinner- og hattrick-EV pr. runde. Matchwinner bruger eksisterende `match_n_goal_ev`, win/draw probability og Holdet.dk-reglerne 40.000/20.000. Hattrick bruger Poisson ud fra `match_n_goal_ev` og 100.000-reglen.
- `fwd_price_value_cap_moderate`: FWD price/value-effekt capped ved 0,75 model-growth units.
- `fwd_price_value_cap_strong`: FWD price/value-effekt capped ved 0,35 model-growth units.
- `combined`: ceiling-komponenter plus samme FWD price/value-cap.
- `player_of_the_match_ev` og `penalty_ev` er ikke beregnet, fordi outputtet mangler forsvarlig POTM-sandsynlighed og penalty attempt/miss probability.

## Variantoversigt

| variant | avg_total_score | one_fwd_low_ceiling_flags |
|---|---:|---:|
| baseline | 98.293 | 8 |
| ceiling_components_only | 99.853 | 8 |
| ceiling_components_plus_fwd_price_value_cap_moderate | 99.853 | 8 |
| ceiling_components_plus_fwd_price_value_cap_strong | 99.820 | 8 |
| fwd_price_value_cap_moderate | 98.293 | 8 |
| fwd_price_value_cap_strong | 98.285 | 8 |

## Fokus: next_round 4-5-1

| player_name | baseline_score | ceiling_components_score | fwd_price_value_cap_score | combined_score | match_winner_goal_ev | hattrick_ev | price_value_effect_baseline | round_context_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Erling Haaland | 7.879471 | 8.194336 | 7.879471 | 8.194336 | 0.142645 | 0.003617 | 0.544147 | fixture_specific_context_present |
| Kylian Mbappe | 9.015854 | 9.406512 | 9.015854 | 9.406512 | 0.235488 | 0.007345 | -0.071514 | fixture_specific_context_present |
| Harry Kane | 7.251106 | 7.553455 | 7.251106 | 7.553455 | 0.206604 | 0.005318 | 0.247151 | fixture_specific_context_present |
| Luis Diaz | 6.42541 | 6.630267 | 6.42541 | 6.630267 | 0.112697 | 0.001518 | 0.623676 | fixture_specific_context_present |
| Michael Olise | 5.573452 | 5.731622 | 5.393991 | 5.471285 | 0.098034 | 0.000597 | 0.968854 | fixture_specific_context_present |
| Jamal Musiala | 6.075286 | 6.244213 | 6.075286 | 6.244213 | 0.089706 | 0.000434 | 0.395867 | fixture_specific_context_present |
| Konrad Laimer | 7.077091 | 7.28967 | 7.077091 | 7.28967 | 0.089145 | 0.001235 | -0.294432 | fixture_specific_context_present |
| Scott McTominay | 7.478489 | 7.695032 | 7.478489 | 7.695032 | 0.084451 | 0.001561 | 0.02149 | fixture_specific_context_present |

## Spillerkonklusioner

- Haaland: ceiling-only gennemsnitlig scoreændring 0.287; combined-moderate 0.287. Round context er stadig hovedproblemet.
- Kane: ceiling-only ændring 0.348; combined-moderate 0.348. Han forbedres lidt, men er ikke primært et price/value-problem.
- Diaz: ceiling-only ændring 0.213; mangler stadig POTM/penalty/multi-ceiling output.
- Olise: ceiling-only ændring 0.172; ser fortsat som mulig ceiling-undervægtning.
- Musiala: ceiling-only ændring 0.161; påvirkes lidt, men MID-positionen gør FWD-cap irrelevant.
- Laimer: FWD-cap påvirker ham ikke direkte; hvis han stadig slår premium FWD, skyldes det round context/start/strategy score.
- McTominay: FWD-cap påvirker ham ikke direkte; hans høje next_round-score er drevet af round context, ikke price/value.

## Formation-aware floor audit

- One-FWD low-ceiling flags efter baseline: 8.
- Efter combined moderate: 8.
- Efter combined strong: 8.
- Hvis 4-5-1/5-4-1 stadig vælger lav offensiv upside, peger auditten først på round EV/absolute EV og price/value-lag. Et hard floor kan skjule problemet og bør fortsat undgås, indtil EV-komponenterne er bedre.

## Anbefalet næste produktionsvariant

Test først `ceiling_components_plus_fwd_price_value_cap_moderate` i en separat produktionspipeline-run. Den er mest kontrolleret: den tilføjer kun beregnelige Holdet.dk-ceiling-komponenter og reducerer kun ekstrem FWD price/value-effekt, uden at hardcode premiumangribere eller formation floors.
