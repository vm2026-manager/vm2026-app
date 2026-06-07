# Recent non-starter start probability audit

## Modelændring

- `in squad` uden minutter tæller som en tilgængelig ikke-start.
- Skade og suspension er neutral utilgængelighed og indgår ikke i start-rate-nævneren.
- Conditional start-rate vægter de tre seneste tilgængelige observationer 70% og recency-vægtet historik 30%.
- Context-overrides anvendes bagefter og beholder højeste prioritet.

- Spillere med nyere start-rate højst 1/3: 672

## Rüdiger og fem øvrige sanity-spillere

| player_name | team_id | recent_available_observations | recent_available_start_rate | historical_weighted_start_rate | old_start_prob | new_start_prob | old_conditional_start_prob | new_conditional_start_prob | context_override |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Antonio Rüdiger | GER | 2026-06-06:in squad:start=0; 2026-05-31:in squad:start=0; 2026-03-30:played:start=0 | 0 | 0.7308 | 0.2168 | 0.208 | 0.229 | 0.2193 |  |
| Rodrigo de Paul | ARG | 2026-06-07:played:start=0; 2026-03-31:played:start=0; 2026-03-27:played:start=0 | 0 | 0.7526 | 0.4553 | 0.2199 | 0.4678 | 0.2258 |  |
| Lyndon Dykes | SCO | 2026-06-06:in squad:start=0; 2026-05-30:played:start=0; 2026-03-31:played:start=0 | 0 | 0.4973 | 0.3801 | 0.1452 | 0.391 | 0.1492 |  |
| Leon Goretzka | GER | 2026-06-06:played:start=0; 2026-05-31:in squad:start=0; 2026-03-30:played:start=0 | 0 | 0.6137 | 0.3713 | 0.1607 | 0.4272 | 0.1841 |  |
| Duke Lacroix | HAI | 2026-06-06:played:start=0; 2026-06-03:played:start=0; 2026-03-31:in squad:start=0 | 0 | 0.4837 | 0.3646 | 0.1367 | 0.3881 | 0.1451 |  |
| Kadir Barria | PAN | 2026-06-06:in squad:start=0; 2026-06-04:played:start=0; 2026-05-31:played:start=0 | 0 | 0.4006 | 0.3598 | 0.1158 | 0.3752 | 0.1202 |  |
