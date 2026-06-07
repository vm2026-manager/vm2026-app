# Recent non-starter start probability audit

## Modelændring

- `in squad` uden minutter tæller som en tilgængelig ikke-start.
- Skade og suspension er neutral utilgængelighed og indgår ikke i start-rate-nævneren.
- Conditional start-rate vægter de tre seneste tilgængelige observationer 70% og recency-vægtet historik 30%.
- Context-overrides anvendes bagefter og beholder højeste prioritet.

- Spillere med nyere start-rate højst 1/3: 754

## Rüdiger og fem øvrige sanity-spillere

| player_name | team_id | recent_available_observations | recent_available_start_rate | historical_weighted_start_rate | old_start_prob | new_start_prob | old_conditional_start_prob | new_conditional_start_prob | context_override |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Antonio Rüdiger | GER | 2026-06-06:in squad:start=0; 2026-05-31:in squad:start=0; 2026-03-30:played:start=0 | 0 | 0.7308 | 0.208 | 0.208 | 0.2193 | 0.2193 |  |
| Rodrigo de Paul | ARG | 2026-06-07:played:start=0; 2026-03-31:played:start=0; 2026-03-27:played:start=0 | 0 | 0.7526 | 0.2199 | 0.2199 | 0.2258 | 0.2258 |  |
| Tomas Holes | CZE | 2026-06-05:played:start=0; 2026-05-31:played:start=0; 2026-03-31:in squad:start=0 | 0 | 0.6686 | 0.187 | 0.187 | 0.2006 | 0.2006 |  |
| Yuto Nagatomo | JPN | 2026-05-31:played:start=0; 2025-10-14:in squad:start=0; 2025-10-10:in squad:start=0 | 0 | 0.6819 | 0.1759 | 0.1759 | 0.2046 | 0.2046 |  |
| Jearl Margaritha | CUW | 2026-05-30:in squad:start=0; 2026-03-31:played:start=0; 2026-03-27:played:start=0 | 0 | 0.596 | 0.1755 | 0.1755 | 0.1788 | 0.1788 |  |
| Rami Rabia | EGY | 2026-06-06:in squad:start=0; 2026-05-28:played:start=0; 2026-03-31:in squad:start=0 | 0 | 0.6741 | 0.1739 | 0.1739 | 0.2022 | 0.2022 |  |
