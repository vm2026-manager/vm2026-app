# Recent non-starter start probability audit

## Modelændring

- `in squad` uden minutter tæller som en tilgængelig ikke-start.
- Skade og suspension er neutral utilgængelighed og indgår ikke i start-rate-nævneren.
- Conditional start-rate vægter de tre seneste tilgængelige observationer 70% og recency-vægtet historik 30%.
- Context-overrides anvendes bagefter og beholder højeste prioritet.

- Spillere med nyere start-rate højst 1/3: 658

## Rüdiger og fem øvrige sanity-spillere

| player_name | team_id | recent_available_observations | recent_available_start_rate | historical_weighted_start_rate | old_start_prob | new_start_prob | old_conditional_start_prob | new_conditional_start_prob | context_override |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Antonio Rüdiger | GER | 2026-05-31:in squad:start=0; 2026-03-30:played:start=0; 2026-03-27:in squad:start=0 | 0 | 0.7633 | 0.2165 | 0.2165 | 0.229 | 0.229 |  |
| Tomas Holes | CZE | 2026-06-05:played:start=0; 2026-05-31:played:start=0; 2026-03-31:in squad:start=0 | 0 | 0.6686 | 0.1868 | 0.1868 | 0.2006 | 0.2006 |  |
| Rami Rabia | EGY | 2026-05-28:played:start=0; 2026-03-31:in squad:start=0; 2026-03-27:played:start=0 | 0 | 0.706 | 0.1821 | 0.1821 | 0.2118 | 0.2118 |  |
| Cameron Burgess | AUS | 2026-05-31:played:start=0; 2026-03-31:played:start=0; 2026-03-27:played:start=0 | 0 | 0.6009 | 0.1772 | 0.1772 | 0.1803 | 0.1803 |  |
| Yuto Nagatomo | JPN | 2026-05-31:played:start=0; 2025-10-14:in squad:start=0; 2025-10-10:in squad:start=0 | 0 | 0.6819 | 0.1759 | 0.1759 | 0.2046 | 0.2046 |  |
| Jearl Margaritha | CUW | 2026-05-30:in squad:start=0; 2026-03-31:played:start=0; 2026-03-27:played:start=0 | 0 | 0.596 | 0.1755 | 0.1755 | 0.1788 | 0.1788 |  |
