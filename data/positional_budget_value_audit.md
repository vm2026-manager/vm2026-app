# Positional Budget Value Audit

Audit baseret paa eksisterende spillerdata og strategi-/formationsoutput. Ingen optimizer eller strategioutput er genkoert.

## Kort konklusion

- Premium FWD sammenligninger: 336; premium har positiv strategiscore mod billig FWD i ca. 44% af rækkerne.
- MID/DEF/GK upgrade-rækker med lav marginal strategireturnering (-0,25 til 0,75 pr. mio.): ca. 21%.
- Det peger ikke paa en universel premium-FWD-undervurdering, men Haaland/Kane kan se relativt svage ud i bestemte next_round/runde-kontekster, mens Mbappe typisk scorer som premium.
- Centrale MID/DEF kan stadig fremstå attraktive, især naar de kombinerer starter-sikkerhed og pris/value; det er et kalibreringsspor, ikke en sikker datafejl.

## Marginal returnering pr. position

| comparison_type | avg_marginal_ev_per_million | avg_marginal_strategy_score_per_million | rows |
|---|---:|---:|---:|
| low_upside_mid_def_gk_upgrade | -0.378 | 0.841 | 14 |
| position_marginal_DEF | 0.601 | 1.065 | 28 |
| position_marginal_FWD | 0.478 | 0.965 | 28 |
| position_marginal_GK | 0.742 | 1.583 | 28 |
| position_marginal_MID | 0.831 | 1.925 | 28 |
| premium_fwd_vs_cheap_fwd | 0.053 | -0.119 | 336 |
| selected_DEF_upgrade | 0.540 | 0.153 | 112 |
| selected_GK_upgrade | -0.236 | -1.410 | 28 |
| selected_MID_upgrade | 0.755 | 1.970 | 98 |
| two_player_swap_premium_fwd_plus_cheaper_mid_def_gk | -0.122 | -0.849 | 11 |

## Premium FWD vs cheap FWD

| strategy | formation | premium | cheap_fwd | price_diff | strategy_score_diff | marginal_score_per_mio | interpretation |
|---|---|---|---:|---:|---:|---:|---|
| group_stage | 4-5-1 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 5-3-2 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 5-4-1 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 4-4-2 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 3-5-2 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 3-4-3 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| group_stage | 4-3-3 | Erling Haaland | Jonathan David | 4000000 | -4.9957 | -1.2489 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 3-5-2 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 3-4-3 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-3-3 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 5-4-1 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-4-2 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-5-1 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 5-3-2 | Erling Haaland | Jonathan David | 4000000 | -4.6222 | -1.1555 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-4-2 | Harry Kane | Jonathan David | 5000000 | -4.0306 | -0.8061 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-3-3 | Harry Kane | Jonathan David | 5000000 | -4.0306 | -0.8061 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 4-5-1 | Harry Kane | Jonathan David | 5000000 | -4.0306 | -0.8061 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |
| round1_2 | 3-4-3 | Harry Kane | Jonathan David | 5000000 | -4.0306 | -0.8061 | Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; negative values indicate the cheap/value FWD is ahead on current model output. |

## Formation-risk

| strategy | formation | fwd_count | premium_fwd_count | low_upside_mid_count | note |
|---|---|---:|---:|---:|---|
| next_round | 4-5-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| next_round | 5-4-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| round1_2 | 4-5-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| round1_2 | 5-4-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| group_stage | 4-5-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| group_stage | 5-4-1 | 1 | 0 | 0 | Saerligt udsat for lav FWD-ceiling. |
| long_run | 3-5-2 | 2 | 0 | 3 | Flere low-upside MID uden premium FWD. |
| long_run | 4-4-2 | 2 | 0 | 2 | Flere low-upside MID uden premium FWD. |
| long_run | 4-5-1 | 1 | 0 | 2 | Saerligt udsat for lav FWD-ceiling. |
| long_run | 5-3-2 | 2 | 0 | 2 | Flere low-upside MID uden premium FWD. |
| long_run | 5-4-1 | 1 | 0 | 1 | Saerligt udsat for lav FWD-ceiling. |

## Svar paa auditspoergsmaal

- Premiumangribere ser ikke systematisk undervurderede ud paa alle strategier; Mbappe er tydeligt staerk, mens Haaland/Kane kan blive presset af billig value og runde-kontekst.
- Centrale/lav-upside midtbanespillere ser potentielt overvurderede ud i nogle strategy-score-rækker, især naar starter-sikkerhed og pris/value kombineres.
- Problemet virker mest strategi- og formationsafhaengigt, ikke globalt. Formationer med en enkelt FWD, især 4-5-1/5-4-1, er mest udsatte for at ofre offensiv ceiling.
- Der er grundlag for en senere modelaudit af offensive ceiling-komponenter, men ikke for at aendre vaegte uden ny godkendelse.
