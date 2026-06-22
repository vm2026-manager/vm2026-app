# Suspensions Model Impact Audit

- Aktive karantæner: 5
- Matchet i player_pool: 3
- Fundet i EV-fil: 3
- Fundet i optimizer-squads: 0
- Unmatched: Mohammed Al Amin (QAT), Assan O. Madibo (QAT)

## Spillere

| Spiller | Team | Runde | start_prob | availability_status | holdet_is_out | EV | optimizer_ev | I optimizer? | Anbefalet handling |
|---|---|---:|---:|---|---|---:|---:|---|---|
| Nathan Ngoy | BEL | 3 | 0.25 | unknown | False | 0.1235 | 0.1235 | no | set next-match start_prob to 0; mark availability_status suspended; keep holdet_is_out false; exclude from optimizer for affected round |
| Miguel Almiron | PAR | 3 | 0.824 | medium_risk | False | 2.221845 | 2.221845 | no | set next-match start_prob to 0; mark availability_status suspended; keep holdet_is_out false; exclude from optimizer for affected round |
| Mohammed Al Amin | QAT | 3 | - | - | - | - | - | no | manual review - no model match found |
| Assan O. Madibo | QAT | 3 | - | - | - | - | - | no | manual review - no model match found |
| Tarik Muharemovic | BIH | 3 | 0.66 | medium_risk | False | 1.020419 | 1.020419 | no | set next-match start_prob to 0; mark availability_status suspended; keep holdet_is_out false; exclude from optimizer for affected round |
