# Transfermarkt Manual URL Integration Audit

Manual URL-prioritet: `tools/transfermarkt_manual_urls.csv` -> eksisterende cache-URL -> automatisk Transfermarkt-soegning.

## Counts

- Rækker i manualfilen: 28
- Matchet til player pool: 10
- Uden match: 18
- Tidligere `status=error`: 0
- Efter test `status=ok_manual_url`: 10
- Dubletter: 0
- Tvetydige navnmatch: 0

## Sanity-spillere

| manual_player_name | matched_player_id | matched_player_name | team_id | previous_status | test_status | match_status | duplicate_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cucho Hernández |  |  |  |  |  | no_match |  |
| Alex Robertson | alexander_alex_robertson__aus | Alexander 'Alex' Robertson | AUS | ok_manual_url | ok_manual_url | manual_alias |  |
| Matías Fernández-Pardo | matias_fernandez_pardo__bel | Matias Fernandez-Pardo | BEL | ok_manual_url | ok_manual_url | exact_name |  |
| Marc Pubill |  |  |  |  |  | no_match |  |
| Bento Krepski | bento_krepski__bra | Bento Krepski | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Ederson Moraes | ederson_moraes__bra | Ederson Moraes | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Luis Suárez Charris |  |  |  |  |  | no_match |  |
| Pablo Gavi |  |  |  |  |  | no_match |  |

## Uden match eller tvetydige

| manual_player_name | match_status | match_detail | duplicate_warning |
| --- | --- | --- | --- |
| Shararh | no_match |  |  |
| Cucho Hernández | no_match |  |  |
| Jahkeele Marshall-Rutty | no_match |  |  |
| Ange-Yoan Bonny | no_match |  |  |
| Lionel Mpasi-Nzau | no_match |  |  |
| João Paulo | no_match |  |  |
| CJ dos Santos | no_match |  |  |
| Kevin Lenini | no_match |  |  |
| José Andrés Hurtado | no_match |  |  |
| Marc Pubill | no_match |  |  |
| Odeh Fakhoury | no_match |  |  |
| Abdulrahman Al-Oboud | no_match |  |  |
| Carlos Rodríguez | no_match |  |  |
| Bara Sapoko Ndiaye | no_match |  |  |
| Mohamed Belhadj Mahmoud | no_match |  |  |
| Umarali Rakhmonaliyev | no_match |  |  |
| Luis Suárez Charris | no_match |  |  |
| Pablo Gavi | no_match |  |  |
