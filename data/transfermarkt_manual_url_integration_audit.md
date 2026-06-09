# Transfermarkt Manual URL Integration Audit

Manual URL-prioritet: `tools/transfermarkt_manual_urls.csv` -> eksisterende cache-URL -> automatisk Transfermarkt-soegning.

## Counts

- Rækker i manualfilen: 29
- Matchet til player pool: 11
- Uden match: 18
- Tidligere `status=error`: 0
- Efter test `status=ok_manual_url`: 11
- Dubletter: 0
- Tvetydige navnmatch: 0

## Sanity-spillere

| manual_player_name | matched_player_id | matched_player_name | team_id | previous_status | test_status | match_status | duplicate_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cucho Hernández | juan_cucho_hernandez__col | Juan 'Cucho' Hernandez | COL | ok_manual_url | ok_manual_url | manual_alias |  |
| Alex Robertson |  |  |  |  |  | no_match |  |
| Matías Fernández-Pardo |  |  |  |  |  | no_match |  |
| Marc Pubill | marc_pubill__esp | Marc Pubill | ESP | ok_manual_url | ok_manual_url | exact_name |  |
| Bento Krepski | bento_krepski__bra | Bento Krepski | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Ederson Moraes | ederson_moraes__bra | Ederson Moraes | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Luis Suárez Charris | luis_suarez_charris__col | Luis Suarez Charris | COL | ok_manual_url | ok_manual_url | exact_name |  |
| Pablo Gavi | pablo_gavi__esp | Pablo Gavi | ESP | ok_manual_url | ok_manual_url | exact_name |  |

## Uden match eller tvetydige

| manual_player_name | match_status | match_detail | duplicate_warning |
| --- | --- | --- | --- |
| Abdulaziz Hatim | no_match |  |  |
| Alex Robertson | no_match |  |  |
| Matías Fernández-Pardo | no_match |  |  |
| Jahkeele Marshall-Rutty | no_match |  |  |
| Ange-Yoan Bonny | no_match |  |  |
| Lionel Mpasi-Nzau | no_match |  |  |
| João Paulo | no_match |  |  |
| CJ dos Santos | no_match |  |  |
| Kevin Lenini | no_match |  |  |
| José Andrés Hurtado | no_match |  |  |
| Lenny Joseph | no_match |  |  |
| Dominique Simon | no_match |  |  |
| Odeh Fakhoury | no_match |  |  |
| Abdulrahman Al-Oboud | no_match |  |  |
| Carlos Rodríguez | no_match |  |  |
| Bara Sapoko Ndiaye | no_match |  |  |
| Mohamed Belhadj Mahmoud | no_match |  |  |
| Alejandro Romero Gamarra | no_match |  |  |
