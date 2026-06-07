# Transfermarkt Manual URL Integration Audit

Manual URL-prioritet: `tools/transfermarkt_manual_urls.csv` -> eksisterende cache-URL -> automatisk Transfermarkt-soegning.

## Counts

- Rækker i manualfilen: 29
- Matchet til player pool: 24
- Uden match: 5
- Tidligere `status=error`: 1
- Efter test `status=ok_manual_url`: 24
- Dubletter: 0
- Tvetydige navnmatch: 0

## Sanity-spillere

| manual_player_name | matched_player_id | matched_player_name | team_id | previous_status | test_status | match_status | duplicate_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cucho Hernández | juan_cucho_hernandez__col | Juan 'Cucho' Hernandez | COL | ok_manual_url | ok_manual_url | manual_alias |  |
| Alex Robertson | alexander_alex_robertson__aus | Alexander 'Alex' Robertson | AUS | ok_manual_url | ok_manual_url | manual_alias |  |
| Matías Fernández-Pardo | matias_fernandez_pardo__bel | Matias Fernandez-Pardo | BEL | ok_manual_url | ok_manual_url | exact_name |  |
| Marc Pubill | marc_pubill__esp | Marc Pubill | ESP | ok_manual_url | ok_manual_url | exact_name |  |
| Bento Krepski | bento_krepski__bra | Bento Krepski | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Ederson Moraes | ederson_moraes__bra | Ederson Moraes | BRA | ok_manual_url | ok_manual_url | exact_name |  |
| Luis Suárez Charris | luis_suarez_charris__col | Luis Suarez Charris | COL | ok_manual_url | ok_manual_url | exact_name |  |
| Pablo Gavi | pablo_gavi__esp | Pablo Gavi | ESP | ok_manual_url | ok_manual_url | exact_name |  |

## Uden match eller tvetydige

| manual_player_name | match_status | match_detail | duplicate_warning |
| --- | --- | --- | --- |
| Jahkeele Marshall-Rutty | no_match |  |  |
| José Andrés Hurtado | no_match |  |  |
| Odeh Fakhoury | no_match |  |  |
| Abdulrahman Al-Oboud | no_match |  |  |
| Mohamed Belhadj Mahmoud | no_match |  |  |
