Set-Location "$env:USERPROFILE\Documents\GitHub\vm2026-app"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "data\holdet_master_new_players_pipeline_$stamp.log"
Start-Transcript -Path $log

Write-Host "=== HOLDET MASTER PIPELINE ==="
Write-Host "Backup dir: data\backup_before_holdet_master_pipeline_20260607_152751"
Write-Host "New active Holdet players to Transfermarkt refresh: 140"

Write-Host "=== 1/12 Build player_pool from Holdet 616 master ==="
python .\tools\build_player_pool_from_holdet_616.py --write

Write-Host "=== 2/12 Rebase EV to Holdet master ==="
python .\tools\rebase_player_ev_to_holdet_master.py

Write-Host "=== 3/12 Transfermarkt refresh for new active Holdet players ==="
Write-Host "Transfermarkt chunk 1: 25 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 25 --manual-name 'Aaron Tshibola' --manual-name 'Abbosbek Fayzullaev' --manual-name 'Abdul Mumin' --manual-name 'Abdul Rahman Baba' --manual-name 'Abdulla Abdullaev' --manual-name 'Adam Hlozek' --manual-name 'Agustin Canobbio' --manual-name 'Ahmed Al-Ganehi' --manual-name 'Ahmed Al-Kassar' --manual-name 'Ahmed Qasem' --manual-name 'Ahmed Reda Tagnaouti' --manual-name 'Ahmed Yahya' --manual-name 'Al-Hashmi Al-Hussain' --manual-name 'Alaa Hejji' --manual-name 'Alberto Quintero' --manual-name 'Alejandro Romero Gamarra' --manual-name 'Alex Zendejas' --manual-name 'Alexandr Sojka' --manual-name 'Alexandro Maidana' --manual-name 'Alfie Jones' --manual-name 'Ali Lajami' --manual-name 'Ali Olwan' --manual-name 'Alidu Seidu' --manual-name 'Amir Mohammad Razzaghinia' --manual-name 'Anas Badawi'

Write-Host "Transfermarkt chunk 2: 25 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 25 --manual-name 'Arya Yousefi' --manual-name 'Assan Ouedraogo' --manual-name 'Augustine Boakye' --manual-name 'Auston Trusty' --manual-name 'Avazbek Ulmasaliyev' --manual-name 'Ayoube Amaimouni-Echghouyab' --manual-name 'Ayyoub Bouaddi' --manual-name 'Azarias Londono' --manual-name 'Azizbek Amanov' --manual-name 'Behruzjon Karimov' --manual-name 'Bradley Cross' --manual-name 'Caglar Soyuncu' --manual-name 'Cameron Devlin' --manual-name 'Can Uzun' --manual-name 'Carney Chukwuemeka' --manual-name 'Cesar Huerta' --manual-name 'Christopher Operi' --manual-name 'Cristian Volpato' --manual-name 'Daniyal Eiri' --manual-name 'David Affengruber' --manual-name 'David Doudera' --manual-name 'David Zima' --manual-name 'Denil Castillo' --manual-name 'Dennis ''Dargahi'' Eckert Ayensa' --manual-name 'Edgardo Farina'

Write-Host "Transfermarkt chunk 3: 25 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 25 --manual-name 'Ehsan Haddad' --manual-name 'Ernest Nuamah' --manual-name 'Fabian Balbuena' --manual-name 'Fabinho' --manual-name 'Facundo Medina' --manual-name 'Firas Al Brikan' --manual-name 'Florian Wiegele' --manual-name 'Gedeon Kalulu' --manual-name 'Gessime Yassine' --manual-name 'Gilberto Mora' --manual-name 'Gustavo Caballero' --manual-name 'Guus Til' --manual-name 'Haji Wright' --manual-name 'Harry Souttar' --manual-name 'Hassan Al Tambakti' --manual-name 'Hassan Al-Haydos' --manual-name 'Hassan Kadesh' --manual-name 'Herman Johansson' --manual-name 'Homam Al-Amin Ahmed' --manual-name 'Hugo Sochurek' --manual-name 'Iqraam Rayners' --manual-name 'Isidro Pitta' --manual-name 'Jackson Irvine' --manual-name 'Jacob Shaffelburg' --manual-name 'Jalal Hassan'

Write-Host "Transfermarkt chunk 4: 25 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 25 --manual-name 'Jaloliddin Masharipov' --manual-name 'Jan Kuchta' --manual-name 'Jaouen Hadjam' --manual-name 'Jehad Thakri' --manual-name 'Jindrich Stanek' --manual-name 'Joe Scally' --manual-name 'Jonas Adjetey' --manual-name 'Jose Canale' --manual-name 'Justin Kluivert' --manual-name 'Kamogelo Sebelebele' --manual-name 'Keeto Thermoncy' --manual-name 'Khalid Al-Ghannam' --manual-name 'Lisandro Martinez' --manual-name 'Luis Chavez' --manual-name 'Luis Romo' --manual-name 'Marten de Roon' --manual-name 'Mateo Chavez' --manual-name 'Mathew Leckie' --manual-name 'Mats Wieffer' --manual-name 'Mauricio Magalhaes Prado' --manual-name 'Mehdi Torabi' --manual-name 'Melvin Feycal Mastil' --manual-name 'Michael Svoboda' --manual-name 'Mladen Jurkas' --manual-name 'Mohamed Amine Tougai'

Write-Host "Transfermarkt chunk 5: 25 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 25 --manual-name 'Mohammed Abu Al-Shamat' --manual-name 'Mohammed Al-Owais' --manual-name 'Moteb Al-Harbi' --manual-name 'Moïse Bombito' --manual-name 'Munir Mohamedi El Kajoui' --manual-name 'Mustafa Saadoon' --manual-name 'Nabil Bentaleb' --manual-name 'Nawaf Bu Washl' --manual-name 'Nilson Angulo' --manual-name 'Noureddin Bani Attiah' --manual-name 'Olwethu Makhanya' --manual-name 'Oussama Benbout' --manual-name 'Paul Wanner' --manual-name 'Rajaei Ayed' --manual-name 'Redouane Halhal' --manual-name 'Rodrigo Zalazar' --manual-name 'Roozbeh Cheshmi' --manual-name 'Saed Al-Rosan' --manual-name 'Salah Zakaria' --manual-name 'Saleh Al-Shehri' --manual-name 'Saleh Hardani' --manual-name 'Samir Chergui' --manual-name 'Samir El Mourabet' --manual-name 'Santiago Gimenez' --manual-name 'Sasa Kalajdzic'

Write-Host "Transfermarkt chunk 6: 15 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --limit 15 --manual-name 'Sebastian Caceres' --manual-name 'Seydinaissa Laye' --manual-name 'Seyed Hossein Hosseini' --manual-name 'Sherzod Esanov' --manual-name 'Soufiane Rahimi' --manual-name 'Stephen Eustaquio' --manual-name 'Sultan Mandash' --manual-name 'Tahsin Jamshid' --manual-name 'Tete Yengi' --manual-name 'Tomas Rodriguez' --manual-name 'Tyler Fletcher' --manual-name 'Valentin Barco' --manual-name 'Wi-je Cho' --manual-name 'Youssef Belammari' --manual-name 'Yusuf Abdurisag'

Write-Host "=== 4/12 Classify Transfermarkt national matches ==="
python .\tools\classify_transfermarkt_national_matches.py

Write-Host "=== 5/12 Write Transfermarkt start watchlist ==="
python .\tools\write_transfermarkt_start_watchlist.py

Write-Host "=== 6/12 Merge Transfermarkt summary to player_pool ==="
python .\tools\merge_transfermarkt_competitive_summary_to_player_pool.py

Write-Host "=== 7/12 Re-apply manual start overrides ==="
python .\tools\apply_manual_start_overrides_review.py

Write-Host "=== 8/12 Repair EV start_prob from player_pool ==="
python .\tools\repair_ev_start_prob_from_player_pool.py

Write-Host "=== 9/12 Repair EV components ==="
python .\tools\repair_ev_components_after_start_prob_repair.py

Write-Host "=== 10/12 Repair EV price-quality consistency ==="
python .\tools\repair_ev_price_quality_consistency.py

Write-Host "=== 11/12 Build start signal layer ==="
python .\tools\build_start_signal_layer.py

Write-Host "=== 12/12 Optimize squads ==="
python .\tools\optimize_squad_group_stage.py

Write-Host "=== JSON sanity ==="
python .\tools\sanity_check_active_json.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Active JSON sanity failed. Pipeline stops before any downstream commit/push."
    Stop-Transcript
    exit $LASTEXITCODE
}

Write-Host "=== DONE ==="
Write-Host "Logfil: $log"
Stop-Transcript
