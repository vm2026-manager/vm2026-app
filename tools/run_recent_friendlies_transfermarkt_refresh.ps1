Set-Location "$env:USERPROFILE\Documents\GitHub\vm2026-app"

Write-Host "Refreshing Transfermarkt manual URLs for recent-friendly VM teams..."

Write-Host "Batch 1: 4 players"
python .\tools\batch_transfermarkt_national_usage.py --refresh --manual-only --manual-name "Bento Krepski" --manual-name "Dominique Simon" --manual-name "Ederson Moraes" --manual-name "Lenny Joseph"
