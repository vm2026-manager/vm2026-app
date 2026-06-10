from pathlib import Path

path = Path("index.html")
html = path.read_text(encoding="utf-8", errors="replace")

changed = False

# 1) Tilføj helperfunktion, hvis den ikke findes
helper = r'''
    function isRobertoAlvaradoPlayer(player) {
      const name = String(player?.player_name || player?.name || "").toLowerCase();
      const playerId = String(player?.player_id || "").toLowerCase();
      const team = String(player?.team_id || player?.team_name || player?.team || "").toLowerCase();
      return (
        (name.includes("roberto alvarado") || playerId.includes("roberto_alvarado")) &&
        (team.includes("mex") || team.includes("mexico") || playerId.endsWith("__mex"))
      );
    }
'''

if "function isRobertoAlvaradoPlayer(player)" not in html:
    marker = "    function hasManualCheck(player) {"
    if marker in html:
        html = html.replace(marker, helper + "\n" + marker, 1)
        changed = True
    else:
        marker2 = "    function getPlayerPrice(player) {"
        if marker2 in html:
            html = html.replace(marker2, helper + "\n" + marker2, 1)
            changed = True
        else:
            raise SystemExit("Kunne ikke finde et sikkert sted at indsætte helperfunktionen.")

# 2) Sørg for, at hasManualCheck aldrig giver Tjek start for Roberto Alvarado
old = "    function hasManualCheck(player) {\n"
new = "    function hasManualCheck(player) {\n      if (isRobertoAlvaradoPlayer(player)) return false;\n"

if old in html and new not in html:
    html = html.replace(old, new, 1)
    changed = True

# 3) Fjern pitch-udråbstegnet for Roberto Alvarado
old_condition = '      if (manualStatus === "check" || manualStartStatus === "doubtful") {'
new_condition = '      if (!isRobertoAlvaradoPlayer(player) && (manualStatus === "check" || manualStartStatus === "doubtful")) {'

if old_condition in html:
    html = html.replace(old_condition, new_condition, 1)
    changed = True
elif new_condition in html:
    pass
else:
    print("Advarsel: fandt ikke præcis pitch-warning condition. Måske er den allerede ændret.")

# 4) Ekstra sikkerhed: fjern evt. manuel Alvarado check i tekstbaserede manual notes, hvis navnet står direkte
html = html.replace(
    '"Roberto Alvarado": { manual_status: "check"',
    '"Roberto Alvarado": { manual_status: ""'
)
html = html.replace(
    '"Roberto Alvarado": { manual_start_status: "doubtful"',
    '"Roberto Alvarado": { manual_start_status: ""'
)

path.write_text(html, encoding="utf-8")

print("Changed index.html:", changed)
print("Roberto Alvarado manual Tjek start exception is now patched.")
