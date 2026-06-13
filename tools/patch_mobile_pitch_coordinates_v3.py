from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_pitch_coordinates_v3_{stamp}.html")
shutil.copy2(p, backup)

changes = []

# 1) Tilføj mobil-specifik render-slot helper før renderPitch
helper_marker = "/* MOBILE_PITCH_COORDINATES_V3_JS_START */"

helper = r'''
/* MOBILE_PITCH_COORDINATES_V3_JS_START */
function shouldUseMobilePitchCoordinates() {
  return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
}

function getMobilePitchXPositions(count) {
  if (count <= 1) return [50];
  if (count === 2) return [34, 66];
  if (count === 3) return [20, 50, 80];
  if (count === 4) return [13, 37.5, 62.5, 87];
  return [10, 30, 50, 70, 90];
}

function getMobilePitchY(positionCode) {
  const code = normalizePositionLabel(positionCode || "");
  if (code === "GK") return 10;
  if (code === "DEF") return 31;
  if (code === "MID") return 56;
  if (code === "FWD") return 82;
  return 50;
}

function getRenderSlots() {
  const slots = getCurrentSlots();

  if (!shouldUseMobilePitchCoordinates()) {
    return slots;
  }

  const groups = {
    GK: slots.filter(slot => normalizePositionLabel(slot.positionCode || slot.position) === "GK"),
    DEF: slots.filter(slot => normalizePositionLabel(slot.positionCode || slot.position) === "DEF"),
    MID: slots.filter(slot => normalizePositionLabel(slot.positionCode || slot.position) === "MID"),
    FWD: slots.filter(slot => normalizePositionLabel(slot.positionCode || slot.position) === "FWD")
  };

  const byKey = new Map();

  Object.entries(groups).forEach(([positionCode, groupSlots]) => {
    const xs = getMobilePitchXPositions(groupSlots.length);
    groupSlots.forEach((slot, index) => {
      byKey.set(slot.key, {
        ...slot,
        x: xs[index] ?? slot.x,
        y: getMobilePitchY(positionCode)
      });
    });
  });

  return slots.map(slot => byKey.get(slot.key) || slot);
}
/* MOBILE_PITCH_COORDINATES_V3_JS_END */
'''

if helper_marker not in text:
    marker = "    function renderPitch() {"
    if marker not in text:
        raise SystemExit("Kunne ikke finde renderPitch().")
    text = text.replace(marker, helper + "\n\n" + marker, 1)
    changes.append("Tilføjet mobil-specifikke slot-koordinater")
else:
    changes.append("Mobil-koordinat-helper fandtes allerede")

old = '''    function renderPitch() {
      const slots = getCurrentSlots();'''

new = '''    function renderPitch() {
      const slots = getRenderSlots();'''

if old in text:
    text = text.replace(old, new, 1)
    changes.append("renderPitch bruger nu getRenderSlots()")
elif "const slots = getRenderSlots();" in text:
    changes.append("renderPitch brugte allerede getRenderSlots()")
else:
    raise SystemExit("Kunne ikke skifte getCurrentSlots() i renderPitch().")

# 2) Tilføj stærkere mobil-CSS efter de gamle patches
css_marker = '<script id="mobile-layout-v3-structure">'

script = r'''
<script id="mobile-layout-v3-structure">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-layout-v3-structure-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-layout-v3-structure-style";
    style.textContent = `
@media (max-width: 700px) {
  html,
  body {
    overflow-x: hidden !important;
  }

  .page-wrap {
    padding: 0 !important;
  }

  .left-panel,
  .right-panel {
    padding: 6px !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }

  /* Top og budget */
  .team-title {
    font-size: 18px !important;
    line-height: 1 !important;
  }

  .app-name {
    font-size: 10px !important;
    line-height: 1 !important;
  }

  .bank-row,
  .summary,
  .stats,
  .team-summary,
  .squad-summary {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 5px !important;
  }

  .bank-box,
  .summary > *,
  .stats > *,
  .team-summary > *,
  .squad-summary > * {
    min-width: 0 !important;
    padding: 5px 6px !important;
    border-radius: 9px !important;
  }

  .bank-box strong {
    font-size: 10px !important;
    line-height: 1 !important;
  }

  .bank-box div,
  #spentValue,
  #bankValue,
  #playerCountValue {
    font-size: 9px !important;
    line-height: 1.05 !important;
  }

  .manual-bank-input {
    height: 19px !important;
    min-height: 19px !important;
    max-width: 60px !important;
    padding: 1px 4px !important;
    font-size: 10px !important;
  }

  /* Strategi gøres til kompakt bjælke */
  .strategy-panel {
    padding: 6px !important;
    margin-bottom: 6px !important;
    border-radius: 10px !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto !important;
    gap: 5px !important;
    align-items: center !important;
  }

  .strategy-topline {
    display: flex !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    min-width: 0 !important;
  }

  .strategy-label {
    font-size: 10px !important;
    line-height: 1 !important;
    margin-right: 2px !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    min-width: 0 !important;
  }

  .strategy-btn {
    width: auto !important;
    min-width: 0 !important;
    height: 25px !important;
    min-height: 25px !important;
    padding: 3px 7px !important;
    font-size: 9.5px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
  }

  .info-icon,
  .strategy-info,
  .info-btn {
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    min-height: 16px !important;
    font-size: 10px !important;
  }

  .active-slot-badge {
    min-height: 25px !important;
    padding: 4px 7px !important;
    border-radius: 9px !important;
    font-size: 9.5px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
  }

  .active-slot-badge strong {
    font-size: 9.5px !important;
  }

  /* Formation + handlinger */
  .action-bar {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 5px !important;
    margin-bottom: 6px !important;
  }

  .formation-select-wrap {
    height: 28px !important;
    min-height: 28px !important;
    padding: 4px 7px !important;
    border-radius: 9px !important;
    gap: 5px !important;
  }

  .formation-select-wrap strong {
    font-size: 10px !important;
  }

  .formation-select-wrap select {
    height: 24px !important;
    min-height: 24px !important;
    padding: 2px 6px !important;
    font-size: 10px !important;
    border-radius: 7px !important;
  }

  .action-btn,
  .action-bar button {
    height: 28px !important;
    min-height: 28px !important;
    padding: 3px 7px !important;
    font-size: 9.5px !important;
    line-height: 1 !important;
    border-radius: 9px !important;
  }

  .status-box,
  .status-message {
    padding: 6px 8px !important;
    margin-bottom: 6px !important;
    font-size: 11px !important;
    border-radius: 9px !important;
  }

  /* Banen: større lodret plads, men kort/knapper er små og rækkerne er spredt */
  .pitch {
    height: 640px !important;
    min-height: 640px !important;
    margin-top: 6px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 70px !important;
    min-width: 70px !important;
    max-width: 70px !important;
    transform: translate(-50%, -50%) !important;
    box-sizing: border-box !important;
  }

  .pitch-player-card,
  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 58px !important;
    min-width: 58px !important;
    max-width: 58px !important;
    padding: 9px 2px 3px !important;
    border-radius: 7px !important;
    box-sizing: border-box !important;
  }

  .pitch-player-flag,
  .pitch-flag-row,
  .player-card .flag,
  .slot-card .flag,
  .squad-card .flag,
  .player-flag {
    transform: scale(0.82) !important;
    transform-origin: center !important;
  }

  .player-name {
    max-width: 54px !important;
    font-size: 7px !important;
    line-height: 0.95 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .player-price {
    max-width: 54px !important;
    font-size: 9px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
  }

  .player-opponent,
  .player-next,
  .player-start {
    max-width: 54px !important;
    font-size: 6.5px !important;
    line-height: 0.95 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .slot-actions,
  .player-actions {
    width: 70px !important;
    max-width: 70px !important;
    display: flex !important;
    flex-direction: row !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 2px !important;
    margin-top: 2px !important;
  }

  .slot-btn,
  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button {
    width: 31px !important;
    min-width: 31px !important;
    max-width: 31px !important;
    height: 17px !important;
    min-height: 17px !important;
    max-height: 17px !important;
    padding: 0 !important;
    font-size: 7.5px !important;
    line-height: 1 !important;
    border-radius: 999px !important;
  }

  .player-figure.empty {
    width: 32px !important;
    height: 32px !important;
    margin-bottom: 2px !important;
  }

  .empty-slot-label,
  .slot-empty-label {
    font-size: 7.5px !important;
    line-height: 1 !important;
  }

  .slot .position-label,
  .slot-position {
    font-size: 7px !important;
    line-height: 1 !important;
  }

  /* Spillerlisten holdes inden for bredden */
  .right-panel {
    margin-top: 0 !important;
  }

  .trade-search,
  .filter-row,
  .right-modebar {
    width: 100% !important;
    max-width: 100% !important;
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 6px !important;
  }

  .right-mode-actions {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
  }

  .trade-row {
    grid-template-columns: minmax(0, 1fr) 28px 74px 36px 38px !important;
    column-gap: 4px !important;
    padding: 6px 3px 6px 4px !important;
  }
}

@media (max-width: 430px) {
  .bank-row,
  .summary,
  .stats,
  .team-summary,
  .squad-summary {
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  }

  .pitch {
    height: 625px !important;
    min-height: 625px !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 66px !important;
    min-width: 66px !important;
    max-width: 66px !important;
  }

  .pitch-player-card,
  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 55px !important;
    min-width: 55px !important;
    max-width: 55px !important;
  }

  .slot-actions,
  .player-actions {
    width: 66px !important;
    max-width: 66px !important;
  }

  .slot-btn,
  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button {
    width: 29px !important;
    min-width: 29px !important;
    max-width: 29px !important;
    font-size: 7.2px !important;
  }

  .strategy-content {
    grid-template-columns: 1fr !important;
  }

  .active-slot-badge {
    justify-self: stretch !important;
    display: flex !important;
    justify-content: space-between !important;
  }
}
`;
    document.head.appendChild(style);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addStyle);
  } else {
    addStyle();
  }
}());
</script>
'''

if css_marker in text:
    changes.append("mobile-layout-v3-structure fandtes allerede")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    changes.append("Tilføjet mobil layout v3 CSS")

p.write_text(text, encoding="utf-8")

print("OK: Mobil pitch-koordinater og kompakt top v3 er lagt ind.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "MOBILE_PITCH_COORDINATES_V3_JS_START",
    "function getRenderSlots",
    "const slots = getRenderSlots();",
    "mobile-layout-v3-structure",
    "height: 625px !important",
]:
    print(needle + " => " + str(text2.count(needle)))
