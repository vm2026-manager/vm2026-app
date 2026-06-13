from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_layout_polish_final_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-layout-polish-final">'

script = r'''
<script id="mobile-layout-polish-final">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-layout-polish-final-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-layout-polish-final-style";
    style.textContent = `
@media (max-width: 700px) {
  html,
  body {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
  }

  .page-wrap,
  .page-inner,
  .app {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }

  .page-wrap {
    padding: 0 !important;
  }

  .page-inner {
    height: auto !important;
  }

  .app {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) !important;
    height: auto !important;
  }

  .left-panel,
  .right-panel {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    padding: 8px !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }

  .team-title {
    font-size: 20px !important;
    line-height: 1.05 !important;
  }

  .topbar-auth {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    justify-content: flex-end !important;
  }

  .topbar-user {
    max-width: 210px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    font-size: 11px !important;
    padding: 7px 8px !important;
  }

  .topbar-btn {
    min-height: 32px !important;
    padding: 7px 9px !important;
    font-size: 12px !important;
  }

  .strategy-panel,
  .action-bar,
  .formation-select-wrap {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
  }

  .strategy-content,
  .strategy-topline,
  .action-bar {
    gap: 6px !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 5px !important;
  }

  .strategy-btn,
  .action-bar button,
  .formation-select-wrap select {
    min-height: 32px !important;
    height: 32px !important;
    padding: 4px 7px !important;
    font-size: 11px !important;
    line-height: 1.1 !important;
    border-radius: 9px !important;
  }

  .pitch {
    width: 100% !important;
    max-width: 100% !important;
    height: 600px !important;
    min-height: 600px !important;
    margin: 8px auto 0 !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    border-radius: 14px !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 68px !important;
    min-width: 68px !important;
    max-width: 68px !important;
    gap: 1px !important;
    transform: translateX(-50%) !important;
    box-sizing: border-box !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 64px !important;
    min-width: 64px !important;
    max-width: 64px !important;
    padding: 10px 2px 3px !important;
    border-radius: 7px !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    text-align: center !important;
  }

  .player-card img,
  .slot-card img,
  .squad-card img,
  .mobile-pitch-player-card img,
  .player-card .flag,
  .slot-card .flag,
  .squad-card .flag,
  .mobile-pitch-player-card .flag,
  .player-flag {
    width: 25px !important;
    height: 18px !important;
    max-width: 25px !important;
    max-height: 18px !important;
  }

  .player-card strong,
  .player-card b,
  .slot-card strong,
  .slot-card b,
  .squad-card strong,
  .squad-card b,
  .mobile-pitch-player-card strong,
  .mobile-pitch-player-card b,
  .player-name,
  .name {
    max-width: 58px !important;
    font-size: 7.4px !important;
    line-height: 0.95 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    font-size: 7.2px !important;
    line-height: 0.95 !important;
  }

  .player-next,
  .player-start,
  .slot .player-next,
  .mobile-pitch-player-card .player-next {
    max-width: 58px !important;
    font-size: 6.8px !important;
    line-height: 0.95 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .slot-actions,
  .player-actions,
  .mobile-pitch-player-slot .slot-actions {
    display: flex !important;
    justify-content: center !important;
    gap: 2px !important;
    width: 68px !important;
    max-width: 68px !important;
  }

  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button,
  .mobile-pitch-player-slot button {
    min-width: 28px !important;
    width: 28px !important;
    max-width: 28px !important;
    height: 18px !important;
    min-height: 18px !important;
    padding: 0 2px !important;
    font-size: 8.5px !important;
    line-height: 1 !important;
    border-radius: 999px !important;
  }

  .right-panel {
    margin-top: 0 !important;
  }

  .right-modebar,
  .trade-search,
  .filter-row {
    width: 100% !important;
    max-width: 100% !important;
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 8px !important;
    box-sizing: border-box !important;
  }

  .right-mode-actions {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 8px !important;
  }

  .trade-search input,
  .trade-search select,
  .clearable-select-wrap,
  .toggle-chip,
  #searchInput,
  #teamFilter,
  #positionFilter,
  #sortSelect {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
  }

  .toggle-chip {
    justify-content: flex-start !important;
    padding: 8px 10px !important;
    white-space: normal !important;
  }

  .trade-list,
  #tradeList {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }

  .trade-row {
    width: 100% !important;
    max-width: 100% !important;
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 32px 82px 40px 44px !important;
    column-gap: 5px !important;
    padding: 7px 4px 7px 5px !important;
    box-sizing: border-box !important;
    align-items: center !important;
  }

  .trade-name {
    max-width: 100% !important;
    font-size: 14px !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .trade-meta,
  .trade-next {
    max-width: 100% !important;
    font-size: 11px !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .trade-position-value,
  .mobile-position-pill {
    min-width: 28px !important;
    max-width: 30px !important;
    padding: 5px 4px !important;
    font-size: 10px !important;
    border-radius: 8px !important;
    text-align: center !important;
  }

  .trade-price {
    min-width: 0 !important;
    width: 82px !important;
    max-width: 82px !important;
    font-size: 12px !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-align: right !important;
  }

  .trade-price-label {
    font-size: 10px !important;
    line-height: 1 !important;
  }

  .trade-row .favorite-btn,
  .favorite-btn {
    width: 38px !important;
    min-width: 38px !important;
    max-width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    max-height: 38px !important;
    border-radius: 10px !important;
    padding: 0 !important;
    font-size: 18px !important;
  }

  .trade-row .plus-btn,
  .plus-btn {
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 38px !important;
    min-height: 38px !important;
    max-height: 38px !important;
    border-radius: 10px !important;
    padding: 0 !important;
    font-size: 18px !important;
    justify-self: end !important;
  }
}

@media (max-width: 430px) {
  .pitch {
    height: 580px !important;
    min-height: 580px !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 64px !important;
    min-width: 64px !important;
    max-width: 64px !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 60px !important;
    min-width: 60px !important;
    max-width: 60px !important;
  }

  .slot-actions,
  .player-actions,
  .mobile-pitch-player-slot .slot-actions {
    width: 64px !important;
    max-width: 64px !important;
  }

  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button,
  .mobile-pitch-player-slot button {
    min-width: 26px !important;
    width: 26px !important;
    max-width: 26px !important;
    font-size: 8px !important;
  }

  .trade-row {
    grid-template-columns: minmax(0, 1fr) 30px 76px 38px 40px !important;
    column-gap: 4px !important;
  }

  .trade-price {
    width: 76px !important;
    max-width: 76px !important;
    font-size: 11.5px !important;
  }

  .trade-row .favorite-btn,
  .favorite-btn {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
  }

  .trade-row .plus-btn,
  .plus-btn {
    width: 38px !important;
    min-width: 38px !important;
    max-width: 38px !important;
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

if marker in text:
    print("Mobil final-polish findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet sidste mobil-layout override.")
    print(f"Backup: {backup}")
    print("- Banekort gjort mindre")
    print("- Fjern/Skift gjort mindre")
    print("- Højre panel sat til 100% i stedet for 100vw")
    print("- Spillerlistekolonner gjort smallere")
    print("- Filtre/søgning stables pænere på mobil")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-layout-polish-final",
    "mobile-layout-polish-final-style",
    "grid-template-columns: minmax(0, 1fr) 32px 82px 40px 44px",
    "width: 60px !important",
    "height: 580px !important",
]:
    print(needle + " => " + str(text2.count(needle)))
