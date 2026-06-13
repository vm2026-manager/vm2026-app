from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_tighten_v2_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-layout-tighten-v2">'

script = r'''
<script id="mobile-layout-tighten-v2">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-layout-tighten-v2-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-layout-tighten-v2-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Overordnet topområde gøres mere kompakt */
  .left-panel,
  .right-panel {
    padding: 6px !important;
  }

  .team-header {
    gap: 6px !important;
    margin-bottom: 6px !important;
  }

  .app-name {
    font-size: 11px !important;
    line-height: 1.05 !important;
  }

  .team-title {
    font-size: 17px !important;
    line-height: 1 !important;
    margin-top: 1px !important;
  }

  .bank-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
    width: 100% !important;
  }

  .bank-box {
    min-width: 0 !important;
    padding: 6px 8px !important;
    border-radius: 10px !important;
  }

  .bank-box strong {
    font-size: 11px !important;
  }

  .bank-box div,
  #spentValue,
  #bankValue,
  #playerCountValue {
    font-size: 10px !important;
    line-height: 1.1 !important;
  }

  .manual-bank-input {
    width: 100% !important;
    max-width: 70px !important;
    min-height: 20px !important;
    height: 20px !important;
    margin-top: 2px !important;
    padding: 1px 5px !important;
    font-size: 11px !important;
  }

  /* Strategiområdet skal være meget mindre */
  .strategy-panel {
    padding: 8px !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
  }

  .strategy-content,
  .strategy-topline {
    gap: 6px !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    transform: scale(0.9) !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 11px !important;
    line-height: 1.05 !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    align-items: flex-start !important;
  }

  .strategy-btn {
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 3px 8px !important;
    font-size: 10px !important;
    line-height: 1 !important;
    border-radius: 9px !important;
    white-space: nowrap !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    min-height: 28px !important;
    padding: 4px 8px !important;
    font-size: 10px !important;
    border-radius: 10px !important;
  }

  /* Formation og handlingsknapper komprimeres */
  .formation-select-wrap,
  .action-bar {
    padding: 8px !important;
    border-radius: 12px !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 11px !important;
    line-height: 1.05 !important;
  }

  .formation-select-wrap select,
  .formation-select {
    height: 28px !important;
    min-height: 28px !important;
    padding: 3px 7px !important;
    font-size: 11px !important;
    border-radius: 9px !important;
  }

  .action-bar {
    display: flex !important;
    flex-wrap: wrap !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 30px !important;
    min-height: 30px !important;
    padding: 4px 8px !important;
    font-size: 10px !important;
    line-height: 1 !important;
    border-radius: 10px !important;
  }

  /* Statusboks lidt mindre */
  .status-box,
  .status-message {
    padding: 8px 10px !important;
    font-size: 12px !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
  }

  /* Banen og spillerkort strammes yderligere */
  .pitch {
    height: 545px !important;
    min-height: 545px !important;
    margin-top: 6px !important;
    border-radius: 12px !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 58px !important;
    min-width: 58px !important;
    max-width: 58px !important;
    gap: 1px !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 54px !important;
    min-width: 54px !important;
    max-width: 54px !important;
    padding: 8px 2px 2px !important;
    border-radius: 7px !important;
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
    width: 22px !important;
    height: 16px !important;
    max-width: 22px !important;
    max-height: 16px !important;
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
    max-width: 50px !important;
    font-size: 6.8px !important;
    line-height: 0.9 !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    font-size: 6.7px !important;
    line-height: 0.9 !important;
  }

  .player-next,
  .player-start,
  .slot .player-next,
  .mobile-pitch-player-card .player-next {
    max-width: 50px !important;
    font-size: 6.2px !important;
    line-height: 0.9 !important;
  }

  .slot-actions,
  .player-actions,
  .mobile-pitch-player-slot .slot-actions {
    gap: 2px !important;
    width: 58px !important;
    max-width: 58px !important;
  }

  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button,
  .mobile-pitch-player-slot button {
    min-width: 24px !important;
    width: 24px !important;
    max-width: 24px !important;
    height: 16px !important;
    min-height: 16px !important;
    padding: 0 1px !important;
    font-size: 7.4px !important;
    line-height: 1 !important;
  }

  .empty-slot-label,
  .slot-empty-label {
    font-size: 6.8px !important;
    line-height: 0.95 !important;
  }

  .slot-empty-select-btn,
  .empty-slot-btn {
    min-width: 28px !important;
    height: 16px !important;
    min-height: 16px !important;
    padding: 0 4px !important;
    font-size: 7.5px !important;
  }

  /* Højre spillerliste strammes lidt mere */
  .right-panel {
    margin-top: 0 !important;
  }

  .right-mode-actions {
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
  }

  .trade-search input,
  .trade-search select,
  .clearable-select-wrap,
  .toggle-chip,
  #searchInput,
  #teamFilter,
  #positionFilter,
  #sortSelect {
    font-size: 12px !important;
    min-height: 34px !important;
  }

  .favorite-filter-btn,
  #favoriteFilterBtn,
  .alt-btn,
  .alternatives-btn {
    min-height: 34px !important;
    height: 34px !important;
    padding: 6px 10px !important;
    font-size: 12px !important;
    border-radius: 10px !important;
  }

  .trade-row {
    grid-template-columns: minmax(0, 1fr) 28px 74px 36px 38px !important;
    column-gap: 4px !important;
    padding: 6px 3px 6px 4px !important;
  }

  .trade-name {
    font-size: 13px !important;
  }

  .trade-meta,
  .trade-next {
    font-size: 10px !important;
  }

  .trade-position-value,
  .mobile-position-pill {
    min-width: 24px !important;
    max-width: 26px !important;
    padding: 4px 3px !important;
    font-size: 9px !important;
  }

  .trade-price {
    width: 74px !important;
    max-width: 74px !important;
    font-size: 11px !important;
  }

  .trade-price-label {
    font-size: 9px !important;
  }

  .trade-row .favorite-btn,
  .favorite-btn {
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    font-size: 16px !important;
  }

  .trade-row .plus-btn,
  .plus-btn {
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
    height: 34px !important;
    min-height: 34px !important;
    max-height: 34px !important;
    font-size: 16px !important;
  }
}

@media (max-width: 430px) {
  .pitch {
    height: 520px !important;
    min-height: 520px !important;
  }

  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    width: 54px !important;
    min-width: 54px !important;
    max-width: 54px !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    width: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
  }

  .slot-actions,
  .player-actions,
  .mobile-pitch-player-slot .slot-actions {
    width: 54px !important;
    max-width: 54px !important;
  }

  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button,
  .mobile-pitch-player-slot button {
    min-width: 22px !important;
    width: 22px !important;
    max-width: 22px !important;
    font-size: 7px !important;
  }

  .strategy-btn {
    font-size: 9.5px !important;
    padding: 3px 7px !important;
  }

  .action-bar button {
    font-size: 9.5px !important;
    padding: 3px 7px !important;
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
    print("mobile-layout-tighten-v2 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet ekstra mobil-komprimering v2.")
    print(f"Backup: {backup}")
    print("- Spillerkort gjort mindre igen")
    print("- Top/budget/strategi/formation gjort mere kompakt")
    print("- Handlingsknapper gjort mindre")
    print("- Højre spillerliste strammet yderligere")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-layout-tighten-v2",
    "mobile-layout-tighten-v2-style",
    "height: 520px !important",
    "width: 50px !important",
    "grid-template-columns: minmax(0, 1fr) 28px 74px 36px 38px",
]:
    print(needle + " => " + str(text2.count(needle)))
