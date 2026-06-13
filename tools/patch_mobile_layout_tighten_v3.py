from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_tighten_v3_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-layout-tighten-v3">'

script = r'''
<script id="mobile-layout-tighten-v3">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-layout-tighten-v3-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-layout-tighten-v3-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Centrér spillerkort og action-knapper bedre */
  .slot,
  .player-slot,
  .squad-slot,
  .field-slot,
  .mobile-pitch-player-slot {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    margin-left: auto !important;
    margin-right: auto !important;
  }

  .slot-actions,
  .player-actions,
  .mobile-pitch-player-slot .slot-actions {
    position: static !important;
    left: auto !important;
    right: auto !important;
    transform: none !important;
    margin: 2px auto 0 auto !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 4px !important;
    width: 100% !important;
  }

  .slot button,
  .player-slot button,
  .squad-slot button,
  .field-slot button,
  .mobile-pitch-player-slot button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
  }

  /* Strategiområdet gøres markant mindre */
  .strategy-panel {
    padding: 6px 8px !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: 58px 1fr !important;
    gap: 6px !important;
    align-items: start !important;
  }

  .strategy-topline {
    gap: 4px !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 10px !important;
    line-height: 1 !important;
    margin: 0 !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    transform: scale(0.78) !important;
    margin-top: 1px !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    align-items: flex-start !important;
  }

  .strategy-btn {
    height: 24px !important;
    min-height: 24px !important;
    padding: 2px 7px !important;
    font-size: 9px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    min-height: 24px !important;
    padding: 3px 8px !important;
    font-size: 9px !important;
    border-radius: 8px !important;
  }

  /* Formation og actions lidt strammere */
  .formation-select-wrap,
  .action-bar {
    padding: 6px 8px !important;
    border-radius: 10px !important;
    gap: 6px !important;
    margin-bottom: 6px !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 10px !important;
    line-height: 1 !important;
  }

  .formation-select-wrap select,
  .formation-select {
    height: 26px !important;
    min-height: 26px !important;
    padding: 2px 6px !important;
    font-size: 10px !important;
    border-radius: 8px !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 28px !important;
    min-height: 28px !important;
    padding: 3px 7px !important;
    font-size: 9px !important;
    line-height: 1 !important;
    border-radius: 9px !important;
  }

  /* Hold toppen kompakt */
  .team-header {
    margin-bottom: 6px !important;
  }

  .bank-row {
    gap: 5px !important;
  }

  .bank-box {
    padding: 5px 7px !important;
  }

  .bank-box strong {
    font-size: 10px !important;
  }

  .bank-box div,
  #spentValue,
  #bankValue,
  #playerCountValue {
    font-size: 9px !important;
    line-height: 1.05 !important;
  }

  /* Statusboks lidt mindre */
  .status-box,
  .status-message {
    padding: 7px 9px !important;
    font-size: 11px !important;
    border-radius: 9px !important;
    margin-bottom: 6px !important;
  }
}

@media (max-width: 430px) {
  .strategy-content {
    grid-template-columns: 50px 1fr !important;
  }

  .strategy-btn {
    height: 23px !important;
    min-height: 23px !important;
    font-size: 8.7px !important;
    padding: 2px 6px !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    min-height: 23px !important;
    font-size: 8.7px !important;
    padding: 3px 7px !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 27px !important;
    min-height: 27px !important;
    font-size: 8.7px !important;
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
    print("mobile-layout-tighten-v3 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet mobil-komprimering v3.")
    print(f"Backup: {backup}")
    print("- Fjern/Skift centreres bedre")
    print("- Strategiområdet gøres mindre")
    print("- Hvid boks og valgt plads strammes")
    print("- Formation og handlingsknapper gøres mindre")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-layout-tighten-v3",
    "mobile-layout-tighten-v3-style",
    "grid-template-columns: 58px 1fr",
    "height: 24px !important",
    "position: static !important",
]:
    print(needle + " => " + str(text2.count(needle)))
