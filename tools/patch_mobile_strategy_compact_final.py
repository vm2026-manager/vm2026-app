from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_strategy_compact_final_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-strategy-compact-final">'

script = r'''
<script id="mobile-strategy-compact-final">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-strategy-compact-final-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-strategy-compact-final-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Hele strategiboksen */
  .strategy-panel {
    padding: 6px 8px !important;
    margin: 6px 0 6px !important;
    border-radius: 10px !important;
    min-height: 0 !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto !important;
    grid-template-areas:
      "label selected"
      "buttons selected" !important;
    align-items: center !important;
    gap: 5px 8px !important;
    min-height: 0 !important;
  }

  .strategy-topline {
    grid-area: label !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    min-width: 0 !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 10px !important;
    line-height: 1 !important;
    margin: 0 !important;
    white-space: nowrap !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    width: 16px !important;
    min-width: 16px !important;
    height: 16px !important;
    min-height: 16px !important;
    font-size: 9px !important;
    margin: 0 !important;
    transform: none !important;
  }

  .strategy-buttons {
    grid-area: buttons !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    align-items: center !important;
    justify-content: flex-start !important;
    min-width: 0 !important;
  }

  .strategy-btn {
    width: auto !important;
    min-width: 0 !important;
    max-width: 112px !important;
    height: 26px !important;
    min-height: 26px !important;
    padding: 3px 8px !important;
    font-size: 10px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
    white-space: normal !important;
    text-align: center !important;
  }

  .strategy-btn:first-child {
    max-width: 96px !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    grid-area: selected !important;
    align-self: center !important;
    justify-self: end !important;
    width: 132px !important;
    min-width: 132px !important;
    max-width: 132px !important;
    min-height: 34px !important;
    height: auto !important;
    padding: 5px 8px !important;
    font-size: 9px !important;
    line-height: 1.05 !important;
    border-radius: 9px !important;
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 2px !important;
    text-align: left !important;
  }

  .active-slot-badge strong,
  .selected-slot-badge strong,
  .slot-badge strong {
    font-size: 9px !important;
    line-height: 1 !important;
  }
}

@media (max-width: 430px) {
  .strategy-panel {
    padding: 5px 7px !important;
  }

  .strategy-content {
    grid-template-columns: 1fr !important;
    grid-template-areas:
      "label"
      "buttons"
      "selected" !important;
    gap: 5px !important;
  }

  .strategy-topline {
    justify-content: flex-start !important;
  }

  .strategy-buttons {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 4px !important;
    width: 100% !important;
  }

  .strategy-btn {
    width: 100% !important;
    max-width: none !important;
    height: 25px !important;
    min-height: 25px !important;
    padding: 3px 5px !important;
    font-size: 9px !important;
    border-radius: 8px !important;
  }

  .strategy-btn:first-child {
    max-width: none !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    justify-self: stretch !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    min-height: 26px !important;
    padding: 5px 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
    font-size: 9px !important;
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
    print("mobile-strategy-compact-final findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Strategiknapper og valgt plads gjort mere kompakte på mobil.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-strategy-compact-final",
    "grid-template-areas:",
    "repeat(2, minmax(0, 1fr))",
    "Valgt plads",
]:
    print(needle + " => " + str(text2.count(needle)))
