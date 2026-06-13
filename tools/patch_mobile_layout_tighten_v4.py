from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_tighten_v4_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-layout-tighten-v4">'

script = r'''
<script id="mobile-layout-tighten-v4">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-layout-tighten-v4-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-layout-tighten-v4-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Flag på banen: større og centreret */
  .player-card img,
  .slot-card img,
  .squad-card img,
  .mobile-pitch-player-card img,
  .player-card .flag,
  .slot-card .flag,
  .squad-card .flag,
  .mobile-pitch-player-card .flag,
  .player-flag {
    display: block !important;
    position: absolute !important;
    left: 50% !important;
    top: -13px !important;
    transform: translateX(-50%) !important;
    width: 28px !important;
    height: 20px !important;
    max-width: 28px !important;
    max-height: 20px !important;
    margin: 0 !important;
    object-fit: cover !important;
    border-radius: 4px !important;
    z-index: 3 !important;
  }

  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    position: relative !important;
    padding-top: 13px !important;
  }

  /* Strategiboksen: fra stor kortboks til kompakt bjælke */
  .strategy-panel {
    padding: 5px 7px !important;
    border-radius: 10px !important;
    margin: 5px 0 6px !important;
    min-height: 0 !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: auto 1fr auto !important;
    gap: 5px !important;
    align-items: center !important;
    min-height: 0 !important;
  }

  .strategy-topline {
    display: inline-flex !important;
    align-items: center !important;
    gap: 3px !important;
    min-width: 0 !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 9px !important;
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
    transform: none !important;
    margin: 0 !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 3px !important;
    align-items: center !important;
    min-width: 0 !important;
  }

  .strategy-btn {
    width: auto !important;
    min-width: 0 !important;
    max-width: 120px !important;
    height: 22px !important;
    min-height: 22px !important;
    padding: 2px 6px !important;
    font-size: 8.5px !important;
    line-height: 1 !important;
    border-radius: 7px !important;
    white-space: nowrap !important;
  }

  /* Gør den lange tekst kortere visuelt, så knappen ikke bliver høj */
  .strategy-btn {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    justify-self: end !important;
    min-width: 116px !important;
    max-width: 132px !important;
    min-height: 22px !important;
    height: 22px !important;
    padding: 2px 7px !important;
    font-size: 8.5px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 5px !important;
    white-space: nowrap !important;
  }

  .active-slot-badge strong,
  .selected-slot-badge strong,
  .slot-badge strong {
    font-size: 8.5px !important;
    line-height: 1 !important;
  }

  /* Formation og handlingsområde lidt mere kompakt */
  .formation-select-wrap {
    padding: 5px 7px !important;
    margin-bottom: 5px !important;
    border-radius: 9px !important;
  }

  .action-bar {
    padding: 0 !important;
    margin: 5px 0 7px !important;
    gap: 5px !important;
    background: transparent !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 9px !important;
  }

  .formation-select-wrap select,
  .formation-select {
    height: 24px !important;
    min-height: 24px !important;
    padding: 2px 6px !important;
    font-size: 9.5px !important;
    border-radius: 8px !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 26px !important;
    min-height: 26px !important;
    padding: 3px 7px !important;
    font-size: 8.8px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
  }
}

@media (max-width: 430px) {
  .strategy-content {
    grid-template-columns: auto 1fr !important;
  }

  .strategy-topline {
    grid-column: 1 !important;
  }

  .strategy-buttons {
    grid-column: 2 !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    grid-column: 1 / 3 !important;
    justify-self: end !important;
    margin-top: 3px !important;
    height: 21px !important;
    min-height: 21px !important;
    max-width: 145px !important;
  }

  .strategy-btn {
    height: 21px !important;
    min-height: 21px !important;
    font-size: 8px !important;
    padding: 2px 5px !important;
    max-width: 98px !important;
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
    width: 26px !important;
    height: 19px !important;
    max-width: 26px !important;
    max-height: 19px !important;
    top: -12px !important;
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
    print("mobile-layout-tighten-v4 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet mobil-komprimering v4.")
    print(f"Backup: {backup}")
    print("- Flag centreret og gjort lidt større")
    print("- Strategiboks gjort markant lavere")
    print("- Strategiknapper gjort til små chips")
    print("- Valgt plads gjort mindre")
    print("- Formation/action-bar strammet")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-layout-tighten-v4",
    "mobile-layout-tighten-v4-style",
    "grid-template-columns: auto 1fr auto",
    "height: 22px !important",
    "top: -13px !important",
]:
    print(needle + " => " + str(text2.count(needle)))
