from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_right_panel_cleanup_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-right-panel-cleanup">'

script = r'''
<script id="mobile-right-panel-cleanup">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-right-panel-cleanup-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-right-panel-cleanup-style";
    style.textContent = `
@media (max-width: 700px) {
  html,
  body {
    max-width: 100% !important;
    overflow-x: hidden !important;
  }

  .right-panel,
  .side-panel,
  .players-panel,
  .player-list-panel {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
  }

  .right-panel *,
  .side-panel *,
  .players-panel *,
  .player-list-panel * {
    max-width: 100%;
    box-sizing: border-box;
  }

  /* Login/status-række i spillerlisten */
  .right-panel .auth-row,
  .right-panel .login-row,
  .right-panel .user-row,
  .right-panel .account-row,
  .right-panel [class*="auth"],
  .right-panel [class*="login"],
  .right-panel [class*="user"] {
    max-width: 100% !important;
  }

  .right-panel input[type="email"],
  .right-panel input[type="text"],
  .right-panel .email-display,
  .right-panel .user-email,
  .right-panel [class*="email"] {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }

  .right-panel button {
    max-width: 100% !important;
  }

  /* Topknapper i spillerliste */
  .right-panel .player-list-actions,
  .right-panel .list-actions,
  .right-panel .favorites-row,
  .right-panel .alternatives-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 8px !important;
    width: 100% !important;
    max-width: 100% !important;
  }

  .right-panel .player-list-actions button,
  .right-panel .list-actions button,
  .right-panel .favorites-row button,
  .right-panel .alternatives-row button {
    width: 100% !important;
    min-width: 0 !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 6px 8px !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  /* Søgefelt og filtre */
  .right-panel input,
  .right-panel select {
    width: 100% !important;
    min-width: 0 !important;
  }

  .right-panel .filters,
  .right-panel .filter-panel,
  .right-panel .filter-controls,
  .right-panel .player-filters {
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 7px !important;
    width: 100% !important;
    max-width: 100% !important;
  }

  .right-panel .filters > *,
  .right-panel .filter-panel > *,
  .right-panel .filter-controls > *,
  .right-panel .player-filters > * {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
  }

  /* Den brede toggle/range/filter-linje der stikker ud til højre */
  .right-panel label,
  .right-panel .toggle-row,
  .right-panel .switch-row,
  .right-panel .range-row,
  .right-panel .filter-row {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    overflow: hidden !important;
  }

  .right-panel label span,
  .right-panel .toggle-row span,
  .right-panel .switch-row span,
  .right-panel .range-row span,
  .right-panel .filter-row span {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }

  .right-panel input[type="range"] {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
    max-width: 100% !important;
  }

  .right-panel input[type="checkbox"],
  .right-panel input[type="radio"] {
    flex: 0 0 auto !important;
    width: auto !important;
  }

  /* Spillerækker holdes inden for skærmen */
  .player-row,
  .player-list-row,
  .player-item {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
  }

  .player-row-name,
  .player-name-cell,
  .player-meta,
  .player-title {
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }
}

@media (max-width: 430px) {
  .right-panel,
  .side-panel,
  .players-panel,
  .player-list-panel {
    padding-left: 7px !important;
    padding-right: 7px !important;
  }

  .right-panel .player-list-actions button,
  .right-panel .list-actions button,
  .right-panel .favorites-row button,
  .right-panel .alternatives-row button {
    height: 34px !important;
    min-height: 34px !important;
    font-size: 12px !important;
  }

  .right-panel input,
  .right-panel select {
    min-height: 34px !important;
    font-size: 13px !important;
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
    print("mobile-right-panel-cleanup findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Mobil spillerliste/højre panel strammet.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-right-panel-cleanup",
    "overflow-x: hidden",
    "grid-template-columns: 1fr 1fr",
    "input[type=\"range\"]",
]:
    print(needle + " => " + str(text2.count(needle)))
