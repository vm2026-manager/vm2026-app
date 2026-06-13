from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_strategy_bar_compact_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-strategy-bar-compact">'

script = r'''
<script id="mobile-strategy-bar-compact">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-strategy-bar-compact-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-strategy-bar-compact-style";
    style.textContent = `
@media (max-width: 700px) {
  .strategy-panel {
    min-height: 0 !important;
    padding: 5px 6px !important;
    margin: 5px 0 6px !important;
    border-radius: 10px !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: auto 1fr auto !important;
    align-items: center !important;
    gap: 5px !important;
    min-height: 0 !important;
  }

  .strategy-topline {
    display: inline-flex !important;
    align-items: center !important;
    gap: 3px !important;
    min-width: 0 !important;
  }

  .strategy-label {
    font-size: 9px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    width: 15px !important;
    height: 15px !important;
    min-width: 15px !important;
    min-height: 15px !important;
    font-size: 9px !important;
    margin: 0 !important;
    transform: none !important;
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
    max-width: 92px !important;
    height: 22px !important;
    min-height: 22px !important;
    padding: 2px 6px !important;
    font-size: 8.5px !important;
    line-height: 1 !important;
    border-radius: 7px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .active-slot-badge {
    justify-self: end !important;
    min-width: 102px !important;
    max-width: 118px !important;
    height: 22px !important;
    min-height: 22px !important;
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

  .active-slot-badge strong {
    font-size: 8.5px !important;
    line-height: 1 !important;
  }
}

@media (max-width: 430px) {
  .strategy-content {
    grid-template-columns: auto 1fr !important;
  }

  .strategy-buttons {
    justify-content: flex-start !important;
  }

  .strategy-btn {
    max-width: 78px !important;
    height: 21px !important;
    min-height: 21px !important;
    font-size: 8px !important;
    padding: 2px 5px !important;
  }

  .active-slot-badge {
    grid-column: 1 / 3 !important;
    justify-self: end !important;
    margin-top: 3px !important;
    height: 21px !important;
    min-height: 21px !important;
    max-width: 130px !important;
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
    print("mobile-strategy-bar-compact findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Strategiboksen er gjort mere kompakt på mobil.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-strategy-bar-compact",
    "mobile-strategy-bar-compact-style",
    "grid-template-columns: auto 1fr auto",
    "height: 22px !important",
]:
    print(needle + " => " + str(text2.count(needle)))
