from pathlib import Path
from datetime import datetime
import re
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_safe_mobile_strategy_fix_{stamp}.html")
shutil.copy2(p, backup)

# Fjern tidligere mobile strategy script-patches, som kan konflikte
script_ids = [
    "mobile-strategy-mockup-final",
    "mobile-strategy-bar-compact",
    "mobile-strategy-compact-final",
    "mobile-strategy-mockup"
]

removed = 0
for sid in script_ids:
    pattern = re.compile(
        r'\n?<script id="' + re.escape(sid) + r'">.*?</script>\s*',
        re.DOTALL
    )
    text, n = pattern.subn("\n", text)
    removed += n

new_script = r'''
<script id="mobile-strategy-safe-compact">
(function () {
  function injectStyle() {
    if (document.getElementById("mobile-strategy-safe-compact-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-strategy-safe-compact-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Strategiboks: mindre og mere stabil */
  .strategy-panel,
  .strategy-box,
  .strategy-section {
    padding: 10px 12px !important;
    margin: 8px 0 !important;
    border-radius: 14px !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-content,
  .strategy-inner {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    align-items: stretch !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-topline,
  .strategy-header {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
    margin: 0 0 2px 0 !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-heading,
  .strategy-panel h3,
  .strategy-box h3,
  .strategy-section h3 {
    font-size: 13px !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    white-space: nowrap !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn,
  .strategy-box .info-icon,
  .strategy-box .strategy-info,
  .strategy-box .info-btn {
    width: 22px !important;
    min-width: 22px !important;
    height: 22px !important;
    min-height: 22px !important;
    font-size: 12px !important;
    margin: 0 !important;
  }

  /* Strategiknapper: 2x2, mindre, men stadig læsbare */
  .strategy-buttons,
  .strategy-options {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
    width: 100% !important;
    margin: 0 !important;
  }

  .strategy-btn,
  .strategy-option-btn {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 42px !important;
    min-height: 42px !important;
    padding: 6px 8px !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    line-height: 1.1 !important;
    text-align: center !important;
    white-space: normal !important;
  }

  /* Valgt plads: gør den mindre hvis den stadig er i strategiområdet */
  .selected-slot-badge,
  .active-slot-badge,
  .slot-badge {
    width: auto !important;
    max-width: 100% !important;
    min-height: 34px !important;
    padding: 6px 10px !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    line-height: 1.1 !important;
    margin-top: 4px !important;
    align-self: flex-start !important;
  }

  /* Formation-række mere kompakt */
  .formation-select-wrap,
  .formation-row,
  .formation-control {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
    padding: 8px 10px !important;
    margin: 8px 0 !important;
    border-radius: 12px !important;
    min-height: 0 !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 13px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    margin: 0 !important;
  }

  .formation-select-wrap select,
  .formation-select,
  .formation-row select {
    width: 96px !important;
    min-width: 96px !important;
    max-width: 96px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 4px 8px !important;
    font-size: 13px !important;
    border-radius: 10px !important;
  }

  /* Handlingsknapper mindre og pænere */
  .action-bar,
  .team-actions,
  .action-buttons {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 8px !important;
    margin: 8px 0 !important;
  }

  .action-bar button,
  .team-actions button,
  .action-buttons button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    width: 100% !important;
    min-width: 0 !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 10px !important;
    font-size: 12px !important;
    border-radius: 12px !important;
    white-space: nowrap !important;
  }
}

@media (max-width: 430px) {
  .strategy-panel,
  .strategy-box,
  .strategy-section {
    padding: 8px 10px !important;
  }

  .strategy-buttons,
  .strategy-options {
    gap: 6px !important;
  }

  .strategy-btn,
  .strategy-option-btn {
    height: 40px !important;
    min-height: 40px !important;
    font-size: 12px !important;
    padding: 5px 6px !important;
  }

  .formation-select-wrap,
  .formation-row,
  .formation-control {
    padding: 7px 8px !important;
    gap: 7px !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 12px !important;
  }

  .formation-select-wrap select,
  .formation-select,
  .formation-row select {
    width: 90px !important;
    min-width: 90px !important;
    max-width: 90px !important;
    height: 34px !important;
    min-height: 34px !important;
    font-size: 12px !important;
  }

  .selected-slot-badge,
  .active-slot-badge,
  .slot-badge {
    min-height: 32px !important;
    padding: 5px 8px !important;
    font-size: 11px !important;
  }

  .action-bar button,
  .team-actions button,
  .action-buttons button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 38px !important;
    min-height: 38px !important;
    font-size: 11px !important;
  }
}
`;
    document.head.appendChild(style);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectStyle);
  } else {
    injectStyle();
  }
}());
</script>
'''

if 'id="mobile-strategy-safe-compact"' not in text:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body> i index.html.")
    text = text.replace("</body>", new_script + "\n</body>", 1)

p.write_text(text, encoding="utf-8")

print("OK: Farlig mobil-strategy patch fjernet og erstattet med safe compact patch.")
print(f"Scriptblokke fjernet: {removed}")
print(f"Backup: {backup}")

check = p.read_text(encoding="utf-8")
print("")
print("Sanity:")
for needle in [
    "mobile-strategy-safe-compact",
    "grid-template-columns: repeat(2, minmax(0, 1fr))",
    "height: 42px",
    "height: 40px"
]:
    print(needle + " => " + str(check.count(needle)))
