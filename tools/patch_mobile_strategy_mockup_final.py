from pathlib import Path
from datetime import datetime
import re
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_strategy_mockup_final_{stamp}.html")
shutil.copy2(p, backup)

# Fjern kun tidligere strategi-patches, ikke de brede mobil/pitch fixes.
old_ids = [
    "mobile-strategy-bar-compact",
    "mobile-strategy-compact-final",
]

removed = 0
for sid in old_ids:
    pattern = re.compile(
        r'\n?<script id="' + re.escape(sid) + r'">.*?</script>\s*',
        re.DOTALL
    )
    text, n = pattern.subn("\n", text)
    removed += n

marker = '<script id="mobile-strategy-mockup-final">'

script = r'''
<script id="mobile-strategy-mockup-final">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-strategy-mockup-final-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-strategy-mockup-final-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Strategikort som i mockuppet: lavt, kompakt, uden kæmpe tomrum */
  .strategy-panel {
    padding: 8px 9px !important;
    margin: 6px 0 6px !important;
    border-radius: 11px !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-content {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    justify-content: flex-start !important;
    gap: 7px !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-topline {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 6px !important;
    min-width: 0 !important;
    margin: 0 !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 12px !important;
    line-height: 1 !important;
    margin: 0 !important;
    white-space: nowrap !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    width: 20px !important;
    min-width: 20px !important;
    height: 20px !important;
    min-height: 20px !important;
    font-size: 11px !important;
    line-height: 1 !important;
    margin: 0 !important;
    transform: none !important;
  }

  .strategy-buttons {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 6px !important;
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
  }

  .strategy-btn {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 4px 7px !important;
    font-size: 12px !important;
    line-height: 1.05 !important;
    border-radius: 10px !important;
    white-space: normal !important;
    text-align: center !important;
    overflow: hidden !important;
    text-overflow: clip !important;
  }

  /* Formation + valgt plads på samme lave række */
  .formation-select-wrap {
    display: grid !important;
    grid-template-columns: auto 86px minmax(0, 1fr) !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 7px 8px !important;
    margin: 6px 0 7px !important;
    border-radius: 11px !important;
    min-height: 0 !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 12px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
  }

  .formation-select-wrap select,
  .formation-select {
    width: 86px !important;
    min-width: 86px !important;
    max-width: 86px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 3px 7px !important;
    font-size: 13px !important;
    border-radius: 9px !important;
  }

  .mobile-selected-slot-inline {
    justify-self: stretch !important;
    align-self: center !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 9px !important;
    border-radius: 9px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
    font-size: 11px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
  }

  .mobile-selected-slot-inline strong,
  .mobile-selected-slot-inline b {
    font-size: 11px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
  }

  .mobile-selected-slot-inline span,
  .mobile-selected-slot-inline div {
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }

  /* Hvis valgt-plads stadig ligger tilbage i strategiboksen, skjules den der */
  .strategy-panel .active-slot-badge,
  .strategy-panel .selected-slot-badge,
  .strategy-panel .slot-badge {
    display: none !important;
  }

  /* Handlingsknapper lidt mere som mockuppet */
  .action-bar {
    gap: 6px !important;
    margin: 6px 0 8px !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 34px !important;
    min-height: 34px !important;
    padding: 5px 8px !important;
    font-size: 11px !important;
    border-radius: 10px !important;
    white-space: nowrap !important;
  }
}

@media (max-width: 430px) {
  .strategy-panel {
    padding: 7px 8px !important;
  }

  .strategy-buttons {
    gap: 5px !important;
  }

  .strategy-btn {
    height: 32px !important;
    min-height: 32px !important;
    font-size: 11px !important;
    padding: 4px 5px !important;
  }

  .formation-select-wrap {
    grid-template-columns: auto 82px minmax(0, 1fr) !important;
    gap: 6px !important;
    padding: 6px 7px !important;
  }

  .formation-select-wrap label,
  .formation-label {
    font-size: 11px !important;
  }

  .formation-select-wrap select,
  .formation-select {
    width: 82px !important;
    min-width: 82px !important;
    max-width: 82px !important;
    height: 30px !important;
    min-height: 30px !important;
    font-size: 12px !important;
  }

  .mobile-selected-slot-inline {
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 8px !important;
    font-size: 10px !important;
  }

  .mobile-selected-slot-inline strong,
  .mobile-selected-slot-inline b {
    font-size: 10px !important;
  }

  .action-bar button,
  .primary-action-btn,
  .secondary-action-btn,
  .danger-action-btn {
    height: 32px !important;
    min-height: 32px !important;
    font-size: 10px !important;
    padding: 4px 7px !important;
  }
}
`;
    document.head.appendChild(style);
  }

  function findSelectedBadge() {
    return document.querySelector(
      ".active-slot-badge, .selected-slot-badge, .slot-badge"
    );
  }

  function findFormationRow() {
    return document.querySelector(
      ".formation-select-wrap, .formation-row, .formation-control, .formation-panel"
    );
  }

  function moveSelectedBadge() {
    var badge = findSelectedBadge();
    var formation = findFormationRow();

    if (!badge || !formation) return;

    if (!badge.__mobileOriginalParent) {
      badge.__mobileOriginalParent = badge.parentElement;
      badge.__mobileOriginalNext = badge.nextSibling;
    }

    if (isMobile()) {
      if (badge.parentElement !== formation) {
        formation.appendChild(badge);
      }
      badge.classList.add("mobile-selected-slot-inline");
      badge.style.removeProperty("display");
    } else {
      if (badge.__mobileOriginalParent && badge.parentElement !== badge.__mobileOriginalParent) {
        if (badge.__mobileOriginalNext && badge.__mobileOriginalNext.parentElement === badge.__mobileOriginalParent) {
          badge.__mobileOriginalParent.insertBefore(badge, badge.__mobileOriginalNext);
        } else {
          badge.__mobileOriginalParent.appendChild(badge);
        }
      }
      badge.classList.remove("mobile-selected-slot-inline");
    }
  }

  function install() {
    addStyle();

    function run() {
      moveSelectedBadge();
      setTimeout(moveSelectedBadge, 150);
      setTimeout(moveSelectedBadge, 500);
    }

    run();

    if (!window.__mobileStrategyMockupFinalInstalled) {
      window.__mobileStrategyMockupFinalInstalled = true;
      window.addEventListener("resize", run);
      window.addEventListener("orientationchange", function () {
        setTimeout(run, 250);
      });
      setInterval(moveSelectedBadge, 1000);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
}());
</script>
'''

if marker not in text:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)

p.write_text(text, encoding="utf-8")

print("OK: Mobil strategi/formation ombygget efter mockup.")
print(f"Gamle strategi-scriptblokke fjernet: {removed}")
print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-strategy-mockup-final",
    "mobile-selected-slot-inline",
    "grid-template-columns: repeat(2, minmax(0, 1fr))",
    "formation.appendChild(badge)",
]:
    print(needle + " => " + str(text2.count(needle)))
