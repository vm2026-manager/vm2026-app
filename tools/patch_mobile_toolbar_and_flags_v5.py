from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_toolbar_and_flags_v5_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-toolbar-and-flags-v5">'

script = r'''
<script id="mobile-toolbar-and-flags-v5">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-toolbar-and-flags-v5-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-toolbar-and-flags-v5-style";
    style.textContent = `
@media (max-width: 700px) {
  /* Strategiboks som lav toolbar */
  .strategy-panel {
    display: block !important;
    padding: 4px 6px !important;
    margin: 4px 0 5px !important;
    border-radius: 9px !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-content {
    display: grid !important;
    grid-template-columns: auto minmax(0, 1fr) auto !important;
    align-items: center !important;
    gap: 4px !important;
    min-height: 0 !important;
    height: auto !important;
  }

  .strategy-topline {
    display: inline-flex !important;
    align-items: center !important;
    gap: 2px !important;
    min-width: 0 !important;
    max-width: 74px !important;
  }

  .strategy-label,
  .strategy-title,
  .strategy-panel h3,
  .strategy-heading {
    font-size: 8px !important;
    line-height: 1 !important;
    margin: 0 !important;
    white-space: nowrap !important;
  }

  .strategy-panel .info-icon,
  .strategy-panel .strategy-info,
  .strategy-panel .info-btn {
    width: 14px !important;
    min-width: 14px !important;
    height: 14px !important;
    min-height: 14px !important;
    font-size: 8px !important;
    transform: none !important;
    margin: 0 !important;
  }

  .strategy-buttons {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 3px !important;
    align-items: center !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
    min-width: 0 !important;
    padding: 0 !important;
  }

  .strategy-buttons::-webkit-scrollbar {
    display: none !important;
  }

  .strategy-btn {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 42px !important;
    max-width: 58px !important;
    height: 20px !important;
    min-height: 20px !important;
    padding: 1px 5px !important;
    font-size: 7.8px !important;
    line-height: 1 !important;
    border-radius: 7px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  .strategy-btn.active {
    border-width: 1px !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    justify-self: end !important;
    min-width: 86px !important;
    max-width: 108px !important;
    width: auto !important;
    height: 20px !important;
    min-height: 20px !important;
    padding: 1px 6px !important;
    font-size: 7.8px !important;
    line-height: 1 !important;
    border-radius: 7px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 4px !important;
    white-space: nowrap !important;
  }

  .active-slot-badge strong,
  .selected-slot-badge strong,
  .slot-badge strong {
    font-size: 7.8px !important;
    line-height: 1 !important;
  }

  /* Flag centreret relativt til selve mobil-spillerkortet */
  .mobile-pitch-player-card {
    position: relative !important;
    overflow: visible !important;
    padding-top: 15px !important;
  }

  .mobile-pitch-player-card .mobile-centered-flag,
  .mobile-pitch-player-card img.mobile-centered-flag,
  .mobile-pitch-player-card .flag.mobile-centered-flag,
  .mobile-pitch-player-card .player-flag.mobile-centered-flag {
    position: absolute !important;
    left: 50% !important;
    right: auto !important;
    top: -12px !important;
    transform: translateX(-50%) !important;
    display: block !important;
    width: 29px !important;
    height: 21px !important;
    max-width: 29px !important;
    max-height: 21px !important;
    min-width: 29px !important;
    min-height: 21px !important;
    margin: 0 !important;
    object-fit: cover !important;
    border-radius: 4px !important;
    z-index: 4 !important;
  }

  /* Fjern tidligere flag-positioner i pitch cards */
  .mobile-pitch-player-card img:not(.mobile-centered-flag),
  .mobile-pitch-player-card .flag:not(.mobile-centered-flag),
  .mobile-pitch-player-card .player-flag:not(.mobile-centered-flag) {
    margin-left: auto !important;
    margin-right: auto !important;
  }

  /* Knapper under spillerkort helt centreret */
  .mobile-pitch-player-slot {
    align-items: center !important;
    text-align: center !important;
  }

  .mobile-pitch-player-slot .slot-actions,
  .mobile-pitch-player-slot .player-actions {
    width: 100% !important;
    max-width: 54px !important;
    margin: 2px auto 0 !important;
    justify-content: center !important;
    gap: 3px !important;
  }
}

@media (max-width: 430px) {
  .strategy-content {
    grid-template-columns: auto minmax(0, 1fr) auto !important;
  }

  .strategy-topline {
    max-width: 58px !important;
  }

  .strategy-label {
    font-size: 7.5px !important;
  }

  .strategy-btn {
    min-width: 38px !important;
    max-width: 50px !important;
    height: 19px !important;
    min-height: 19px !important;
    font-size: 7.3px !important;
    padding: 1px 4px !important;
  }

  .active-slot-badge,
  .selected-slot-badge,
  .slot-badge {
    min-width: 78px !important;
    max-width: 96px !important;
    height: 19px !important;
    min-height: 19px !important;
    font-size: 7.3px !important;
  }
}
`;
    document.head.appendChild(style);
  }

  function shortLabel(text) {
    var t = (text || "").replace(/\s+/g, " ").trim();

    if (/næste/i.test(t)) return "Næste";
    if (/1\.\s*\+\s*2/i.test(t)) return "1+2";
    if (/gruppe/i.test(t)) return "Gruppe";
    if (/lang/i.test(t)) return "Lang";

    return t;
  }

  function compactStrategyButtons() {
    if (!isMobile()) return;

    document.querySelectorAll(".strategy-btn").forEach(function (btn) {
      if (!btn.dataset.fullMobileLabel) {
        btn.dataset.fullMobileLabel = (btn.textContent || "").replace(/\s+/g, " ").trim();
      }
      btn.textContent = shortLabel(btn.dataset.fullMobileLabel);
    });

    document.querySelectorAll(".strategy-label, .strategy-title, .strategy-heading").forEach(function (el) {
      var txt = (el.textContent || "").replace(/\s+/g, " ").trim();
      if (/vælg strategi/i.test(txt)) {
        el.textContent = "Strategi";
      }
    });
  }

  function markPitchFlags() {
    if (!isMobile()) return;

    document.querySelectorAll(".mobile-pitch-player-card").forEach(function (card) {
      var flag = card.querySelector("img, .flag, .player-flag");
      if (flag) {
        flag.classList.add("mobile-centered-flag");
      }
    });
  }

  function apply() {
    addStyle();
    compactStrategyButtons();
    markPitchFlags();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }

  window.addEventListener("resize", apply);

  var observer = new MutationObserver(function () {
    apply();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
}());
</script>
'''

if marker in text:
    print("mobile-toolbar-and-flags-v5 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet mobil-toolbar og flagfix v5.")
    print(f"Backup: {backup}")
    print("- Strategier forkortes til Næste / 1+2 / Gruppe / Lang")
    print("- Strategiboks gøres til lav toolbar")
    print("- Valgt plads gøres mindre")
    print("- Pitch-flag markeres og centreres via JS")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-toolbar-and-flags-v5",
    "mobile-toolbar-and-flags-v5-style",
    "mobile-centered-flag",
    'return "Næste"',
    'return "1+2"',
]:
    print(needle + " => " + str(text2.count(needle)))
