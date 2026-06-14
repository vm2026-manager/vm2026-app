from pathlib import Path
from datetime import datetime
import re
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_flag_cleanup_final_{stamp}.html")
shutil.copy2(p, backup)

old_ids = [
    "mobile-flag-center-v5",
    "mobile-flag-reparent-v6",
    "mobile-flag-nudge-v7",
    "mobile-flag-precise-center-v8",
    "mobile-flag-true-center-v9",
]

removed = 0
for sid in old_ids:
    pattern = re.compile(
        r'\n?<script id="' + re.escape(sid) + r'">.*?</script>\s*',
        re.DOTALL
    )
    text, n = pattern.subn("\n", text)
    removed += n

marker = '<script id="mobile-flag-center-final">'

script = r'''
<script id="mobile-flag-center-final">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-flag-center-final-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-flag-center-final-style";
    style.textContent = `
@media (max-width: 700px) {
  .pitch .mobile-final-flag-card {
    position: relative !important;
    overflow: visible !important;
    text-align: center !important;
  }

  .pitch .mobile-final-centered-flag {
    position: absolute !important;
    left: 50% !important;
    right: auto !important;
    top: -9px !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
    margin: 0 !important;
    display: block !important;
    width: 24px !important;
    height: 17px !important;
    max-width: 24px !important;
    max-height: 17px !important;
    object-fit: cover !important;
    border-radius: 3px !important;
    z-index: 50 !important;
    pointer-events: none !important;
  }
}

@media (max-width: 430px) {
  .pitch .mobile-final-centered-flag {
    width: 23px !important;
    height: 16px !important;
    max-width: 23px !important;
    max-height: 16px !important;
    top: -8px !important;
  }
}
`;
    document.head.appendChild(style);
  }

  function looksLikeFlag(el) {
    if (!el) return false;
    var cls = String(el.className || "").toLowerCase();
    var src = String(el.getAttribute && el.getAttribute("src") || "").toLowerCase();
    var alt = String(el.getAttribute && el.getAttribute("alt") || "").toLowerCase();

    return (
      cls.indexOf("flag") >= 0 ||
      src.indexOf("flag") >= 0 ||
      src.indexOf("flags") >= 0 ||
      src.indexOf("country") >= 0 ||
      alt.indexOf("flag") >= 0
    );
  }

  function findCard(slot) {
    return slot.querySelector(".player-card, .slot-card, .squad-card, .mobile-pitch-player-card, .mobile-centered-pitch-card") ||
      Array.from(slot.children).find(function (child) {
        var txt = (child.textContent || "").trim();
        return txt && txt.indexOf("Fjern") < 0 && txt.indexOf("Skift") < 0;
      }) ||
      null;
  }

  function centerFlags() {
    if (!isMobile()) return;

    var slots = document.querySelectorAll(
      ".pitch .slot, .pitch .player-slot, .pitch .squad-slot, .pitch .field-slot, .pitch .mobile-pitch-player-slot"
    );

    slots.forEach(function (slot) {
      var card = findCard(slot);
      if (!card) return;

      var flags = Array.from(slot.querySelectorAll("img, .flag, .player-flag, [class*='flag']"))
        .filter(looksLikeFlag);

      if (!flags.length) return;

      var flag = flags[0];

      if (flag.parentElement !== card) {
        card.insertBefore(flag, card.firstChild);
      }

      card.classList.add("mobile-final-flag-card");
      flag.classList.add("mobile-final-centered-flag");

      flag.style.setProperty("position", "absolute", "important");
      flag.style.setProperty("left", "50%", "important");
      flag.style.setProperty("right", "auto", "important");
      flag.style.setProperty("top", window.innerWidth <= 430 ? "-8px" : "-9px", "important");
      flag.style.setProperty("transform", "translateX(-50%)", "important");
      flag.style.setProperty("margin", "0", "important");
      flag.style.setProperty("width", window.innerWidth <= 430 ? "23px" : "24px", "important");
      flag.style.setProperty("height", window.innerWidth <= 430 ? "16px" : "17px", "important");
      flag.style.setProperty("max-width", window.innerWidth <= 430 ? "23px" : "24px", "important");
      flag.style.setProperty("max-height", window.innerWidth <= 430 ? "16px" : "17px", "important");
    });
  }

  function install() {
    addStyle();

    function runSeveralTimes() {
      centerFlags();
      setTimeout(centerFlags, 150);
      setTimeout(centerFlags, 500);
      setTimeout(centerFlags, 1000);
    }

    runSeveralTimes();

    if (!window.__mobileFlagCenterFinalInstalled) {
      window.__mobileFlagCenterFinalInstalled = true;

      window.addEventListener("resize", runSeveralTimes);
      window.addEventListener("orientationchange", function () {
        setTimeout(runSeveralTimes, 250);
      });

      setInterval(centerFlags, 1000);
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

print("OK: Ryddet gamle flag-fixes og tilføjet én samlet flag-centrering.")
print(f"Gamle scriptblokke fjernet: {removed}")
print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-flag-center-final",
    "mobile-final-centered-flag",
    "translateX(-50%)",
]:
    print(needle + " => " + str(text2.count(needle)))

for sid in old_ids:
    print(sid + " tilbage => " + str(text2.count(sid)))
