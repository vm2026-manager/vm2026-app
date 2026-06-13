from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_flag_reparent_v6_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-flag-reparent-v6">'

script = r'''
<script id="mobile-flag-reparent-v6">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-flag-reparent-v6-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-flag-reparent-v6-style";
    style.textContent = `
@media (max-width: 700px) {
  .mobile-centered-pitch-card {
    position: relative !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding-top: 14px !important;
    overflow: visible !important;
  }

  .mobile-centered-pitch-flag {
    position: absolute !important;
    left: 50% !important;
    right: auto !important;
    top: -12px !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
    margin: 0 !important;
    display: block !important;
    width: 28px !important;
    height: 20px !important;
    max-width: 28px !important;
    max-height: 20px !important;
    object-fit: cover !important;
    border-radius: 4px !important;
    z-index: 10 !important;
    pointer-events: none !important;
  }

  .mobile-centered-pitch-card strong,
  .mobile-centered-pitch-card b,
  .mobile-centered-pitch-card .player-name,
  .mobile-centered-pitch-card .name {
    width: 100% !important;
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
}

@media (max-width: 430px) {
  .mobile-centered-pitch-flag {
    top: -11px !important;
    width: 26px !important;
    height: 19px !important;
    max-width: 26px !important;
    max-height: 19px !important;
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
    return slot.querySelector(".player-card, .slot-card, .squad-card, .mobile-pitch-player-card") ||
      Array.from(slot.children).find(function (child) {
        var txt = (child.textContent || "").trim();
        return txt && txt.indexOf("Fjern") < 0 && txt.indexOf("Skift") < 0;
      }) ||
      null;
  }

  function centerPitchFlags() {
    if (!isMobile()) return;

    var slots = document.querySelectorAll(
      ".pitch .slot, .pitch .player-slot, .pitch .squad-slot, .pitch .field-slot, .pitch .mobile-pitch-player-slot"
    );

    slots.forEach(function (slot) {
      var card = findCard(slot);
      if (!card) return;

      card.classList.add("mobile-centered-pitch-card");

      var flags = Array.from(slot.querySelectorAll("img, .flag, .player-flag, [class*='flag']"))
        .filter(looksLikeFlag);

      if (!flags.length) return;

      var flag = flags[0];

      if (flag.parentElement !== card) {
        card.insertBefore(flag, card.firstChild);
      }

      flag.classList.add("mobile-centered-pitch-flag");
      flag.style.setProperty("left", "50%", "important");
      flag.style.setProperty("right", "auto", "important");
      flag.style.setProperty("top", "-12px", "important");
      flag.style.setProperty("transform", "translateX(-50%)", "important");
      flag.style.setProperty("margin", "0", "important");
    });
  }

  function install() {
    addStyle();
    centerPitchFlags();

    if (!window.__mobileFlagReparentV6Installed) {
      window.__mobileFlagReparentV6Installed = true;

      window.addEventListener("resize", centerPitchFlags);
      window.addEventListener("orientationchange", function () {
        setTimeout(centerPitchFlags, 250);
      });

      setInterval(centerPitchFlags, 1000);
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

if marker in text:
    print("mobile-flag-reparent-v6 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Flag flyttes nu ind i spillerkort og centreres på mobil.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-flag-reparent-v6",
    "mobile-centered-pitch-flag",
    "card.insertBefore(flag, card.firstChild)",
    "setInterval(centerPitchFlags, 1000)",
]:
    print(needle + " => " + str(text2.count(needle)))
