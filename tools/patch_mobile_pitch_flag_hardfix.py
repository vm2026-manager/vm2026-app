from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_pitch_flag_hardfix_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-pitch-flag-hardfix">'

script = r'''
<script id="mobile-pitch-flag-hardfix">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-pitch-flag-hardfix-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-pitch-flag-hardfix-style";
    style.textContent = `
@media (max-width: 700px) {
  .pitch .player-card,
  .pitch .slot-card,
  .pitch .squad-card,
  .pitch .mobile-pitch-player-card,
  .pitch .mobile-centered-pitch-card,
  .pitch .mobile-final-flag-card {
    position: relative !important;
    overflow: visible !important;
    text-align: center !important;
  }

  .pitch .player-card img,
  .pitch .slot-card img,
  .pitch .squad-card img,
  .pitch .mobile-pitch-player-card img,
  .pitch .mobile-centered-pitch-card img,
  .pitch .mobile-final-flag-card img {
    position: absolute !important;
    left: 50% !important;
    right: auto !important;
    top: -8px !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
    margin: 0 !important;
    display: block !important;
    width: 24px !important;
    height: 17px !important;
    max-width: 24px !important;
    max-height: 17px !important;
    min-width: 24px !important;
    min-height: 17px !important;
    object-fit: cover !important;
    border-radius: 3px !important;
    z-index: 80 !important;
    pointer-events: none !important;
  }
}

@media (max-width: 430px) {
  .pitch .player-card img,
  .pitch .slot-card img,
  .pitch .squad-card img,
  .pitch .mobile-pitch-player-card img,
  .pitch .mobile-centered-pitch-card img,
  .pitch .mobile-final-flag-card img {
    width: 23px !important;
    height: 16px !important;
    max-width: 23px !important;
    max-height: 16px !important;
    min-width: 23px !important;
    min-height: 16px !important;
    top: -7px !important;
  }
}
`;
    document.head.appendChild(style);
  }

  function normalizePitchFlags() {
    if (!isMobile()) return;

    var cards = document.querySelectorAll(
      ".pitch .player-card, .pitch .slot-card, .pitch .squad-card, .pitch .mobile-pitch-player-card, .pitch .mobile-centered-pitch-card, .pitch .mobile-final-flag-card"
    );

    cards.forEach(function (card) {
      card.style.setProperty("position", "relative", "important");
      card.style.setProperty("overflow", "visible", "important");
      card.style.setProperty("text-align", "center", "important");

      var imgs = card.querySelectorAll("img");
      imgs.forEach(function (img) {
        img.style.setProperty("position", "absolute", "important");
        img.style.setProperty("left", "50%", "important");
        img.style.setProperty("right", "auto", "important");
        img.style.setProperty("top", window.innerWidth <= 430 ? "-7px" : "-8px", "important");
        img.style.setProperty("bottom", "auto", "important");
        img.style.setProperty("transform", "translateX(-50%)", "important");
        img.style.setProperty("margin", "0", "important");
        img.style.setProperty("width", window.innerWidth <= 430 ? "23px" : "24px", "important");
        img.style.setProperty("height", window.innerWidth <= 430 ? "16px" : "17px", "important");
        img.style.setProperty("max-width", window.innerWidth <= 430 ? "23px" : "24px", "important");
        img.style.setProperty("max-height", window.innerWidth <= 430 ? "16px" : "17px", "important");
        img.style.setProperty("min-width", window.innerWidth <= 430 ? "23px" : "24px", "important");
        img.style.setProperty("min-height", window.innerWidth <= 430 ? "16px" : "17px", "important");
        img.style.setProperty("object-fit", "cover", "important");
        img.style.setProperty("z-index", "80", "important");
      });
    });
  }

  function install() {
    addStyle();

    function run() {
      normalizePitchFlags();
      setTimeout(normalizePitchFlags, 100);
      setTimeout(normalizePitchFlags, 400);
      setTimeout(normalizePitchFlags, 900);
    }

    run();

    if (!window.__mobilePitchFlagHardfixInstalled) {
      window.__mobilePitchFlagHardfixInstalled = true;
      window.addEventListener("resize", run);
      window.addEventListener("orientationchange", function () {
        setTimeout(run, 250);
      });
      setInterval(normalizePitchFlags, 1000);
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
    print("mobile-pitch-flag-hardfix findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Tilføjet hård mobil-fix for baneflag.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-pitch-flag-hardfix",
    "normalizePitchFlags",
    "min-width: 23px !important",
    "translateX(-50%)",
]:
    print(needle + " => " + str(text2.count(needle)))
