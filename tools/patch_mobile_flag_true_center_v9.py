from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_flag_true_center_v9_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-flag-true-center-v9">'

script = r'''
<script id="mobile-flag-true-center-v9">
(function () {
  function isMobile() {
    return window.matchMedia && window.matchMedia("(max-width: 700px)").matches;
  }

  function addStyle() {
    if (document.getElementById("mobile-flag-true-center-v9-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-flag-true-center-v9-style";
    style.textContent = `
@media (max-width: 700px) {
  .mobile-centered-pitch-card {
    position: relative !important;
    overflow: visible !important;
  }

  .mobile-true-centered-flag {
    position: absolute !important;
    right: auto !important;
    top: -12px !important;
    bottom: auto !important;
    transform: none !important;
    margin: 0 !important;
    display: block !important;
    width: 28px !important;
    height: 20px !important;
    max-width: 28px !important;
    max-height: 20px !important;
    object-fit: cover !important;
    border-radius: 4px !important;
    z-index: 30 !important;
    pointer-events: none !important;
  }
}

@media (max-width: 430px) {
  .mobile-true-centered-flag {
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
    return slot.querySelector(".player-card, .slot-card, .squad-card, .mobile-pitch-player-card, .mobile-centered-pitch-card") ||
      Array.from(slot.children).find(function (child) {
        var txt = (child.textContent || "").trim();
        return txt && txt.indexOf("Fjern") < 0 && txt.indexOf("Skift") < 0;
      }) ||
      null;
  }

  function trueCenterFlags() {
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

      flag.classList.add("mobile-true-centered-flag");

      // Vigtigt: left skal være flagets venstre kant, ikke flagets midte.
      var cardWidth = card.getBoundingClientRect().width || card.offsetWidth || 54;
      var flagWidth = flag.getBoundingClientRect().width || flag.offsetWidth || (window.innerWidth <= 430 ? 26 : 28);
      var leftPx = Math.max(0, (cardWidth - flagWidth) / 2);

      flag.style.setProperty("left", leftPx + "px", "important");
      flag.style.setProperty("right", "auto", "important");
      flag.style.setProperty("transform", "none", "important");
      flag.style.setProperty("margin", "0", "important");
    });
  }

  function install() {
    addStyle();
    trueCenterFlags();

    if (!window.__mobileFlagTrueCenterV9Installed) {
      window.__mobileFlagTrueCenterV9Installed = true;

      window.addEventListener("resize", trueCenterFlags);
      window.addEventListener("orientationchange", function () {
        setTimeout(trueCenterFlags, 250);
      });

      setInterval(trueCenterFlags, 800);
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
    print("mobile-flag-true-center-v9 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Flag centreres nu med venstre kant = kortmidte minus halv flagbredde.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-flag-true-center-v9",
    "mobile-true-centered-flag",
    "(cardWidth - flagWidth) / 2",
    'flag.style.setProperty("transform", "none", "important")',
]:
    print(needle + " => " + str(text2.count(needle)))
