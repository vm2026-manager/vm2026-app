from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_flag_center_v5_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-flag-center-v5">'

script = r'''
<script id="mobile-flag-center-v5">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-flag-center-v5-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-flag-center-v5-style";
    style.textContent = `
@media (max-width: 700px) {
  .player-card,
  .slot-card,
  .squad-card,
  .mobile-pitch-player-card {
    position: relative !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding-top: 14px !important;
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
    position: absolute !important;
    left: 50% !important;
    right: auto !important;
    top: -12px !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
    margin: 0 !important;
    display: block !important;
    z-index: 4 !important;
  }

  .player-name,
  .name,
  .player-card strong,
  .player-card b,
  .slot-card strong,
  .slot-card b,
  .squad-card strong,
  .squad-card b,
  .mobile-pitch-player-card strong,
  .mobile-pitch-player-card b {
    width: 100% !important;
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
}

@media (max-width: 430px) {
  .player-card img,
  .slot-card img,
  .squad-card img,
  .mobile-pitch-player-card img,
  .player-card .flag,
  .slot-card .flag,
  .squad-card .flag,
  .mobile-pitch-player-card .flag,
  .player-flag {
    top: -11px !important;
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
    print("mobile-flag-center-v5 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Flag er nu centreret over spillernavnet på mobil.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-flag-center-v5",
    "mobile-flag-center-v5-style",
    "left: 50% !important",
    "transform: translateX(-50%) !important",
    "text-align: center !important",
]:
    print(needle + " => " + str(text2.count(needle)))
