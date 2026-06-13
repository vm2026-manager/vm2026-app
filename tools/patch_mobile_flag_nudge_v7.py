from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_mobile_flag_nudge_v7_{stamp}.html")
shutil.copy2(p, backup)

marker = '<script id="mobile-flag-nudge-v7">'

script = r'''
<script id="mobile-flag-nudge-v7">
(function () {
  function addStyle() {
    if (document.getElementById("mobile-flag-nudge-v7-style")) return;

    var style = document.createElement("style");
    style.id = "mobile-flag-nudge-v7-style";
    style.textContent = `
@media (max-width: 700px) {
  .mobile-centered-pitch-flag,
  .player-card img.mobile-centered-pitch-flag,
  .slot-card img.mobile-centered-pitch-flag,
  .squad-card img.mobile-centered-pitch-flag,
  .mobile-pitch-player-card img.mobile-centered-pitch-flag {
    left: calc(50% - 3px) !important;
    transform: translateX(-50%) !important;
  }
}

@media (max-width: 430px) {
  .mobile-centered-pitch-flag,
  .player-card img.mobile-centered-pitch-flag,
  .slot-card img.mobile-centered-pitch-flag,
  .squad-card img.mobile-centered-pitch-flag,
  .mobile-pitch-player-card img.mobile-centered-pitch-flag {
    left: calc(50% - 3px) !important;
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
    print("mobile-flag-nudge-v7 findes allerede. Ingen ændring.")
else:
    if "</body>" not in text:
        raise SystemExit("Kunne ikke finde </body>.")
    text = text.replace("</body>", script + "\n</body>", 1)
    p.write_text(text, encoding="utf-8")
    print("OK: Flag rykket 3px mod venstre på mobil.")
    print(f"Backup: {backup}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-flag-nudge-v7",
    "left: calc(50% - 3px) !important",
]:
    print(needle + " => " + str(text2.count(needle)))
