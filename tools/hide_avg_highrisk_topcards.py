from pathlib import Path
from datetime import datetime

path = Path("index.html")
text = path.read_text(encoding="utf-8")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_hide_avg_highrisk_topcards_{timestamp}.html")
backup.write_text(text, encoding="utf-8")

marker = "hide-avg-highrisk-topcards-runtime"

patch = r'''
<script id="hide-avg-highrisk-topcards-runtime">
(function () {
  function hideAvgHighRiskTopCards() {
    const all = Array.from(document.querySelectorAll("div, section, header"));
    const card = all.find(el => {
      const t = (el.innerText || "").trim();
      return t.includes("Brugt") && t.includes("Restbank") && t.includes("Aktuel bank") && t.includes("Spillere");
    });

    if (!card) return;

    const directChildren = Array.from(card.children);
    const spillerCard = directChildren.find(el => (el.innerText || "").includes("Spillere"));
    if (!spillerCard) return;

    const idx = directChildren.indexOf(spillerCard);
    const afterSpillere = directChildren.slice(idx + 1);

    // Skjul kun de to små topkort efter "Spillere" – de svarer til tidligere "Gns. start" og "High risk".
    afterSpillere.slice(0, 2).forEach(el => {
      el.style.display = "none";
      el.setAttribute("data-hidden-topcard", "avg-start-high-risk");
    });
  }

  document.addEventListener("DOMContentLoaded", hideAvgHighRiskTopCards);
  window.addEventListener("load", hideAvgHighRiskTopCards);

  const originalRenderBudget = window.renderBudget;
  if (typeof originalRenderBudget === "function") {
    window.renderBudget = function () {
      const result = originalRenderBudget.apply(this, arguments);
      hideAvgHighRiskTopCards();
      return result;
    };
  }

  setTimeout(hideAvgHighRiskTopCards, 100);
  setTimeout(hideAvgHighRiskTopCards, 500);
})();
</script>
'''

if marker not in text:
    text = text.replace("</body>", patch + "\n</body>")

path.write_text(text, encoding="utf-8")

print("Backup oprettet:", backup)
print("Patch indsat:", marker in text)
