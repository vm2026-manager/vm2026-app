from pathlib import Path
from datetime import datetime
import re

path = Path("index.html")
text = path.read_text(encoding="utf-8")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_remove_avg_highrisk_source_{timestamp}.html")
backup.write_text(text, encoding="utf-8")

original = text

text = re.sub(
    r'\s*<div class="bank-box">\s*<div id="avgStartValue">.*?</div>\s*</div>',
    "",
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'\s*<div class="bank-box">\s*<div id="highRiskValue">.*?</div>\s*</div>',
    "",
    text,
    flags=re.DOTALL
)

text = re.sub(r'\n\s*const avgStartValue = document\.getElementById\("avgStartValue"\);', "", text)
text = re.sub(r'\n\s*const highRiskValue = document\.getElementById\("highRiskValue"\);', "", text)

text = re.sub(
    r'\n\s*const avgStart = selectedPlayers\.length\s*\?\s*selectedPlayers\.reduce\(\(sum, player\) => sum \+ getPlayerStart\(player\), 0\) / selectedPlayers\.length\s*:\s*0;',
    "",
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'\n\s*const highRiskCount = selectedPlayers\.filter\(player => String\(player\.availability_risk \|\| ""\)\.toLowerCase\(\) === "high_risk"\)\.length;',
    "",
    text
)

text = re.sub(r'\n\s*avgStartValue\.textContent = selectedPlayers\.length \? `\$\{Math\.round\(avgStart\)\}%` : ".*?";', "", text)
text = re.sub(r'\n\s*highRiskValue\.textContent = String\(highRiskCount\);', "", text)

text = re.sub(
    r'\n*<style id="hide-start-highrisk-cards">.*?</style>\s*',
    "\n",
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'\n*<script id="hide-avg-highrisk-topcards-runtime">.*?</script>\s*',
    "\n",
    text,
    flags=re.DOTALL
)

path.write_text(text, encoding="utf-8")

print("Backup oprettet:", backup)
print("index.html ændret:", text != original)
print("avgStartValue tilbage:", "avgStartValue" in text)
print("highRiskValue tilbage:", "highRiskValue" in text)
print("hide-start-highrisk-cards tilbage:", "hide-start-highrisk-cards" in text)
print("hide-avg-highrisk-topcards-runtime tilbage:", "hide-avg-highrisk-topcards-runtime" in text)
print("Gns. start tilbage:", "Gns. start" in text)
print("High risk tilbage:", "High risk" in text)
