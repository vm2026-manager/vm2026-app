from pathlib import Path
import re
from datetime import datetime

path = Path("index.html")
text = path.read_text(encoding="utf-8")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_remove_start_highrisk_cards_{timestamp}.html")
backup.write_text(text, encoding="utf-8")

original = text

# Fjerner card/blokke hvor label-teksten er Gns. start eller High risk.
# Matcher typiske topkort-divs med indholdet.
patterns = [
    r'\s*<div[^>]*class="[^"]*(?:stat|summary|top|metric|card)[^"]*"[^>]*>\s*<[^>]+>\s*Gns\. start\s*</[^>]+>.*?</div>',
    r'\s*<div[^>]*class="[^"]*(?:stat|summary|top|metric|card)[^"]*"[^>]*>\s*<[^>]+>\s*High risk\s*</[^>]+>.*?</div>',
    r'\s*<div[^>]*>\s*<[^>]+>\s*Gns\. start\s*</[^>]+>.*?</div>',
    r'\s*<div[^>]*>\s*<[^>]+>\s*High risk\s*</[^>]+>.*?</div>',
]

for pattern in patterns:
    text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

# Hvis JS stadig forsøger at opdatere elementer, gør det uskadeligt ved at skjule via CSS som fallback.
fallback_css = """
<style id="hide-start-highrisk-cards">
  [data-card="avg-start"],
  [data-card="high-risk"],
  #avgStartCard,
  #highRiskCard,
  .avg-start-card,
  .high-risk-card {
    display: none !important;
  }
</style>
"""

if "hide-start-highrisk-cards" not in text:
    text = text.replace("</head>", fallback_css + "\n</head>")

path.write_text(text, encoding="utf-8")

print("Backup oprettet:", backup)
print("index.html opdateret:", text != original)
print("Gns. start tilbage i fil:", "Gns. start" in text)
print("High risk tilbage i fil:", "High risk" in text)
