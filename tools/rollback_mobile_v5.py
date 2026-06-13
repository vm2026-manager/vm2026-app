from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_rollback_mobile_v5_{stamp}.html")
shutil.copy2(p, backup)

pattern = re.compile(
    r'\n?<script id="mobile-toolbar-and-flags-v5">.*?</script>\n?',
    flags=re.DOTALL
)

new_text, n = pattern.subn("\n", text)

if n == 0:
    print("Fandt ikke mobile-toolbar-and-flags-v5. Ingen ændring.")
else:
    p.write_text(new_text, encoding="utf-8")
    print("OK: Rullet mobile-toolbar-and-flags-v5 tilbage.")
    print(f"Backup: {backup}")
    print(f"Fjernede blokke: {n}")

print("")
print("Sanity:")
text2 = p.read_text(encoding="utf-8")
for needle in [
    "mobile-toolbar-and-flags-v5",
    "mobile-centered-flag",
    "return \"Næste\"",
]:
    print(needle + " => " + str(text2.count(needle)))
