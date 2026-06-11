from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_test_now_param_{stamp}.html")
shutil.copy2(p, backup)

old = '''    function getNow() {
      return new Date();
    }'''

new = '''    function getNow() {
      try {
        const params = new URLSearchParams(window.location.search);
        const raw = params.get("test_now") || params.get("vm_now") || "";
        if (raw) {
          const simulated = new Date(String(raw).replace(" ", "T"));
          if (!Number.isNaN(simulated.getTime())) return simulated;
        }
      } catch (error) {
        // Ignore invalid test_now/vm_now values and use real time.
      }

      return new Date();
    }'''

if old not in text:
    raise SystemExit("Kunne ikke finde getNow()-blokken.")

text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

print("OK: Tilføjet test_now/vm_now parameter til simuleret tidspunkt.")
print(f"Backup: {backup}")
print("")
print("Eksempel:")
print("http://127.0.0.1:8000/index.html?test_now=2026-06-14T23:00:00")
