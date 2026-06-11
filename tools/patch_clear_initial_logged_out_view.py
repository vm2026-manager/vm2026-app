from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_clear_initial_logged_out_view_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old_helper_part = '''      frontendCaptainOverride = null;
      favoriteOnly = false;

      if (typeof updateFavoriteFilterButton === "function") {'''

new_helper_part = '''      frontendCaptainOverride = null;
      favoriteOnly = false;
      manualBankMillions = null;

      if (manualBankInput) {
        manualBankInput.value = "";
      }

      if (typeof updateFavoriteFilterButton === "function") {'''

if old_helper_part in text:
    text = text.replace(old_helper_part, new_helper_part, 1)
    changes.append("Udlogget visning nulstiller nu også manuel bank")
elif "manualBankMillions = null;" in text and "clearLocalSquadViewAfterLogout" in text:
    print("Logout-helper ser allerede ud til at nulstille manuel bank.")
else:
    raise SystemExit("Kunne ikke finde clearLocalSquadViewAfterLogout-blokken.")

old_init_auth = '''      if (currentUser) {
        await loadAutosaveSquadForUser();
      }

      sanitizeSearchField();'''

new_init_auth = '''      if (currentUser) {
        await loadAutosaveSquadForUser();
      } else {
        clearLocalSquadViewAfterLogout();
      }

      sanitizeSearchField();'''

if old_init_auth in text:
    text = text.replace(old_init_auth, new_init_auth, 1)
    changes.append("Initialt page-load som udlogget bruger rydder nu lokal holdvisning")
elif "clearLocalSquadViewAfterLogout();" in text and "await loadAutosaveSquadForUser();" in text:
    print("initAuth ser muligvis allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde initAuth currentUser-blokken.")

p.write_text(text, encoding="utf-8")

print("OK: Udlogget initialvisning ryddes nu.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "manualBankMillions = null;",
    "clearLocalSquadViewAfterLogout();",
    "if (currentUser) {",
]:
    print(needle, "=>", text.count(needle))
