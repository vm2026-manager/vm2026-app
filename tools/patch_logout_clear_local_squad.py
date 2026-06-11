from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_logout_clear_local_squad_{stamp}.html")
shutil.copy2(p, backup)

changes = []

insert_before = "    async function initAuth() {"
helper = """    function clearLocalSquadViewAfterLogout() {
      squad = {};
      activeSlotKey = null;
      previousReplacementBySlot.clear();
      modelChoiceBySlot.clear();
      frontendCaptainOverride = null;

      if (activeSlotText) {
        activeSlotText.textContent = "Ingen valgt";
      }

      saveState();
      render();
    }

"""

if "function clearLocalSquadViewAfterLogout()" not in text:
    if insert_before not in text:
        raise SystemExit("Kunne ikke finde initAuth, hvor helper-funktionen skal indsættes.")
    text = text.replace(insert_before, helper + insert_before, 1)
    changes.append("Tilføjet clearLocalSquadViewAfterLogout()")
else:
    print("Helper findes allerede.")

old = """        if (currentUser && currentUser.id !== previousUserId) {
          await loadAutosaveSquadForUser();
          showStatus("Dit gemte hold er indlæst.", "info");
        }"""

new = """        if (!currentUser && previousUserId) {
          clearLocalSquadViewAfterLogout();
          showStatus("Du er logget ud.", "info");
          return;
        }

        if (currentUser && currentUser.id !== previousUserId) {
          await loadAutosaveSquadForUser();
          showStatus("Dit gemte hold er indlæst.", "info");
        }"""

if old in text:
    text = text.replace(old, new, 1)
    changes.append("Auth-state logout rydder lokal holdvisning")
elif "clearLocalSquadViewAfterLogout();" in text:
    print("Auth-state callback ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde auth-state login-blokken.")

p.write_text(text, encoding="utf-8")

print("OK: Logout rydder nu lokal holdvisning uden at slette Supabase-data.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "clearLocalSquadViewAfterLogout",
    "Du er logget ud.",
    "await loadAutosaveSquadForUser();",
]:
    print(needle, "=>", text.count(needle))
