from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_render_logged_out_privacy_guard_{stamp}.html")
shutil.copy2(p, backup)

changes = []

insert_before = '''    function render() {
      cleanSquadForCurrentFormation();'''

helper = '''    function enforceLoggedOutPrivacyState() {
      if (currentUser && currentUser.id) return;

      squad = {};
      activeSlotKey = null;
      previousReplacementBySlot.clear();
      modelChoiceBySlot.clear();
      frontendCaptainOverride = null;
      favoriteOnly = false;

      if (activeSlotText) {
        activeSlotText.textContent = "Ingen valgt";
      }

      if (typeof updateFavoriteFilterButton === "function") {
        updateFavoriteFilterButton();
      }
    }

'''

new_render_start = '''    function render() {
      enforceLoggedOutPrivacyState();
      cleanSquadForCurrentFormation();'''

if "function enforceLoggedOutPrivacyState()" not in text:
    if insert_before not in text:
        raise SystemExit("Kunne ikke finde render()-starten.")
    text = text.replace(insert_before, helper + new_render_start, 1)
    changes.append("Tilføjet enforceLoggedOutPrivacyState() og kald i render()")
elif "enforceLoggedOutPrivacyState();" not in text:
    text = text.replace(
        '''    function render() {
      cleanSquadForCurrentFormation();''',
        '''    function render() {
      enforceLoggedOutPrivacyState();
      cleanSquadForCurrentFormation();''',
        1
    )
    changes.append("Tilføjet enforceLoggedOutPrivacyState() i render()")
else:
    print("render() ser allerede ud til at have logged-out privacy guard.")

p.write_text(text, encoding="utf-8")

print("OK: render() har nu hård privacy guard for udlogget bruger.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "function enforceLoggedOutPrivacyState()",
    "enforceLoggedOutPrivacyState();",
    "squad = {};",
    "favoriteOnly = false;",
]:
    print(needle, "=>", text.count(needle))
