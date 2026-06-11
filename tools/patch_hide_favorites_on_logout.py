from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_hide_favorites_on_logout_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old_update = '''function updateFavoriteFilterButton() {
  const btn = document.getElementById("favoriteFilterBtn");
  if (!btn) return;

  btn.textContent = `\\u2665 Favoritter (${favoritePlayerIds.size})`;
  btn.classList.toggle("active", !!favoriteOnly);
}'''

new_update = '''function updateFavoriteFilterButton() {
  const btn = document.getElementById("favoriteFilterBtn");
  if (!btn) return;

  const isLoggedIn = !!currentUser;
  if (!isLoggedIn) {
    favoriteOnly = false;
  }

  btn.textContent = `\\u2665 Favoritter (${favoritePlayerIds.size})`;
  btn.classList.toggle("active", !!favoriteOnly);
  btn.style.display = isLoggedIn ? "" : "none";
  btn.disabled = !isLoggedIn;
  btn.setAttribute("aria-hidden", isLoggedIn ? "false" : "true");
  btn.tabIndex = isLoggedIn ? 0 : -1;
}'''

if old_update in text:
    text = text.replace(old_update, new_update, 1)
    changes.append("updateFavoriteFilterButton skjuler favoritknappen ved logout")
elif 'btn.style.display = isLoggedIn ? "" : "none";' in text:
    print("updateFavoriteFilterButton ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde updateFavoriteFilterButton-blokken.")

old_ensure_start = '''function ensureFavoriteFilterButton() {
  if (document.getElementById("favoriteFilterBtn")) {
    updateFavoriteFilterButton();
    return;
  }'''

new_ensure_start = '''function ensureFavoriteFilterButton() {
  if (!currentUser) {
    const existingBtn = document.getElementById("favoriteFilterBtn");
    if (existingBtn) updateFavoriteFilterButton();
    return;
  }

  if (document.getElementById("favoriteFilterBtn")) {
    updateFavoriteFilterButton();
    return;
  }'''

if old_ensure_start in text:
    text = text.replace(old_ensure_start, new_ensure_start, 1)
    changes.append("ensureFavoriteFilterButton opretter ikke favoritknap når logget ud")
elif 'if (!currentUser) {' in text and 'existingBtn' in text:
    print("ensureFavoriteFilterButton ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde ensureFavoriteFilterButton-starten.")

old_click = '''  btn.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();

    favoriteOnly = !favoriteOnly;
    updateFavoriteFilterButton();

    if (typeof renderTradeList === "function") renderTradeList();
  });'''

new_click = '''  btn.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();

    if (!currentUser) {
      favoriteOnly = false;
      updateFavoriteFilterButton();
      if (typeof renderTradeList === "function") renderTradeList();
      return;
    }

    favoriteOnly = !favoriteOnly;
    updateFavoriteFilterButton();

    if (typeof renderTradeList === "function") renderTradeList();
  });'''

if old_click in text:
    text = text.replace(old_click, new_click, 1)
    changes.append("favoritknap-click blokeres når logget ud")
elif 'if (!currentUser) {' in text and 'favoriteOnly = !favoriteOnly;' in text:
    print("favorite click ser muligvis allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde favoritknap-click-blokken.")

old_logout = '''      frontendCaptainOverride = null;

      if (activeSlotText) {'''

new_logout = '''      frontendCaptainOverride = null;
      favoriteOnly = false;

      if (typeof updateFavoriteFilterButton === "function") {
        updateFavoriteFilterButton();
      }

      if (activeSlotText) {'''

if old_logout in text:
    text = text.replace(old_logout, new_logout, 1)
    changes.append("logout slår favoritvisning fra")
elif 'favoriteOnly = false;' in text and 'clearLocalSquadViewAfterLogout' in text:
    print("logout-helper ser muligvis allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde logout-helper-blokken.")

p.write_text(text, encoding="utf-8")

print("OK: Favoritlisten skjules nu ved logout uden at slette favoritter.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    'btn.style.display = isLoggedIn ? "" : "none";',
    "existingBtn",
    "favoriteOnly = false;",
    "clearLocalSquadViewAfterLogout",
]:
    print(needle, "=>", text.count(needle))
