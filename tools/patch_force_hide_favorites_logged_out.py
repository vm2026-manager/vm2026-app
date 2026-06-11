from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_force_hide_favorites_logged_out_{stamp}.html")
shutil.copy2(p, backup)

changes = []

old_update = '''function updateFavoriteFilterButton() {
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

new_update = '''function updateFavoriteFilterButton() {
  const btn = document.getElementById("favoriteFilterBtn");
  if (!btn) return;

  const isLoggedIn = !!(currentUser && currentUser.id);
  if (!isLoggedIn) {
    favoriteOnly = false;
  }

  btn.textContent = `\\u2665 Favoritter (${favoritePlayerIds.size})`;
  btn.classList.toggle("active", !!favoriteOnly);
  btn.hidden = !isLoggedIn;
  btn.disabled = !isLoggedIn;
  btn.setAttribute("aria-hidden", isLoggedIn ? "false" : "true");
  btn.tabIndex = isLoggedIn ? 0 : -1;

  if (isLoggedIn) {
    btn.style.removeProperty("display");
  } else {
    btn.style.setProperty("display", "none", "important");
  }
}'''

if old_update in text:
    text = text.replace(old_update, new_update, 1)
    changes.append("updateFavoriteFilterButton bruger nu display:none !important ved logout")
elif 'btn.style.setProperty("display", "none", "important");' in text:
    print("updateFavoriteFilterButton ser allerede force-patchet ud.")
else:
    raise SystemExit("Kunne ikke finde den nuværende updateFavoriteFilterButton-blok.")

old_move = '''      var favBtn = document.getElementById("favoriteFilterBtn");
      var altBtn = findAlternativesButton();

      if (!favBtn || !altBtn || !altBtn.parentElement) return;'''

new_move = '''      var favBtn = document.getElementById("favoriteFilterBtn");
      var altBtn = findAlternativesButton();

      if (typeof updateFavoriteFilterButton === "function") {
        updateFavoriteFilterButton();
      }

      if (!currentUser || !currentUser.id) return;
      if (!favBtn || !altBtn || !altBtn.parentElement) return;'''

if old_move in text:
    text = text.replace(old_move, new_move, 1)
    changes.append("moveFavoritesButtonNextToAlternatives stopper nu hvis logget ud")
elif "if (!currentUser || !currentUser.id) return;" in text:
    print("moveFavoritesButtonNextToAlternatives ser allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde moveFavoritesButtonNextToAlternatives-blokken.")

old_render = '''      const visiblePlayers = favoriteOnly
        ? players
            .filter(player => isFavoritePlayer(player.player_id))
            .sort((a, b) => favoritePositionOrder(a) - favoritePositionOrder(b))
        : players;'''

new_render = '''      if (!currentUser || !currentUser.id) {
        favoriteOnly = false;
        if (typeof updateFavoriteFilterButton === "function") updateFavoriteFilterButton();
      }

      const visiblePlayers = favoriteOnly
        ? players
            .filter(player => isFavoritePlayer(player.player_id))
            .sort((a, b) => favoritePositionOrder(a) - favoritePositionOrder(b))
        : players;'''

if old_render in text:
    text = text.replace(old_render, new_render, 1)
    changes.append("renderTradeList slår favoritvisning fra hvis logget ud")
elif "renderTradeList slår" in text or "if (!currentUser || !currentUser.id) {" in text:
    print("renderTradeList ser muligvis allerede patchet ud.")
else:
    raise SystemExit("Kunne ikke finde visiblePlayers/favoriteOnly-blokken.")

p.write_text(text, encoding="utf-8")

print("OK: Favoritknappen force-skjules nu ved logout.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    'btn.style.setProperty("display", "none", "important");',
    "if (!currentUser || !currentUser.id) return;",
    "favoriteOnly = false;",
]:
    print(needle, "=>", text.count(needle))
