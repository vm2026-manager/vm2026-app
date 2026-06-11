from pathlib import Path
import re

p = Path("index.html")
text = p.read_text(encoding="utf-8")
lines = text.splitlines()

patterns = [
    "logout",
    "signOut",
    "supabase.auth.signOut",
    "auth.signOut",
    "handleLogout",
    "logoutButton",
    "login",
    "currentUser",
    "squad =",
    "renderSquad",
    "renderBudget",
    "favorite_player_ids",
    "loadSavedSquad",
    "saveSquad",
]

seen = set()

for pat in patterns:
    print("\n" + "="*90)
    print("PATTERN:", pat)
    print("="*90)

    rx = re.compile(re.escape(pat), re.IGNORECASE)
    hits = [i for i, line in enumerate(lines) if rx.search(line)]

    if not hits:
        print("Ingen hits")
        continue

    for i in hits[:12]:
        start = max(0, i - 10)
        end = min(len(lines), i + 16)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- linje {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:5}: {lines[j]}")
