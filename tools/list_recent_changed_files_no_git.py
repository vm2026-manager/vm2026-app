from pathlib import Path
from datetime import datetime, timedelta
import os

ROOT = Path(".")
DAYS_BACK = 3

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

IMPORTANT_EXTS = {
    ".html",
    ".py",
    ".json",
    ".csv",
    ".md",
    ".txt",
}

cutoff = datetime.now() - timedelta(days=DAYS_BACK)
rows = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    parts = set(path.parts)
    if parts & IGNORE_DIRS:
        continue

    if path.suffix.lower() not in IMPORTANT_EXTS:
        continue

    try:
        stat = path.stat()
    except OSError:
        continue

    modified = datetime.fromtimestamp(stat.st_mtime)
    if modified < cutoff:
        continue

    rows.append({
        "path": str(path),
        "modified": modified,
        "size_kb": round(stat.st_size / 1024, 1),
    })

rows.sort(key=lambda r: r["modified"], reverse=True)

out = Path("data/recent_changed_files_no_git.txt")
out.parent.mkdir(exist_ok=True)

lines = []
lines.append(f"Filer ændret/oprettet seneste {DAYS_BACK} dage")
lines.append(f"Kørt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("")
lines.append(f"{'Tidspunkt':19}  {'KB':>8}  Fil")
lines.append("-" * 100)

for r in rows:
    lines.append(
        f"{r['modified'].strftime('%Y-%m-%d %H:%M:%S')}  {r['size_kb']:>8}  {r['path']}"
    )

out.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines[:80]))
print()
print("Skrevet:", out)
