from pathlib import Path

text = Path("index.html").read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

ranges = [
    (4540, 4665, "fixture/core lookup"),
    (7280, 7425, "fixture tooltip render"),
    (7425, 7555, "fixture tooltip handlers"),
    (6280, 6335, "trade row player list html"),
    (5060, 5115, "alternative row html"),
]

out_lines = []

for start, end, title in ranges:
    out_lines.append("\n" + "="*100)
    out_lines.append(f"{title} | lines {start}-{end}")
    out_lines.append("="*100)
    for n in range(start, min(end, len(lines)) + 1):
        out_lines.append(f"{n:5}: {lines[n-1]}")

out = Path("data/audit_tooltip_relevant_index_lines.txt")
out.write_text("\n".join(out_lines), encoding="utf-8")

print("Skrevet:", out)
print("\n".join(out_lines[:220]))
