from pathlib import Path

path = Path("tools/export_strategy_squads_direct.py")
text = path.read_text(encoding="utf-8")

# Indsæt skip-liste efter strategy_data = load_json(...)
needle = "strategy_data = load_json(STRATEGY_FILE)\n"
insert = """strategy_data = load_json(STRATEGY_FILE)

# Legacy-dublet. practical_start er den rigtige/app-brugte '1. + 2. runde'.
SKIP_STRATEGY_KEYS = {"round1_2"}
"""

if needle in text and "SKIP_STRATEGY_KEYS" not in text:
    text = text.replace(needle, insert, 1)

# Skip round1_2 i hovedloop
old = "for strategy_key, strategy_entry in strategy_data.items():\n"
new = """for strategy_key, strategy_entry in strategy_data.items():
    if strategy_key in SKIP_STRATEGY_KEYS:
        continue
"""

if old in text and "if strategy_key in SKIP_STRATEGY_KEYS" not in text:
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Patched export_strategy_squads_direct.py to skip round1_2")
