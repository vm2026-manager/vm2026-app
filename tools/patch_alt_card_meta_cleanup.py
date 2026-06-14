from pathlib import Path
from datetime import datetime

p = Path("index.html")
txt = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_alt_card_meta_cleanup_{stamp}.html")
backup.write_text(txt, encoding="utf-8")

# 1) Fjern spillerens land fra meta-linjen og brug kun position.
old = '''<div class="alternative-player-meta">${escapeHtml(positionFullLabel(player.position))} &middot; ${escapeHtml(teamDisplayName(player.team_id, player.team_name))}</div>
                <div class="alternative-player-opponent">${escapeHtml(nextOpponentText(player.team_id))}</div>'''

new = '''<div class="alternative-player-meta">${escapeHtml(positionFullLabel(player.position))}</div>
                <div class="alternative-player-opponent">${escapeHtml(nextOpponentText(player.team_id))}</div>'''

if old not in txt:
    print("ADVARSEL: Fandt ikke præcis meta-linje. CSS-override tilføjes stadig.")
else:
    txt = txt.replace(old, new, 1)

# 2) CSS-override: næste modstander står ved siden af positionen, start% står på samme linje,
# og start får ikke længere sin egen række.
start = "/* ALT_CARD_META_CLEANUP_START */"
end = "/* ALT_CARD_META_CLEANUP_END */"

override = r'''
/* ALT_CARD_META_CLEANUP_START */
/* Alternativliste: fjern land fra visning, vis næste modstander i metaområdet og flyt start% op. */

.right-panel.alternative-mode .trade-row.alternative .trade-player-main {
  grid-template-columns: minmax(0, 1fr) 58px 104px !important;
  grid-template-areas:
    "name name price"
    "meta start price"
    "opp opp price" !important;
  column-gap: 8px !important;
  row-gap: 1px !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-name {
  grid-area: name !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-meta {
  grid-area: meta !important;
  font-size: 11px !important;
  line-height: 1.15 !important;
  color: #d7e1ef !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-opponent {
  grid-area: opp !important;
  font-size: 11px !important;
  line-height: 1.15 !important;
  color: #ffe45c !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col {
  grid-area: start !important;
  justify-self: end !important;
  align-self: start !important;
  text-align: right !important;
  margin-top: 0 !important;
  font-size: 10px !important;
  line-height: 1.05 !important;
  white-space: nowrap !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col strong,
.right-panel.alternative-mode .trade-row.alternative .trade-start-col .trade-start-percent {
  display: inline !important;
  font-size: 13px !important;
  line-height: 1 !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col div,
.right-panel.alternative-mode .trade-row.alternative .trade-start-col span {
  display: inline !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col::after {
  content: "" !important;
}

.right-panel.alternative-mode .trade-row.alternative {
  min-height: 76px !important;
}
/* ALT_CARD_META_CLEANUP_END */
'''

if start in txt and end in txt:
    before = txt[:txt.index(start)]
    after = txt[txt.index(end) + len(end):]
    txt = before + override + after
else:
    txt = txt.replace("</style>", override + "\n</style>", 1)

p.write_text(txt, encoding="utf-8")

print("Backup:", backup)
print("Patched:", p)
print("Meta cleanup applied")
