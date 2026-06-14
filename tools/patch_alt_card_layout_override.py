from pathlib import Path
from datetime import datetime

p = Path("index.html")
txt = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"index.backup_before_alt_card_override_{stamp}.html")
backup.write_text(txt, encoding="utf-8")

start = "/* ALT_CARD_LAYOUT_OVERRIDE_START */"
end = "/* ALT_CARD_LAYOUT_OVERRIDE_END */"

override = r'''
/* ALT_CARD_LAYOUT_OVERRIDE_START */
/* Robust alternativliste-layout: kompakt mini-card uden vandret scroll. Kun visuel override. */

.right-panel.alternative-mode,
.right-panel.alternative-mode #tradeList,
.right-panel.alternative-mode .trade-list {
  overflow-x: hidden !important;
}

.right-panel.alternative-mode .trade-row.alternative {
  display: grid !important;
  grid-template-columns: 30px 30px minmax(0, 1fr) 64px !important;
  grid-template-areas:
    "rank flag main button" !important;
  column-gap: 8px !important;
  align-items: start !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  padding: 12px 8px !important;
  min-height: 86px !important;
  overflow: hidden !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-rank {
  grid-area: rank !important;
  width: 28px !important;
  height: 28px !important;
  min-width: 28px !important;
  align-self: start !important;
  justify-self: center !important;
  margin-top: 2px !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-flag-wrap {
  grid-area: flag !important;
  width: 28px !important;
  min-width: 28px !important;
  align-self: start !important;
  justify-self: center !important;
  margin-top: 3px !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-list-flag {
  width: 25px !important;
  height: 17px !important;
  max-width: 25px !important;
  object-fit: cover !important;
  border-radius: 3px !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-player-main {
  grid-area: main !important;
  min-width: 0 !important;
  max-width: 100% !important;
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) 104px !important;
  grid-template-areas:
    "name price"
    "meta price"
    "opp price"
    "start price" !important;
  column-gap: 8px !important;
  row-gap: 1px !important;
  align-items: start !important;
  overflow: hidden !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-name {
  grid-area: name !important;
  min-width: 0 !important;
  max-width: 100% !important;
  display: block !important;
  font-size: 15px !important;
  line-height: 1.12 !important;
  font-weight: 800 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-meta {
  grid-area: meta !important;
  min-width: 0 !important;
  display: block !important;
  font-size: 11px !important;
  line-height: 1.15 !important;
  color: #d7e1ef !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-player-opponent {
  grid-area: opp !important;
  min-width: 0 !important;
  display: block !important;
  font-size: 11px !important;
  line-height: 1.15 !important;
  color: #ffe45c !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.right-panel.alternative-mode .trade-row.alternative .alternative-bottom {
  display: contents !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col {
  grid-area: start !important;
  justify-self: start !important;
  text-align: left !important;
  min-width: 0 !important;
  margin-top: 4px !important;
  font-size: 11px !important;
  line-height: 1.05 !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-start-col strong,
.right-panel.alternative-mode .trade-row.alternative .trade-start-col .trade-start-percent {
  font-size: 14px !important;
  line-height: 1 !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-price {
  grid-area: price !important;
  justify-self: end !important;
  align-self: start !important;
  width: 104px !important;
  min-width: 104px !important;
  max-width: 104px !important;
  text-align: right !important;
  white-space: normal !important;
  overflow: visible !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-price-main {
  font-size: 15px !important;
  line-height: 1.1 !important;
  font-weight: 800 !important;
  white-space: nowrap !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-diff {
  font-size: 13px !important;
  line-height: 1.1 !important;
  font-weight: 800 !important;
  white-space: nowrap !important;
}

.right-panel.alternative-mode .trade-row.alternative .trade-transfer-note {
  font-size: 10px !important;
  line-height: 1.15 !important;
  white-space: nowrap !important;
  margin-top: 4px !important;
}

.right-panel.alternative-mode .trade-row.alternative .choose-alt-btn {
  grid-area: button !important;
  width: 60px !important;
  min-width: 60px !important;
  max-width: 60px !important;
  justify-self: end !important;
  align-self: start !important;
  padding: 8px 7px !important;
  box-sizing: border-box !important;
  white-space: nowrap !important;
  margin-top: 2px !important;
}

/* Sikkerhed: skjul vandret scrollbar fra tidligere grid-forsøg */
.right-panel.alternative-mode * {
  max-width: 100%;
}
/* ALT_CARD_LAYOUT_OVERRIDE_END */
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
print("Added/replaced ALT_CARD_LAYOUT_OVERRIDE")
