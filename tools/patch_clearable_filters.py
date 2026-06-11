from pathlib import Path
from datetime import datetime
import shutil

p = Path("index.html")
text = p.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = p.with_name(f"index.backup_before_clearable_filters_{stamp}.html")
shutil.copy2(p, backup)

css_marker = "/* CLEARABLE_FILTER_SELECTS_CSS_START */"
js_marker = "/* CLEARABLE_FILTER_SELECTS_JS_START */"

css = '''
/* CLEARABLE_FILTER_SELECTS_CSS_START */
.clearable-select-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.clearable-select-wrap select {
  width: 100%;
  padding-right: 44px;
}

.clear-filter-select-btn {
  position: absolute;
  right: 28px;
  top: 50%;
  transform: translateY(-50%);
  width: 18px;
  height: 18px;
  border: 0;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.32);
  color: #e5e7eb;
  font-size: 15px;
  line-height: 18px;
  font-weight: 800;
  cursor: pointer;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0;
  z-index: 3;
}

.clear-filter-select-btn:hover {
  background: rgba(248, 250, 252, 0.42);
  color: #ffffff;
}

.clearable-select-wrap.has-value .clear-filter-select-btn {
  display: inline-flex;
}
/* CLEARABLE_FILTER_SELECTS_CSS_END */
'''

js = '''
/* CLEARABLE_FILTER_SELECTS_JS_START */
function setupClearableFilterSelect(selectEl) {
  if (!selectEl || selectEl.dataset.clearableFilterReady === "1") return;

  const wrapper = document.createElement("div");
  wrapper.className = "clearable-select-wrap";

  selectEl.parentNode.insertBefore(wrapper, selectEl);
  wrapper.appendChild(selectEl);

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "clear-filter-select-btn";
  clearBtn.setAttribute("aria-label", "Ryd filter");
  clearBtn.title = "Ryd filter";
  clearBtn.textContent = "×";

  wrapper.appendChild(clearBtn);

  function syncClearButton() {
    wrapper.classList.toggle("has-value", !!selectEl.value);
  }

  clearBtn.addEventListener("click", event => {
    event.preventDefault();
    event.stopPropagation();

    if (!selectEl.value) return;

    selectEl.value = "";
    syncClearButton();

    selectEl.dispatchEvent(new Event("input", { bubbles: true }));
    selectEl.dispatchEvent(new Event("change", { bubbles: true }));
  });

  selectEl.addEventListener("input", syncClearButton);
  selectEl.addEventListener("change", syncClearButton);

  selectEl.dataset.clearableFilterReady = "1";
  syncClearButton();
}

function setupClearableFilterSelects() {
  setupClearableFilterSelect(teamFilter);
  setupClearableFilterSelect(positionFilter);
}
/* CLEARABLE_FILTER_SELECTS_JS_END */
'''

changes = []

if css_marker not in text:
    if "</style>" not in text:
        raise SystemExit("Kunne ikke finde </style> til CSS.")
    text = text.replace("</style>", css + "\n</style>", 1)
    changes.append("Tilføjet CSS til ryd-knapper på filtre")
else:
    print("CSS findes allerede.")

if js_marker not in text:
    anchor = "    function setupEvents() {"
    if anchor not in text:
        raise SystemExit("Kunne ikke finde setupEvents().")
    text = text.replace(anchor, js + "\n\n" + anchor, 1)
    changes.append("Tilføjet JS-helper til clearable filtre")
else:
    print("JS findes allerede.")

old = '''    function setupEvents() {
      openLoginBtn.addEventListener("click", () => openAuthModal("login"));'''

new = '''    function setupEvents() {
      setupClearableFilterSelects();

      openLoginBtn.addEventListener("click", () => openAuthModal("login"));'''

if old in text:
    text = text.replace(old, new, 1)
    changes.append("Aktiveret clearable filtre i setupEvents")
elif "setupClearableFilterSelects();" in text:
    print("setupEvents kalder allerede setupClearableFilterSelects().")
else:
    raise SystemExit("Kunne ikke indsætte kald i setupEvents().")

p.write_text(text, encoding="utf-8")

print("OK: Land og position har nu lille ryd-kryds, når filter er valgt.")
print(f"Backup: {backup}")
for c in changes:
    print("- " + c)

print("")
print("Sanity:")
for needle in [
    "clearable-select-wrap",
    "clear-filter-select-btn",
    "function setupClearableFilterSelect",
    "setupClearableFilterSelect(teamFilter)",
    "setupClearableFilterSelect(positionFilter)",
]:
    print(needle, "=>", text.count(needle))
