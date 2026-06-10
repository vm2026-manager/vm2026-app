const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const APP_URL = "http://127.0.0.1:8000/index.html?v=export_all";
const OUT_DIR = path.join("data", "strategy_squad_exports");

const STRATEGIES = [
  { label: "Næste runde", match: "Næste runde" },
  { label: "1. + 2. runde", match: "1. + 2." },
  { label: "Gruppespil", match: "Gruppespil" },
  { label: "Lang sigt", match: "Lang sigt" },
];

const FORMATIONS = [
  "3-4-3",
  "3-5-2",
  "4-3-3",
  "4-4-2",
  "4-5-1",
  "5-3-2",
  "5-4-1",
];

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n\r;]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

async function clickButtonContaining(page, text) {
  const ok = await page.evaluate((needle) => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const btn = buttons.find(b => (b.textContent || "").replace(/\s+/g, " ").trim().includes(needle));
    if (!btn) return false;
    btn.click();
    return true;
  }, text);

  if (!ok) throw new Error(`Kunne ikke finde knap med tekst: ${text}`);
}

async function setFormation(page, formation) {
  const ok = await page.evaluate((formation) => {
    const selects = Array.from(document.querySelectorAll("select"));
    for (const select of selects) {
      const option = Array.from(select.options).find(o =>
        (o.textContent || "").trim() === formation || String(o.value).trim() === formation
      );
      if (option) {
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      }
    }

    const buttons = Array.from(document.querySelectorAll("button"));
    const btn = buttons.find(b => (b.textContent || "").replace(/\s+/g, " ").trim() === formation);
    if (btn) {
      btn.click();
      return true;
    }

    return false;
  }, formation);

  if (!ok) throw new Error(`Kunne ikke sætte formation: ${formation}`);
}

async function extractSelectedSquad(page, strategyLabel, formation) {
  return await page.evaluate(async ({ strategyLabel, formation }) => {
    function clean(text) {
      return String(text || "").replace(/\s+/g, " ").trim();
    }

    function normName(text) {
      return clean(text).toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "");
    }

    function posDk(pos) {
      const p = String(pos || "").toUpperCase();
      if (p === "GK" || p === "MÅL" || p === "MAL") return "Mål";
      if (p === "DEF" || p === "FOR") return "For";
      if (p === "MID") return "Mid";
      if (p === "FWD" || p === "ANG") return "Ang";
      return pos || "";
    }

    function money(value) {
      const n = Number(value || 0);
      if (!Number.isFinite(n) || n <= 0) return "";
      return (n / 1000000).toFixed(1).replace(".", ",") + " mio.";
    }

    function pct(value) {
      const n = Number(value || 0);
      if (!Number.isFinite(n) || n <= 0) return "";
      return Math.round(n <= 1 ? n * 100 : n) + "%";
    }

    const poolRaw = await fetch("data/player_pool_v1.json").then(r => r.json());
    const players = Array.isArray(poolRaw) ? poolRaw : (poolRaw.players || poolRaw.data || []);

    const selectedSlotNodes = Array.from(document.querySelectorAll("button"))
      .filter(b => clean(b.textContent) === "Fjern")
      .map(button => {
        let node = button.parentElement;
        for (let i = 0; i < 7 && node; i++) {
          const text = clean(node.innerText || node.textContent);
          if (text.includes("Fjern") && text.includes("Skift") && text.length < 500) return node;
          node = node.parentElement;
        }
        return null;
      })
      .filter(Boolean);

    const uniqueNodes = [];
    const seenText = new Set();

    for (const node of selectedSlotNodes) {
      const text = clean(node.innerText || node.textContent);
      if (!seenText.has(text)) {
        seenText.add(text);
        uniqueNodes.push(node);
      }
    }

    const rows = [];

    for (const node of uniqueNodes) {
      const slotText = clean(node.innerText || node.textContent);

      let best = null;
      let bestLen = 0;

      for (const p of players) {
        const name = p.player_name || p.name || "";
        if (!name) continue;

        const n = normName(name);
        const s = normName(slotText);

        if (s.includes(n) && n.length > bestLen) {
          best = p;
          bestLen = n.length;
        }
      }

      if (!best) {
        rows.push({
          Strategi: strategyLabel,
          Formation: formation,
          Pos: "",
          Spiller: slotText.replace("Fjern", "").replace("Skift", "").trim(),
          Land: "",
          Pris: "",
          Start: "",
          EV: "",
          player_id: "",
          Raw: slotText,
        });
        continue;
      }

      rows.push({
        Strategi: strategyLabel,
        Formation: formation,
        Pos: posDk(best.position || best.holdet_position),
        Spiller: best.player_name || best.name || "",
        Land: best.team_name || best.team_id || best.team || "",
        Pris: money(best.price || best.price_estimate || best.holdet_price),
        Start: pct(best.start_probability_pct || best.start_prob || best.start_security),
        EV: best.optimizer_ev || best.weighted_group_stage_ev || "",
        player_id: best.player_id || "",
        Raw: slotText,
      });
    }

    const order = { "Mål": 1, "For": 2, "Mid": 3, "Ang": 4 };
    rows.sort((a, b) => (order[a.Pos] || 99) - (order[b.Pos] || 99) || a.Spiller.localeCompare(b.Spiller));

    return rows;
  }, { strategyLabel, formation });
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await chromium.launch({
    headless: false,
    channel: "chrome",
  });

  const page = await browser.newPage({
    viewport: { width: 1440, height: 1200 },
  });

  await page.goto(APP_URL, { waitUntil: "networkidle" });
  await sleep(1500);

  const allRows = [];

  for (const strategy of STRATEGIES) {
    console.log(`\nStrategi: ${strategy.label}`);

    await clickButtonContaining(page, strategy.match);
    await sleep(500);

    for (const formation of FORMATIONS) {
      console.log(`  Formation: ${formation}`);

      await setFormation(page, formation);
      await sleep(300);

      await clickButtonContaining(page, "Vælg optimalt hold");
      await sleep(900);

      const rows = await extractSelectedSquad(page, strategy.label, formation);

      rows.forEach(r => allRows.push(r));

      console.log(`    Spillere fundet: ${rows.length}`);
    }
  }

  await browser.close();

  const ts = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  const csvPath = path.join(OUT_DIR, `frontend_alle_strategier_formationer_${ts}.csv`);
  const mdPath = path.join(OUT_DIR, `frontend_alle_strategier_formationer_${ts}.md`);

  const headers = ["Strategi", "Formation", "Pos", "Spiller", "Land", "Pris", "Start", "EV", "player_id"];
  const csv = [
    headers.join(";"),
    ...allRows.map(r => headers.map(h => csvEscape(r[h])).join(";")),
  ].join("\n");

  fs.writeFileSync(csvPath, "\uFEFF" + csv, "utf8");

  let md = "# Alle hold - alle strategier og formationer\n\n";
  let current = "";

  for (const row of allRows) {
    const group = `${row.Strategi}|||${row.Formation}`;
    if (group !== current) {
      current = group;
      md += `\n## ${row.Strategi} - ${row.Formation}\n\n`;
      md += "| Pos | Spiller | Land | Pris | Start | EV |\n";
      md += "|---|---|---:|---:|---:|---:|\n";
    }

    md += `| ${row.Pos} | ${row.Spiller} | ${row.Land} | ${row.Pris} | ${row.Start} | ${row.EV} |\n`;
  }

  fs.writeFileSync(mdPath, md, "utf8");

  console.log("\nFÆRDIG");
  console.log("CSV:", csvPath);
  console.log("Markdown:", mdPath);
  console.log("Antal spillerrækker:", allRows.length);
})();
