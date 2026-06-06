from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PLAYER_POOL_PATH = DATA / "player_pool_v1.json"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
SIGNAL_PATH = DATA / "player_start_signal_layer_v1.csv"
STRATEGIES_PATH = DATA / "optimal_squads_by_strategy.json"

OUT_CSV = DATA / "expert_article_player_audit.csv"
OUT_MD = DATA / "expert_article_player_audit.md"


# Spillere nævnt i BetXpert-artiklen/kommentarer 5.-6. juni 2026.
# Fokus: sanity mod modeltal, ikke automatiske overrides.
WATCHLIST = [
    # Indlysende / oplagte / dyre offensive
    {"label": "Haaland", "query": "Erling Haaland", "team_id": "NOR", "note": "Artiklen: klart bedste kaptajnvalg i runde 1; ca. halvdelen af Norges xG."},
    {"label": "Wirtz", "query": "Florian Wirtz", "team_id": "GER", "note": "Artiklen: indlysende valg; stærk R1 mod Curacao."},
    {"label": "Mbappe", "query": "Kylian Mbappe", "team_id": "FRA", "note": "Artiklen: monster for Frankrig, men dyr og R1 mod Senegal er lunken."},
    {"label": "Oyarzabal", "query": "Mikel Oyarzabal", "team_id": "ESP", "note": "Artiklen: høj xG/90 og ligner sikker R1-starter."},
    {"label": "Ronaldo", "query": "Cristiano Ronaldo", "team_id": "POR", "note": "Artiklen: 3. bedste dyre valg efter Haaland/Oyarzabal; stor xG-andel."},
    {"label": "Musiala", "query": "Jamal Musiala", "team_id": "GER", "note": "Artiklen: ønskelig tysk offensiv."},
    {"label": "Taremi", "query": "Mehdi Taremi", "team_id": "IRN", "note": "Artiklen: billig angriber; over halvdelen af Irans xG."},
    {"label": "Embolo", "query": "Breel Embolo", "team_id": "SUI", "note": "Artiklen: mindre oplagt valg."},
    {"label": "Raphinha", "query": "Raphinha", "team_id": "BRA", "note": "Artiklen: mindre oplagt valg; vores fallback løftede ham."},
    {"label": "Bruno Fernandes", "query": "Bruno Fernandes", "team_id": "POR", "note": "Artiklen: ser godt ud til første to runder."},
    {"label": "Kane", "query": "Harry Kane", "team_id": "ENG", "note": "Artiklen nævner ham som øvrigt valg; vi har manuel nailed-on starter override."},
    {"label": "Messi", "query": "Lionel Messi", "team_id": "ARG", "note": "Kommentar: Messi har langt højere xGI pr. start end Alvarez."},
    {"label": "Julian Alvarez", "query": "Julian Alvarez", "team_id": "ARG", "note": "Kommentar: mindre god R1 value end Messi; evt. R3/R4."},

    # Midtbane / talismaner
    {"label": "McTominay", "query": "Scott McTominay", "team_id": "SCO", "note": "Artiklen: hyper oplagt, Skotlands klart farligste våben."},
    {"label": "James Rodriguez", "query": "James Rodriguez", "team_id": "COL", "note": "Artiklen: høj assist-rate og straffetjans."},
    {"label": "Sabitzer", "query": "Marcel Sabitzer", "team_id": "AUT", "note": "Artiklen: dødbolde og stor offensiv involvering for Østrig."},
    {"label": "Pedri", "query": "Pedri", "team_id": "ESP", "note": "Artiklen: overraskende høje landskamp-xG+xA."},
    {"label": "Fabian Ruiz", "query": "Fabian Ruiz", "team_id": "ESP", "note": "Artiklen positiv, men vi nedjusterede pga. skader/fravær."},
    {"label": "Nmecha", "query": "Felix Nmecha", "team_id": "GER", "note": "Artiklen: billig enabler, usikker men høj value hvis starter."},
    {"label": "Alvarado", "query": "Roberto Alvarado", "team_id": "MEX", "note": "Artiklen: billig enabler fra Mexico."},
    {"label": "Nusa", "query": "Antonio Nusa", "team_id": "NOR", "note": "Kommentar: eksperten har Nusa på flere hold."},
    {"label": "McGinn", "query": "John McGinn", "team_id": "SCO", "note": "Kommentar: interessant billig midtbane."},
    {"label": "Tielemans", "query": "Youri Tielemans", "team_id": "BEL", "note": "Kommentar: måske undervurderet; stærke xGI-tal."},

    # Forsvar
    {"label": "Ajer", "query": "Kristoffer Ajer", "team_id": "NOR", "note": "Artiklen: foretrukken Norge-forsvarer foran Ryerson/Wolfe."},
    {"label": "Ryerson", "query": "Julian Ryerson", "team_id": "NOR", "note": "Artiklen: relevant Norge-forsvarer, men dyrere end Ajer."},
    {"label": "Wolfe", "query": "David Moller Wolfe", "team_id": "NOR", "note": "Artiklen: Norge-defensiv value."},
    {"label": "Schlotterbeck", "query": "Nico Schlotterbeck", "team_id": "GER", "note": "Artiklen: tysk forsvarer med god R1."},
    {"label": "Nathaniel Brown", "query": "Nathaniel Brown", "team_id": "GER", "note": "Artiklen: enormt interessant hvis starter; ca. 60 pct. nok."},
    {"label": "Davinson Sanchez", "query": "Davinson Sanchez", "team_id": "COL", "note": "Artiklen: suverænt køb til to runder."},
    {"label": "Nuno Mendes", "query": "Nuno Mendes", "team_id": "POR", "note": "Artiklen: offensivt forsvarsvalg."},
    {"label": "Cucurella", "query": "Marc Cucurella", "team_id": "ESP", "note": "Artiklen: oplagt men dyr spansk defensiv."},
    {"label": "Laporte", "query": "Aymeric Laporte", "team_id": "ESP", "note": "Artiklen: oplagt men dyr spansk defensiv."},

    # Målmænd
    {"label": "Kobel", "query": "Gregor Kobel", "team_id": "SUI", "note": "Artiklen: foretrukken keeper."},
    {"label": "Nyland", "query": "Orjan Nyland", "team_id": "NOR", "note": "Artiklen: godt alternativ til Schweiz-stack."},
    {"label": "Beiranvand", "query": "Alireza Beiranvand", "team_id": "IRN", "note": "Artiklen: stærk billig keeper til én runde."},
    {"label": "Crepeau", "query": "Maxime Crepeau", "team_id": "CAN", "note": "Kommentar: Canadas træner har meldt ham som R1-starter ifølge artiklen."},

    # Canada value
    {"label": "De Fougerolles", "query": "Luc De Fougerolles", "team_id": "CAN", "note": "Kommentar: startede inde og forventes genvalg; 2.0m value hvis i pool."},
]


def norm(value: object) -> str:
    return str(value or "").strip().casefold()


def contains_name(series: pd.Series, query: str) -> pd.Series:
    q = norm(query)
    return series.astype(str).str.casefold().str.contains(q, regex=False, na=False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pool = pd.DataFrame(json.loads(PLAYER_POOL_PATH.read_text(encoding="utf-8")))
    ev = pd.read_csv(EV_PATH)
    sig = pd.read_csv(SIGNAL_PATH) if SIGNAL_PATH.exists() else pd.DataFrame()
    strategies = json.loads(STRATEGIES_PATH.read_text(encoding="utf-8")) if STRATEGIES_PATH.exists() else {}
    return pool, ev, sig, strategies


def find_player(pool: pd.DataFrame, query: str, team_id: str) -> pd.DataFrame:
    team_id = team_id.upper().strip()
    exact = pool[
        pool["player_name"].astype(str).str.casefold().eq(norm(query))
        & pool["team_id"].astype(str).str.upper().eq(team_id)
    ]
    if len(exact) > 0:
        return exact

    fuzzy = pool[
        contains_name(pool["player_name"], query)
        & pool["team_id"].astype(str).str.upper().eq(team_id)
    ]
    if len(fuzzy) > 0:
        return fuzzy

    # Fallback: søg på label uden teamfilter, hvis Holdet/team-mapping afviger.
    fuzzy_any_team = pool[contains_name(pool["player_name"], query)]
    return fuzzy_any_team


def selected_strategies(strategies: dict, player_id: str) -> str:
    hits = []
    for strategy, value in strategies.items():
        for p in value.get("best_squad", []):
            if str(p.get("player_id", "")) == player_id:
                hits.append(strategy)
    return ";".join(hits)


def issue_flags(row: dict) -> str:
    flags: list[str] = []

    status = row.get("match_status")
    if status != "found":
        flags.append(status or "not_found")

    pos = row.get("position", "")
    start = row.get("start_prob")
    ev = row.get("optimizer_ev")
    price = row.get("price_m")
    selected = row.get("selected_strategies", "")

    try:
        start_f = float(start)
    except Exception:
        start_f = 0.0

    try:
        ev_f = float(ev)
    except Exception:
        ev_f = 0.0

    try:
        price_f = float(price)
    except Exception:
        price_f = 0.0

    if status == "found":
        if start_f < 0.50:
            flags.append("low_start_prob")
        elif start_f < 0.70:
            flags.append("start_prob_watch")

        if pos in {"MID", "FWD"} and ev_f < 2.0 and start_f >= 0.70:
            flags.append("offensive_ev_maybe_low")

        if pos == "FWD" and price_f >= 7.0 and ev_f < 3.0:
            flags.append("premium_fwd_low_ev")

        if selected:
            flags.append("selected_by_optimizer")

    return ";".join(flags)


def main() -> int:
    pool, ev, sig, strategies = load_data()

    ev_cols = [
        "player_id",
        "optimizer_ev",
        "weighted_group_stage_ev",
        "weighted_group_stage_ev_before_price_quality",
        "price_quality_ev",
        "start_prob",
        "conditional_start_prob",
        "availability_risk",
        "offensive_fallback_applied",
        "goal_share_norm",
        "assist_share_norm",
        "sot_share_norm",
        "round_context_source",
        "match_1_opponent_team",
        "match_1_weighted_match_ev",
        "match_2_opponent_team",
        "match_2_weighted_match_ev",
        "match_3_opponent_team",
        "match_3_weighted_match_ev",
    ]
    ev_small = ev[[c for c in ev_cols if c in ev.columns]].copy()

    sig_cols = ["player_id", "start_signal_tier", "start_signal_rank_adjusted"]
    sig_small = sig[[c for c in sig_cols if c in sig.columns]].copy() if not sig.empty else pd.DataFrame(columns=sig_cols)

    rows: list[dict] = []

    for item in WATCHLIST:
        matches = find_player(pool, item["query"], item["team_id"])

        if len(matches) == 0:
            rows.append({
                "label": item["label"],
                "query": item["query"],
                "expected_team_id": item["team_id"],
                "match_status": "not_found",
                "note": item["note"],
                "issue_flags": "not_found",
            })
            continue

        if len(matches) > 1:
            # Vælg bedste team-match hvis muligt, men markér tvetydigt.
            team_matches = matches[matches["team_id"].astype(str).str.upper().eq(item["team_id"])]
            selected = team_matches.iloc[0] if len(team_matches) else matches.iloc[0]
            match_status = f"multiple_matches_{len(matches)}"
        else:
            selected = matches.iloc[0]
            match_status = "found"

        player_id = str(selected.get("player_id", ""))

        row = {
            "label": item["label"],
            "query": item["query"],
            "expected_team_id": item["team_id"],
            "match_status": match_status,
            "player_id": player_id,
            "player_name": selected.get("player_name", ""),
            "team_id": selected.get("team_id", ""),
            "position": selected.get("position", ""),
            "price": selected.get("price", ""),
            "price_m": (float(selected.get("price", 0) or 0) / 1_000_000),
            "note": item["note"],
        }

        e = ev_small[ev_small["player_id"].astype(str).eq(player_id)]
        if len(e):
            for col, val in e.iloc[0].items():
                if col != "player_id":
                    row[col] = val

        s = sig_small[sig_small["player_id"].astype(str).eq(player_id)]
        if len(s):
            for col, val in s.iloc[0].items():
                if col != "player_id":
                    row[col] = val

        row["selected_strategies"] = selected_strategies(strategies, player_id)
        row["issue_flags"] = issue_flags(row)
        rows.append(row)

    out = pd.DataFrame(rows)

    # Gør output mere læsbart.
    preferred_cols = [
        "label",
        "match_status",
        "player_name",
        "team_id",
        "position",
        "price_m",
        "start_prob",
        "conditional_start_prob",
        "availability_risk",
        "start_signal_tier",
        "optimizer_ev",
        "weighted_group_stage_ev_before_price_quality",
        "price_quality_ev",
        "offensive_fallback_applied",
        "goal_share_norm",
        "assist_share_norm",
        "sot_share_norm",
        "round_context_source",
        "match_1_opponent_team",
        "match_1_weighted_match_ev",
        "match_2_opponent_team",
        "match_2_weighted_match_ev",
        "match_3_opponent_team",
        "match_3_weighted_match_ev",
        "selected_strategies",
        "issue_flags",
        "note",
        "query",
        "expected_team_id",
        "player_id",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    out = out[cols]

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    found = out[out["match_status"].astype(str).eq("found")]
    not_found = out[out["match_status"].astype(str).eq("not_found")]
    flagged = out[out["issue_flags"].fillna("").astype(str).ne("")]

    lines = []
    lines.append("# Expert article player audit")
    lines.append("")
    lines.append("Audit baseret på BetXpert-artiklen om VM 2026 Manager og efterfølgende kommentarer.")
    lines.append("Formålet er sanity-check, ikke automatiske overrides.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Watchlist entries: {len(out)}")
    lines.append(f"- Found exact/single matches: {len(found)}")
    lines.append(f"- Not found: {len(not_found)}")
    lines.append(f"- Entries with flags: {len(flagged)}")
    lines.append("")
    lines.append("## Selected by optimizer")
    lines.append("")
    selected = out[out["selected_strategies"].fillna("").astype(str).ne("")]
    if selected.empty:
        lines.append("Ingen watchlist-spillere er valgt i de aktuelle optimizer-hold.")
    else:
        lines.append("| Player | Team | Pos | Price | Start | EV | Strategies | Flags |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for _, r in selected.sort_values(["selected_strategies", "optimizer_ev"], ascending=[True, False]).iterrows():
            lines.append(
                f"| {r.get('player_name','')} | {r.get('team_id','')} | {r.get('position','')} | "
                f"{float(r.get('price_m') or 0):.1f} | {float(r.get('start_prob') or 0):.3f} | "
                f"{float(r.get('optimizer_ev') or 0):.3f} | {r.get('selected_strategies','')} | {r.get('issue_flags','')} |"
            )

    lines.append("")
    lines.append("## Important flags")
    lines.append("")
    if flagged.empty:
        lines.append("Ingen flags.")
    else:
        lines.append("| Player/label | Status | Team | Pos | Start | EV | Flags | Note |")
        lines.append("|---|---|---|---:|---:|---:|---|---|")
        for _, r in flagged.iterrows():
            player = r.get("player_name") or r.get("label")
            start = r.get("start_prob")
            evv = r.get("optimizer_ev")
            start_txt = "" if pd.isna(start) else f"{float(start):.3f}"
            ev_txt = "" if pd.isna(evv) else f"{float(evv):.3f}"
            lines.append(
                f"| {player} | {r.get('match_status','')} | {r.get('team_id','')} | {r.get('position','')} | "
                f"{start_txt} | {ev_txt} | {r.get('issue_flags','')} | {r.get('note','')} |"
            )

    lines.append("")
    lines.append("## Full table")
    lines.append("")
    lines.append("| Label | Player | Team | Pos | Price | Start | EV | Strategies | Flags |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
    for _, r in out.iterrows():
        start = r.get("start_prob")
        evv = r.get("optimizer_ev")
        start_txt = "" if pd.isna(start) else f"{float(start):.3f}"
        ev_txt = "" if pd.isna(evv) else f"{float(evv):.3f}"
        price = r.get("price_m")
        price_txt = "" if pd.isna(price) else f"{float(price):.1f}"
        lines.append(
            f"| {r.get('label','')} | {r.get('player_name','')} | {r.get('team_id','')} | {r.get('position','')} | "
            f"{price_txt} | {start_txt} | {ev_txt} | {r.get('selected_strategies','')} | {r.get('issue_flags','')} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Expert article player audit")
    print("---------------------------")
    print(f"Watchlist entries: {len(out)}")
    print(f"Found: {len(found)}")
    print(f"Not found: {len(not_found)}")
    print(f"Flagged: {len(flagged)}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")
    print()
    print("Selected watchlist players:")
    if selected.empty:
        print("  none")
    else:
        show_cols = ["player_name", "team_id", "position", "price_m", "start_prob", "optimizer_ev", "selected_strategies", "issue_flags"]
        print(selected[[c for c in show_cols if c in selected.columns]].to_string(index=False))

    print()
    print("Important flags:")
    if flagged.empty:
        print("  none")
    else:
        show_cols = ["label", "player_name", "team_id", "position", "start_prob", "optimizer_ev", "issue_flags"]
        print(flagged[[c for c in show_cols if c in flagged.columns]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())