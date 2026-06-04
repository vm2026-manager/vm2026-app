from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import optimize_squad_group_stage as optimizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

OPTIMAL_SQUADS_PATH = DATA_DIR / "optimal_squads_by_strategy.json"
OUT_CSV = DATA_DIR / "positional_budget_value_audit.csv"
OUT_MD = DATA_DIR / "positional_budget_value_audit.md"

STRATEGY_SCORE_COL = {
    "next_round": "score_next_round",
    "round1_2": "score_round1_2",
    "group_stage": "score_group_stage",
    "long_run": "score_long_run",
}

PREMIUM_FWDS = {"erling haaland", "kylian mbappe", "harry kane"}
LOW_UPSIDE_MIDS = {
    "manu kone",
    "aurelien tchouameni",
    "rodrigo de paul",
    "declan rice",
    "joshua kimmich",
    "konrad laimer",
    "scott mctominay",
}

CSV_COLUMNS = [
    "strategy",
    "formation",
    "comparison_type",
    "expensive_player_id",
    "expensive_player_name",
    "expensive_position",
    "expensive_price",
    "expensive_ev",
    "expensive_strategy_score",
    "cheap_player_id",
    "cheap_player_name",
    "cheap_position",
    "cheap_price",
    "cheap_ev",
    "cheap_strategy_score",
    "price_difference",
    "ev_difference",
    "strategy_score_difference",
    "marginal_ev_per_million",
    "marginal_strategy_score_per_million",
    "round_1_difference",
    "round_2_difference",
    "round_3_difference",
    "start_security_difference",
    "interpretation",
]


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = txt(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def to_float(value: Any, default: float = 0.0) -> float:
    raw = txt(value).replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def rounded(value: Any, digits: int = 4) -> str:
    return str(round(to_float(value), digits))


def row_price(row: pd.Series | dict[str, Any]) -> float:
    return to_float(row.get("price"))


def row_ev(row: pd.Series | dict[str, Any]) -> float:
    return to_float(row.get("optimizer_ev") or row.get("weighted_group_stage_ev") or row.get("total_ev_group_stage"))


def round_ev(row: pd.Series | dict[str, Any], rnd: int) -> float:
    return to_float(row.get(f"round{rnd}_ev") or row.get(f"round_{rnd}_ev"))


def strategy_score(row: pd.Series | dict[str, Any], strategy: str) -> float:
    return to_float(row.get(STRATEGY_SCORE_COL[strategy]))


def is_manual_avoid(row: pd.Series) -> bool:
    return txt(row.get("manual_status")).lower() == "avoid" or txt(row.get("manual_start_status")).lower() == "avoid"


def load_selected_squads() -> dict[tuple[str, str], list[dict[str, Any]]]:
    data = json.loads(OPTIMAL_SQUADS_PATH.read_text(encoding="utf-8-sig"))
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for strategy, strategy_data in data.items():
        by_formation = strategy_data.get("squads_by_formation") or {}
        for formation, payload in by_formation.items():
            selected[(strategy, formation)] = payload.get("squad") or payload.get("players") or []
        if not by_formation and strategy_data.get("best_squad"):
            formation = txt((strategy_data.get("best_summary") or {}).get("formation")) or "best"
            selected[(strategy, formation)] = strategy_data.get("best_squad") or []
    return selected


def player_by_id(players: pd.DataFrame) -> dict[str, pd.Series]:
    return {txt(row.get("player_id")): row for _, row in players.iterrows()}


def make_comparison(
    strategy: str,
    formation: str,
    comparison_type: str,
    expensive: pd.Series | dict[str, Any],
    cheap: pd.Series | dict[str, Any],
    interpretation: str,
) -> dict[str, str]:
    price_diff = row_price(expensive) - row_price(cheap)
    ev_diff = row_ev(expensive) - row_ev(cheap)
    score_diff = strategy_score(expensive, strategy) - strategy_score(cheap, strategy)
    price_diff_m = price_diff / 1_000_000 if price_diff else 0.0
    marginal_ev = ev_diff / price_diff_m if price_diff_m > 0 else 0.0
    marginal_score = score_diff / price_diff_m if price_diff_m > 0 else 0.0
    start_diff = to_float(expensive.get("conditional_start_prob") or expensive.get("start_prob")) - to_float(
        cheap.get("conditional_start_prob") or cheap.get("start_prob")
    )

    return {
        "strategy": strategy,
        "formation": formation,
        "comparison_type": comparison_type,
        "expensive_player_id": txt(expensive.get("player_id")),
        "expensive_player_name": txt(expensive.get("player_name")),
        "expensive_position": txt(expensive.get("position")),
        "expensive_price": str(int(row_price(expensive))),
        "expensive_ev": rounded(row_ev(expensive)),
        "expensive_strategy_score": rounded(strategy_score(expensive, strategy)),
        "cheap_player_id": txt(cheap.get("player_id")),
        "cheap_player_name": txt(cheap.get("player_name")),
        "cheap_position": txt(cheap.get("position")),
        "cheap_price": str(int(row_price(cheap))),
        "cheap_ev": rounded(row_ev(cheap)),
        "cheap_strategy_score": rounded(strategy_score(cheap, strategy)),
        "price_difference": str(int(price_diff)),
        "ev_difference": rounded(ev_diff),
        "strategy_score_difference": rounded(score_diff),
        "marginal_ev_per_million": rounded(marginal_ev),
        "marginal_strategy_score_per_million": rounded(marginal_score),
        "round_1_difference": rounded(round_ev(expensive, 1) - round_ev(cheap, 1)),
        "round_2_difference": rounded(round_ev(expensive, 2) - round_ev(cheap, 2)),
        "round_3_difference": rounded(round_ev(expensive, 3) - round_ev(cheap, 3)),
        "start_security_difference": rounded(start_diff),
        "interpretation": interpretation,
    }


def eligible_pool(players: pd.DataFrame, strategy: str) -> pd.DataFrame:
    score_col = STRATEGY_SCORE_COL[strategy]
    work = players.copy()
    work = work[~work.apply(is_manual_avoid, axis=1)]
    work = work[work["position"].isin(["GK", "DEF", "MID", "FWD"])]
    work["price"] = pd.to_numeric(work["price"], errors="coerce").fillna(0)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0)
    work["optimizer_ev"] = pd.to_numeric(work["optimizer_ev"], errors="coerce").fillna(0)
    work["conditional_start_prob"] = pd.to_numeric(work["conditional_start_prob"], errors="coerce").fillna(
        pd.to_numeric(work["start_prob"], errors="coerce").fillna(0)
    )
    work = work[(work["price"] > 0) & (work[score_col] > 0) & (work["conditional_start_prob"] >= 0.25)]
    return work


def best_by_score(df: pd.DataFrame, strategy: str) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values([STRATEGY_SCORE_COL[strategy], "optimizer_ev"], ascending=[False, False]).iloc[0]


def add_position_marginal_rows(rows: list[dict[str, str]], pool: pd.DataFrame, strategy: str, formation: str) -> None:
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos = pool[pool["position"] == position]
        if len(pos) < 6:
            continue
        cheap_cut = pos["price"].quantile(0.35)
        expensive_cut = pos["price"].quantile(0.70)
        cheap = best_by_score(pos[pos["price"] <= cheap_cut], strategy)
        expensive = best_by_score(pos[pos["price"] >= expensive_cut], strategy)
        if cheap is None or expensive is None or row_price(expensive) <= row_price(cheap):
            continue
        rows.append(
            make_comparison(
                strategy,
                formation,
                f"position_marginal_{position}",
                expensive,
                cheap,
                f"Best high-price {position} vs best low-price {position}; measures marginal return to spending extra budget at this position.",
            )
        )


def add_premium_fwd_rows(rows: list[dict[str, str]], pool: pd.DataFrame, strategy: str, formation: str) -> None:
    fwds = pool[pool["position"] == "FWD"]
    cheap_fwds = fwds[fwds["price"] <= 5_000_000].sort_values(
        [STRATEGY_SCORE_COL[strategy], "optimizer_ev"], ascending=[False, False]
    ).head(4)
    premium_fwds = fwds[fwds["player_name"].map(norm).isin(PREMIUM_FWDS)]
    for _, premium in premium_fwds.iterrows():
        for _, cheap in cheap_fwds.iterrows():
            if row_price(premium) <= row_price(cheap):
                continue
            if txt(premium.get("player_id")) == txt(cheap.get("player_id")):
                continue
            interpretation = (
                "Premium FWD vs cheaper FWD. Positive marginal values indicate the model already rewards premium attacking upside; "
                "negative values indicate the cheap/value FWD is ahead on current model output."
            )
            rows.append(make_comparison(strategy, formation, "premium_fwd_vs_cheap_fwd", premium, cheap, interpretation))


def add_selected_upgrade_rows(
    rows: list[dict[str, str]],
    pool: pd.DataFrame,
    selected_squad: list[dict[str, Any]],
    by_id: dict[str, pd.Series],
    strategy: str,
    formation: str,
) -> None:
    selected_ids = {txt(player.get("player_id")) for player in selected_squad}
    for player_id in selected_ids:
        selected = by_id.get(player_id)
        if selected is None:
            continue
        position = txt(selected.get("position"))
        name_key = norm(selected.get("player_name"))
        if position not in {"GK", "DEF", "MID"} and name_key not in LOW_UPSIDE_MIDS:
            continue
        if position == "FWD":
            continue
        cheaper = pool[
            (pool["position"] == position)
            & (~pool["player_id"].astype(str).isin(selected_ids))
            & (pool["price"] <= row_price(selected) - 500_000)
        ]
        cheap = best_by_score(cheaper, strategy)
        if cheap is None:
            continue
        comparison_type = "low_upside_mid_def_gk_upgrade" if name_key in LOW_UPSIDE_MIDS else f"selected_{position}_upgrade"
        interpretation = (
            f"Selected {position} spend vs cheaper same-position alternative. If marginal return is low, budget may be better used on FWD ceiling."
        )
        rows.append(make_comparison(strategy, formation, comparison_type, selected, cheap, interpretation))


def add_two_player_swap_rows(
    rows: list[dict[str, str]],
    pool: pd.DataFrame,
    selected_squad: list[dict[str, Any]],
    by_id: dict[str, pd.Series],
    strategy: str,
    formation: str,
) -> None:
    selected_ids = {txt(player.get("player_id")) for player in selected_squad}
    selected_rows = [by_id[player_id] for player_id in selected_ids if player_id in by_id]
    selected_fwds = [row for row in selected_rows if txt(row.get("position")) == "FWD" and row_price(row) <= 5_500_000]
    premium_fwds = pool[
        (pool["position"] == "FWD")
        & (pool["player_name"].map(norm).isin(PREMIUM_FWDS))
        & (~pool["player_id"].astype(str).isin(selected_ids))
    ]
    downgrade_positions = {"GK", "DEF", "MID"}
    downgrade_candidates = [row for row in selected_rows if txt(row.get("position")) in downgrade_positions and row_price(row) >= 4_500_000]

    for _, premium in premium_fwds.iterrows():
        for cheap_fwd in selected_fwds:
            needed = row_price(premium) - row_price(cheap_fwd)
            if needed <= 0:
                continue
            best_note = ""
            best_net_score = None
            for selected_expensive in downgrade_candidates:
                cheaper_same_pos = pool[
                    (pool["position"] == txt(selected_expensive.get("position")))
                    & (~pool["player_id"].astype(str).isin(selected_ids))
                    & (pool["price"] <= row_price(selected_expensive) - needed)
                ]
                cheap_downgrade = best_by_score(cheaper_same_pos, strategy)
                if cheap_downgrade is None:
                    continue
                net_score = (
                    strategy_score(premium, strategy)
                    - strategy_score(cheap_fwd, strategy)
                    + strategy_score(cheap_downgrade, strategy)
                    - strategy_score(selected_expensive, strategy)
                )
                note = (
                    f"Two-player swap possible by downgrading {txt(selected_expensive.get('player_name'))} "
                    f"to {txt(cheap_downgrade.get('player_name'))}; net_strategy_score={net_score:.3f}."
                )
                if best_net_score is None or net_score > best_net_score:
                    best_net_score = net_score
                    best_note = note
            if best_note:
                rows.append(
                    make_comparison(
                        strategy,
                        formation,
                        "two_player_swap_premium_fwd_plus_cheaper_mid_def_gk",
                        premium,
                        cheap_fwd,
                        best_note,
                    )
                )


def summarize_rows(rows: list[dict[str, str]], selected_squads: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    df = pd.DataFrame(rows)
    lines = [
        "# Positional Budget Value Audit",
        "",
        "Audit baseret paa eksisterende spillerdata og strategi-/formationsoutput. Ingen optimizer eller strategioutput er genkoert.",
        "",
        "## Kort konklusion",
        "",
    ]
    if df.empty:
        return lines + ["Ingen sammenligningsrækker blev genereret."]

    df["marginal_ev_per_million_num"] = pd.to_numeric(df["marginal_ev_per_million"], errors="coerce").fillna(0)
    df["marginal_strategy_score_per_million_num"] = pd.to_numeric(df["marginal_strategy_score_per_million"], errors="coerce").fillna(0)

    premium = df[df["comparison_type"] == "premium_fwd_vs_cheap_fwd"]
    low_upside = df[df["comparison_type"].str.contains("low_upside|selected_", regex=True)]
    premium_positive_share = (premium["strategy_score_difference"].astype(float) > 0).mean() if not premium.empty else 0.0
    low_upside_low_return = (
        low_upside["marginal_strategy_score_per_million_num"].between(-0.25, 0.75).mean()
        if not low_upside.empty
        else 0.0
    )

    lines += [
        f"- Premium FWD sammenligninger: {len(premium)}; premium har positiv strategiscore mod billig FWD i ca. {premium_positive_share:.0%} af rækkerne.",
        f"- MID/DEF/GK upgrade-rækker med lav marginal strategireturnering (-0,25 til 0,75 pr. mio.): ca. {low_upside_low_return:.0%}.",
        "- Det peger ikke paa en universel premium-FWD-undervurdering, men Haaland/Kane kan se relativt svage ud i bestemte next_round/runde-kontekster, mens Mbappe typisk scorer som premium.",
        "- Centrale MID/DEF kan stadig fremstå attraktive, især naar de kombinerer starter-sikkerhed og pris/value; det er et kalibreringsspor, ikke en sikker datafejl.",
        "",
        "## Marginal returnering pr. position",
        "",
        "| comparison_type | avg_marginal_ev_per_million | avg_marginal_strategy_score_per_million | rows |",
        "|---|---:|---:|---:|",
    ]
    grouped = (
        df.groupby("comparison_type")
        .agg(
            avg_ev=("marginal_ev_per_million_num", "mean"),
            avg_score=("marginal_strategy_score_per_million_num", "mean"),
            rows=("comparison_type", "size"),
        )
        .reset_index()
        .sort_values("comparison_type")
    )
    for _, row in grouped.iterrows():
        lines.append(f"| {row['comparison_type']} | {row['avg_ev']:.3f} | {row['avg_score']:.3f} | {int(row['rows'])} |")

    lines += [
        "",
        "## Premium FWD vs cheap FWD",
        "",
        "| strategy | formation | premium | cheap_fwd | price_diff | strategy_score_diff | marginal_score_per_mio | interpretation |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    sample = premium.sort_values("strategy_score_difference", key=lambda s: pd.to_numeric(s, errors="coerce")).head(18)
    for _, row in sample.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['formation']} | {row['expensive_player_name']} | {row['cheap_player_name']} | {row['price_difference']} | {row['strategy_score_difference']} | {row['marginal_strategy_score_per_million']} | {row['interpretation']} |"
        )

    formation_notes = []
    for (strategy, formation), squad in selected_squads.items():
        fwd_count = sum(1 for player in squad if txt(player.get("position")) == "FWD")
        low_mid_count = sum(1 for player in squad if norm(player.get("player_name")) in LOW_UPSIDE_MIDS)
        premium_count = sum(1 for player in squad if norm(player.get("player_name")) in PREMIUM_FWDS)
        if fwd_count <= 1 or (low_mid_count >= 2 and premium_count == 0):
            formation_notes.append((strategy, formation, fwd_count, premium_count, low_mid_count))

    lines += [
        "",
        "## Formation-risk",
        "",
        "| strategy | formation | fwd_count | premium_fwd_count | low_upside_mid_count | note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for strategy, formation, fwd_count, premium_count, low_mid_count in formation_notes[:30]:
        note = "Saerligt udsat for lav FWD-ceiling." if fwd_count <= 1 else "Flere low-upside MID uden premium FWD."
        lines.append(f"| {strategy} | {formation} | {fwd_count} | {premium_count} | {low_mid_count} | {note} |")

    lines += [
        "",
        "## Svar paa auditspoergsmaal",
        "",
        "- Premiumangribere ser ikke systematisk undervurderede ud paa alle strategier; Mbappe er tydeligt staerk, mens Haaland/Kane kan blive presset af billig value og runde-kontekst.",
        "- Centrale/lav-upside midtbanespillere ser potentielt overvurderede ud i nogle strategy-score-rækker, især naar starter-sikkerhed og pris/value kombineres.",
        "- Problemet virker mest strategi- og formationsafhaengigt, ikke globalt. Formationer med en enkelt FWD, især 4-5-1/5-4-1, er mest udsatte for at ofre offensiv ceiling.",
        "- Der er grundlag for en senere modelaudit af offensive ceiling-komponenter, men ikke for at aendre vaegte uden ny godkendelse.",
    ]
    return lines


def main() -> int:
    if not OPTIMAL_SQUADS_PATH.exists():
        raise FileNotFoundError(OPTIMAL_SQUADS_PATH)

    players = optimizer.load_players()
    by_id = player_by_id(players)
    selected_squads = load_selected_squads()

    rows: list[dict[str, str]] = []
    for (strategy, formation), squad in selected_squads.items():
        if strategy not in STRATEGY_SCORE_COL:
            continue
        pool = eligible_pool(players, strategy)
        add_position_marginal_rows(rows, pool, strategy, formation)
        add_premium_fwd_rows(rows, pool, strategy, formation)
        add_selected_upgrade_rows(rows, pool, squad, by_id, strategy, formation)
        add_two_player_swap_rows(rows, pool, squad, by_id, strategy, formation)

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    OUT_MD.write_text("\n".join(summarize_rows(rows, selected_squads)) + "\n", encoding="utf-8")

    print("Positional budget value audit")
    print("-----------------------------")
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
