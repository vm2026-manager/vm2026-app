from __future__ import annotations

import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import optimize_squad_group_stage as optimizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FIXTURE_MULTIPLIERS_PATH = DATA_DIR / "fixture_strength_multipliers.csv"
SET_PIECE_MATCHED_PATH = DATA_DIR / "set_piece_takers_matched.csv"

OUT_HAALAND_KANE_CSV = DATA_DIR / "haaland_kane_round_context_audit.csv"
OUT_HAALAND_KANE_MD = DATA_DIR / "haaland_kane_round_context_audit.md"
OUT_EXPERIMENT_CSV = DATA_DIR / "offensive_ceiling_model_experiment.csv"
OUT_EXPERIMENT_MD = DATA_DIR / "offensive_ceiling_model_experiment.md"
OUT_PLAYER_COMPARISON_CSV = DATA_DIR / "model_experiment_player_score_comparison.csv"
OUT_FORMATION_COMPARISON_CSV = DATA_DIR / "model_experiment_formation_comparison.csv"
OUT_SQUAD_COMPARISON_MD = DATA_DIR / "model_experiment_squad_comparison.md"

TARGET_NAMES = {
    "erling haaland",
    "kylian mbappe",
    "harry kane",
    "luis diaz",
    "michael olise",
    "jamal musiala",
    "konrad laimer",
    "scott mctominay",
    "manu kone",
    "aurelien tchouameni",
    "rodrigo de paul",
    "declan rice",
    "joshua kimmich",
}

PREMIUM_FWD = {"erling haaland", "kylian mbappe", "harry kane"}
ONE_FWD_FORMATIONS = {"4-5-1", "5-4-1"}

# Holdet.dk growth rules represented in the same 100,000 DKK growth unit as
# existing goal constants in build_player_ev_group_stage.py (FWD goal = 4.0).
MATCH_WINNER_WIN_UNIT = 0.40
MATCH_WINNER_DRAW_UNIT = 0.20
HATTRICK_UNIT = 1.00
PLAYER_OF_MATCH_UNIT = 0.33
MISSED_PENALTY_UNIT = -0.30
ROUND_WEIGHTS = {1: 1.00, 2: 0.95, 3: 0.90}

CAP_LEVELS = {
    "none": None,
    "moderate": 0.75,
    "strong": 0.35,
}

VARIANTS = [
    ("baseline", False, "none"),
    ("ceiling_components_only", True, "none"),
    ("fwd_price_value_cap_moderate", False, "moderate"),
    ("fwd_price_value_cap_strong", False, "strong"),
    ("ceiling_components_plus_fwd_price_value_cap_moderate", True, "moderate"),
    ("ceiling_components_plus_fwd_price_value_cap_strong", True, "strong"),
]

PLAYER_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "position",
    "price",
    "strategy",
    "formation",
    "target_round",
    "baseline_score",
    "ceiling_components_score",
    "fwd_price_value_cap_score",
    "combined_score",
    "fwd_price_value_cap_strong_score",
    "combined_strong_score",
    "match_winner_goal_ev",
    "hattrick_ev",
    "player_of_the_match_ev",
    "penalty_ev",
    "price_value_effect_baseline",
    "price_value_effect_capped",
    "round_context_quality",
    "interpretation",
]

FORMATION_COLUMNS = [
    "variant",
    "strategy",
    "formation",
    "cap_level",
    "ceiling_components_enabled",
    "total_score",
    "total_ev",
    "total_price",
    "avg_conditional_start_prob",
    "high_risk_players",
    "fwd_count",
    "premium_fwd_count",
    "one_fwd_low_ceiling_flag",
    "player_names",
]

HAALAND_KANE_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "round",
    "opponent",
    "kickoff",
    "win_probability",
    "draw_probability",
    "goal_multiplier",
    "assist_multiplier",
    "clean_sheet_multiplier",
    "goal_share_norm",
    "assist_share_norm",
    "sot_share_norm",
    "start_prob",
    "conditional_start_prob",
    "minutes_if_start",
    "match_goal_ev",
    "match_assist_ev",
    "match_sot_ev",
    "match_total_ev_next_match",
    "round_weight",
    "round_ev",
    "fixture_mapping_status",
    "round_context_type",
    "plausibility_note",
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


def fmt(value: Any, digits: int = 6) -> str:
    return str(round(to_float(value), digits))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_draw_lookup() -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in read_csv(FIXTURE_MULTIPLIERS_PATH):
        home = txt(row.get("home")).upper()
        away = txt(row.get("away")).upper()
        draw = to_float(row.get("draw_prob_fair"))
        lookup[(home, away)] = draw
        lookup[(away, home)] = draw
    return lookup


def load_penalty_taker_ids() -> set[str]:
    ids: set[str] = set()
    for row in read_csv(SET_PIECE_MATCHED_PATH):
        if txt(row.get("role")).lower() == "penalty" and txt(row.get("matched_player_id")):
            ids.add(txt(row.get("matched_player_id")))
    return ids


def poisson_at_least_three(lam: float) -> float:
    lam = max(lam, 0.0)
    return 1.0 - math.exp(-lam) * (1.0 + lam + (lam * lam / 2.0))


def match_winner_component(row: pd.Series, rnd: int, draw_lookup: dict[tuple[str, str], float]) -> float:
    goal_lambda = to_float(row.get(f"match_{rnd}_goal_ev"))
    team = txt(row.get("team_id")).upper()
    opponent = txt(row.get(f"round{rnd}_opponent") or row.get(f"match_{rnd}_opponent_team")).upper()
    win_prob = to_float(row.get(f"round{rnd}_win_prob"))
    draw_prob = draw_lookup.get((team, opponent), 0.0)
    return goal_lambda * ((win_prob * MATCH_WINNER_WIN_UNIT) + (draw_prob * MATCH_WINNER_DRAW_UNIT))


def hattrick_component(row: pd.Series, rnd: int) -> float:
    goal_lambda = to_float(row.get(f"match_{rnd}_goal_ev"))
    return poisson_at_least_three(goal_lambda) * HATTRICK_UNIT


def ceiling_components_by_round(row: pd.Series, draw_lookup: dict[tuple[str, str]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for rnd in [1, 2, 3]:
        winner = match_winner_component(row, rnd, draw_lookup)
        hattrick = hattrick_component(row, rnd)
        out[rnd] = {
            "match_winner_goal_ev": winner * ROUND_WEIGHTS[rnd],
            "hattrick_ev": hattrick * ROUND_WEIGHTS[rnd],
            "player_of_the_match_ev": 0.0,
            "penalty_ev": 0.0,
        }
    return out


def ceiling_total(row: pd.Series, draw_lookup: dict[tuple[str, str]]) -> float:
    by_round = ceiling_components_by_round(row, draw_lookup)
    return sum(sum(values.values()) for values in by_round.values())


def price_value_effect(row: pd.Series) -> float:
    before = to_float(row.get("optimizer_ev_before_price_quality"))
    after = to_float(row.get("optimizer_ev"))
    return after - before if before else 0.0


def round_context_quality(row: pd.Series) -> str:
    if to_float(row.get("optimizer_ev")) <= 0.05 and max(to_float(row.get(f"round{rnd}_ev")) for rnd in [1, 2, 3]) <= 0.05:
        return "missing_ev_source"
    if norm(row.get("player_name")) in {"erling haaland", "harry kane"}:
        max_round = max(to_float(row.get(f"round{rnd}_ev")) for rnd in [1, 2, 3])
        if max_round < to_float(row.get("optimizer_ev")) * 0.30:
            return "weak_round_context_for_premium_fwd"
    has_fixture = any(txt(row.get(f"round{rnd}_opponent")) for rnd in [1, 2, 3])
    if has_fixture:
        return "fixture_specific_context_present"
    return "weak_or_missing_fixture_specific_ev"


def apply_experiment_variant(
    players: pd.DataFrame,
    draw_lookup: dict[tuple[str, str], float],
    include_ceiling: bool,
    cap_level: str,
) -> pd.DataFrame:
    work = players.copy()
    for col in ["optimizer_ev", "weighted_group_stage_ev", "round1_ev", "round2_ev", "round3_ev", "score_next_round", "score_round1_2", "score_group_stage", "score_long_run"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    if include_ceiling:
        additions = work.apply(lambda row: ceiling_components_by_round(row, draw_lookup), axis=1)
        for rnd in [1, 2, 3]:
            add = additions.map(lambda values: sum(values[rnd].values()))
            work[f"round{rnd}_ev"] = work[f"round{rnd}_ev"] + add
            growth_col = f"round{rnd}_captain_growth"
            if growth_col in work.columns:
                work[growth_col] = pd.to_numeric(work[growth_col], errors="coerce").fillna(0.0) + add
        total_add = additions.map(lambda values: sum(sum(round_values.values()) for round_values in values.values()))
        work["optimizer_ev"] = work["optimizer_ev"] + total_add
        work["weighted_group_stage_ev"] = work["weighted_group_stage_ev"] + total_add

    cap = CAP_LEVELS[cap_level]
    if cap is not None:
        before = pd.to_numeric(work.get("optimizer_ev_before_price_quality"), errors="coerce").fillna(work["optimizer_ev"])
        effect = work["optimizer_ev"] - before
        capped_effect = effect.clip(upper=cap)
        fwd_mask = work["position"].astype(str).str.upper().eq("FWD") & (effect > cap)
        work.loc[fwd_mask, "optimizer_ev"] = before.loc[fwd_mask] + capped_effect.loc[fwd_mask]
        work.loc[fwd_mask, "weighted_group_stage_ev"] = work.loc[fwd_mask, "optimizer_ev"]

    work = optimizer.add_strategy_scores(work)
    numeric_cols = [
        "optimizer_ev",
        "weighted_group_stage_ev",
        "score_next_round",
        "score_round1_2",
        "score_group_stage",
        "score_long_run",
        "round1_ev",
        "round2_ev",
        "round3_ev",
        "conditional_start_prob",
        "price_m",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").replace([math.inf, -math.inf], 0.0).fillna(0.0)
    return work


def build_variant_players(players: pd.DataFrame, draw_lookup: dict[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for variant, include_ceiling, cap_level in VARIANTS:
        variants[variant] = apply_experiment_variant(players, draw_lookup, include_ceiling, cap_level)
    return variants


def solve_variant_formations(variant_players: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, include_ceiling, cap_level in VARIANTS:
        players = variant_players[variant]
        for strategy in optimizer.STRATEGIES:
            for formation_name, formation in optimizer.FORMATIONS.items():
                squad = optimizer.solve_formation(players, strategy, formation_name, formation)
                if squad.empty:
                    rows.append({
                        "variant": variant,
                        "strategy": strategy,
                        "formation": formation_name,
                        "cap_level": cap_level,
                        "ceiling_components_enabled": str(include_ceiling),
                        "total_score": "0",
                        "total_ev": "0",
                        "total_price": "0",
                        "avg_conditional_start_prob": "0",
                        "high_risk_players": "0",
                        "fwd_count": "0",
                        "premium_fwd_count": "0",
                        "one_fwd_low_ceiling_flag": "no_valid_solution",
                        "player_names": "",
                    })
                    continue
                fwd_count = int((squad["position"] == "FWD").sum())
                premium_count = int(squad["player_name"].map(norm).isin(PREMIUM_FWD).sum())
                one_fwd_low = formation_name in ONE_FWD_FORMATIONS and premium_count == 0
                rows.append({
                    "variant": variant,
                    "strategy": strategy,
                    "formation": formation_name,
                    "cap_level": cap_level,
                    "ceiling_components_enabled": str(include_ceiling),
                    "total_score": fmt(squad["strategy_score"].sum()),
                    "total_ev": fmt(squad["optimizer_ev"].sum()),
                    "total_price": str(int(round(float(squad["price_m"].sum()) * 1_000_000))),
                    "avg_conditional_start_prob": fmt(squad["conditional_start_prob"].mean(), 4),
                    "high_risk_players": str(int((squad["availability_risk"] == "high_risk").sum())),
                    "fwd_count": str(fwd_count),
                    "premium_fwd_count": str(premium_count),
                    "one_fwd_low_ceiling_flag": "yes" if one_fwd_low else "no",
                    "player_names": "; ".join(squad["player_name"].astype(str).tolist()),
                })
    return rows


def score_for(variant_players: dict[str, pd.DataFrame], variant: str, player_id: str, strategy: str) -> float:
    df = variant_players[variant]
    row = df[df["player_id"].astype(str) == player_id]
    if row.empty:
        return 0.0
    return to_float(row.iloc[0].get(f"score_{strategy}"))


def build_player_comparison_rows(
    players: pd.DataFrame,
    variant_players: dict[str, pd.DataFrame],
    draw_lookup: dict[tuple[str, str], float],
    penalty_taker_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = set(players.loc[players["player_name"].map(norm).isin(TARGET_NAMES), "player_id"].astype(str))
    target_round = int(optimizer.get_current_target_round().get("target_round") or 1)
    rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    for _, player in players.iterrows():
        player_id = txt(player.get("player_id"))
        if player_id not in target_ids:
            continue
        by_round = ceiling_components_by_round(player, draw_lookup)
        total_match_winner = sum(values["match_winner_goal_ev"] for values in by_round.values())
        total_hattrick = sum(values["hattrick_ev"] for values in by_round.values())
        has_penalty_signal = player_id in penalty_taker_ids or "penalty" in txt(player.get("manual_set_piece_role")).lower()
        penalty_ev = "" if has_penalty_signal else ""
        penalty_note = "penalty taker signal exists, but no attempt/miss probability exists" if has_penalty_signal else "missing penalty taker probability"
        price_effect = price_value_effect(player)
        capped_price_effect = min(price_effect, CAP_LEVELS["moderate"]) if txt(player.get("position")) == "FWD" and price_effect > CAP_LEVELS["moderate"] else price_effect
        for strategy in optimizer.STRATEGIES:
            for formation in optimizer.FORMATIONS:
                base = score_for(variant_players, "baseline", player_id, strategy)
                ceiling = score_for(variant_players, "ceiling_components_only", player_id, strategy)
                cap_mod = score_for(variant_players, "fwd_price_value_cap_moderate", player_id, strategy)
                combined = score_for(variant_players, "ceiling_components_plus_fwd_price_value_cap_moderate", player_id, strategy)
                cap_strong = score_for(variant_players, "fwd_price_value_cap_strong", player_id, strategy)
                combined_strong = score_for(variant_players, "ceiling_components_plus_fwd_price_value_cap_strong", player_id, strategy)
                interpretation_bits = []
                if norm(player.get("player_name")) in PREMIUM_FWD and round_context_quality(player) == "weak_round_context_for_premium_fwd":
                    interpretation_bits.append("weak round context is a primary drag")
                if norm(player.get("player_name")) in TARGET_NAMES and txt(player.get("position")) in {"MID", "FWD"}:
                    interpretation_bits.append("ceiling adds only match-winner/hattrick because POTM/penalty are not safely calculable")
                if txt(player.get("position")) == "FWD" and price_effect > CAP_LEVELS["moderate"]:
                    interpretation_bits.append("moderate FWD price/value cap reduces score")
                if norm(player.get("player_name")) in {"konrad laimer", "scott mctominay"}:
                    interpretation_bits.append("compare against FWD ceiling before production weight changes")
                if not interpretation_bits:
                    interpretation_bits.append("baseline control row")
                row = {
                    "player_id": player_id,
                    "player_name": txt(player.get("player_name")),
                    "team": txt(player.get("team_id")),
                    "position": txt(player.get("position")),
                    "price": str(int(to_float(player.get("price")))),
                    "strategy": strategy,
                    "formation": formation,
                    "target_round": str(target_round),
                    "baseline_score": fmt(base),
                    "ceiling_components_score": fmt(ceiling),
                    "fwd_price_value_cap_score": fmt(cap_mod),
                    "combined_score": fmt(combined),
                    "fwd_price_value_cap_strong_score": fmt(cap_strong),
                    "combined_strong_score": fmt(combined_strong),
                    "match_winner_goal_ev": fmt(total_match_winner),
                    "hattrick_ev": fmt(total_hattrick),
                    "player_of_the_match_ev": "",
                    "penalty_ev": penalty_ev,
                    "price_value_effect_baseline": fmt(price_effect),
                    "price_value_effect_capped": fmt(capped_price_effect),
                    "round_context_quality": round_context_quality(player),
                    "interpretation": "; ".join(interpretation_bits) + f"; {penalty_note}",
                }
                rows.append(row)
                experiment_rows.append(row)
    return rows, experiment_rows


def build_haaland_kane_rows(players: pd.DataFrame, draw_lookup: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, player in players[players["player_name"].map(norm).isin({"erling haaland", "harry kane"})].iterrows():
        for rnd in [1, 2, 3]:
            team = txt(player.get("team_id")).upper()
            opponent = txt(player.get(f"round{rnd}_opponent") or player.get(f"match_{rnd}_opponent_team")).upper()
            draw = draw_lookup.get((team, opponent), 0.0)
            mapping = "ok" if opponent and to_float(player.get(f"round{rnd}_win_prob")) > 0 else "missing_or_weak"
            round_ev = to_float(player.get(f"round{rnd}_ev"))
            total_next = to_float(player.get(f"match_{rnd}_total_ev_next_match"))
            context_type = "fixture_specific" if txt(player.get(f"match_{rnd}_opponent_team")) else "missing_fixture_export"
            if norm(player.get("player_name")) == "erling haaland" and rnd == 1:
                note = (
                    "NOR-IRQ mapping is present and fixture is favorable, but match_goal_ev=0.047 is very low for a premium FWD; "
                    "low value is not explained by fixture mapping."
                )
            elif norm(player.get("player_name")) == "harry kane":
                note = "Round context is present; values are moderate, not missing, but may understate premium ceiling."
            else:
                note = "Round context present."
            rows.append({
                "player_id": txt(player.get("player_id")),
                "player_name": txt(player.get("player_name")),
                "team": team,
                "round": str(rnd),
                "opponent": opponent,
                "kickoff": txt(player.get(f"match_{rnd}_kickoff")),
                "win_probability": fmt(player.get(f"round{rnd}_win_prob"), 4),
                "draw_probability": fmt(draw, 4),
                "goal_multiplier": fmt(player.get(f"round{rnd}_goal_multiplier"), 4),
                "assist_multiplier": fmt(player.get(f"round{rnd}_assist_multiplier"), 4),
                "clean_sheet_multiplier": fmt(player.get(f"round{rnd}_clean_sheet_multiplier"), 4),
                "goal_share_norm": fmt(player.get("goal_share_norm"), 4),
                "assist_share_norm": fmt(player.get("assist_share_norm"), 4),
                "sot_share_norm": fmt(player.get("sot_share_norm"), 4),
                "start_prob": fmt(player.get("start_prob"), 4),
                "conditional_start_prob": fmt(player.get("conditional_start_prob"), 4),
                "minutes_if_start": fmt(player.get(f"match_{rnd}_minutes_if_start"), 3),
                "match_goal_ev": fmt(player.get(f"match_{rnd}_goal_ev"), 6),
                "match_assist_ev": fmt(player.get(f"match_{rnd}_assist_ev"), 6),
                "match_sot_ev": fmt(player.get(f"match_{rnd}_shots_on_target_ev"), 6),
                "match_total_ev_next_match": fmt(total_next, 6),
                "round_weight": fmt(ROUND_WEIGHTS[rnd], 2),
                "round_ev": fmt(round_ev, 6),
                "fixture_mapping_status": mapping,
                "round_context_type": context_type,
                "plausibility_note": note,
            })
    return rows


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> list[str]:
    if not rows:
        return ["Ingen rækker."]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(txt(row.get(col)).replace("|", "/") for col in columns) + " |")
    return out


def write_haaland_kane_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Haaland/Kane Round Context Audit",
        "",
        "Ren audit af eksisterende runde- og fixturefelter. Ingen produktionsoutput er skrevet.",
        "",
        "## Fokus",
        "",
        *md_table(rows, ["player_name", "round", "opponent", "win_probability", "goal_multiplier", "goal_share_norm", "start_prob", "minutes_if_start", "match_goal_ev", "round_ev", "fixture_mapping_status", "plausibility_note"], 10),
        "",
        "## Konklusion",
        "",
        "- Haaland er korrekt mappet som NOR mod IRQ i runde 1 med win probability 0,7579 og goal multiplier 1,35.",
        "- Hans lave next-round value skyldes derfor ikke åbenlys team/fixture-mappingfejl. Den skyldes især meget lav `match_1_goal_ev` på 0,047385 og lav samlet `match_1_total_ev_next_match` på 0,237368.",
        "- Det virker svagt for Haaland mod Irak ud fra de tilgængelige outputfelter, men det bør testes i EV/round-context-modellen, ikke rettes manuelt her.",
        "- Kane har reel fixture-specifik kontekst og moderate rundeværdier; hans problem er mere ceiling/round-weighting end missing data.",
    ]
    OUT_HAALAND_KANE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_md(player_rows: list[dict[str, Any]], formation_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(player_rows)
    form = pd.DataFrame(formation_rows)

    def avg_delta(name: str, variant_col: str) -> float:
        sub = df[df["player_name"].map(norm).eq(norm(name))]
        if sub.empty:
            return 0.0
        return float((pd.to_numeric(sub[variant_col], errors="coerce") - pd.to_numeric(sub["baseline_score"], errors="coerce")).mean())

    one_fwd = form[form["formation"].isin(ONE_FWD_FORMATIONS)].copy()
    low_flags = one_fwd.groupby("variant")["one_fwd_low_ceiling_flag"].apply(lambda s: int((s == "yes").sum())).to_dict() if not one_fwd.empty else {}
    variant_summary = form.groupby("variant").agg(
        avg_score=("total_score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        low_one_fwd=("one_fwd_low_ceiling_flag", lambda s: int((s == "yes").sum())),
    ).reset_index().to_dict("records") if not form.empty else []

    focus_rows = []
    for name in ["Erling Haaland", "Kylian Mbappe", "Harry Kane", "Luis Diaz", "Michael Olise", "Jamal Musiala", "Konrad Laimer", "Scott McTominay"]:
        sub = df[(df["player_name"] == name) & (df["strategy"] == "next_round") & (df["formation"] == "4-5-1")]
        if not sub.empty:
            focus_rows.append(sub.iloc[0].to_dict())

    lines = [
        "# Offensive Ceiling Model Experiment",
        "",
        "Kontrolleret eksperiment baseret på eksisterende modeloutput. Produktionsfiler for optimizer, strategi-output, EV, player pool og frontend er ikke overskrevet.",
        "",
        "## Variantformler",
        "",
        "- `baseline`: uændret `optimizer.load_players()` og eksisterende scoreformler.",
        "- `ceiling_components_only`: baseline plus beregnelige matchwinner- og hattrick-EV pr. runde. Matchwinner bruger eksisterende `match_n_goal_ev`, win/draw probability og Holdet.dk-reglerne 40.000/20.000. Hattrick bruger Poisson ud fra `match_n_goal_ev` og 100.000-reglen.",
        "- `fwd_price_value_cap_moderate`: FWD price/value-effekt capped ved 0,75 model-growth units.",
        "- `fwd_price_value_cap_strong`: FWD price/value-effekt capped ved 0,35 model-growth units.",
        "- `combined`: ceiling-komponenter plus samme FWD price/value-cap.",
        "- `player_of_the_match_ev` og `penalty_ev` er ikke beregnet, fordi outputtet mangler forsvarlig POTM-sandsynlighed og penalty attempt/miss probability.",
        "",
        "## Variantoversigt",
        "",
        "| variant | avg_total_score | one_fwd_low_ceiling_flags |",
        "|---|---:|---:|",
    ]
    for row in variant_summary:
        lines.append(f"| {row['variant']} | {float(row['avg_score']):.3f} | {int(row['low_one_fwd'])} |")

    lines += [
        "",
        "## Fokus: next_round 4-5-1",
        "",
        *md_table(focus_rows, ["player_name", "baseline_score", "ceiling_components_score", "fwd_price_value_cap_score", "combined_score", "match_winner_goal_ev", "hattrick_ev", "price_value_effect_baseline", "round_context_quality"], 20),
        "",
        "## Spillerkonklusioner",
        "",
        f"- Haaland: ceiling-only gennemsnitlig scoreændring {avg_delta('Erling Haaland', 'ceiling_components_score'):.3f}; combined-moderate {avg_delta('Erling Haaland', 'combined_score'):.3f}. Round context er stadig hovedproblemet.",
        f"- Kane: ceiling-only ændring {avg_delta('Harry Kane', 'ceiling_components_score'):.3f}; combined-moderate {avg_delta('Harry Kane', 'combined_score'):.3f}. Han forbedres lidt, men er ikke primært et price/value-problem.",
        f"- Diaz: ceiling-only ændring {avg_delta('Luis Diaz', 'ceiling_components_score'):.3f}; mangler stadig POTM/penalty/multi-ceiling output.",
        f"- Olise: ceiling-only ændring {avg_delta('Michael Olise', 'ceiling_components_score'):.3f}; ser fortsat som mulig ceiling-undervægtning.",
        f"- Musiala: ceiling-only ændring {avg_delta('Jamal Musiala', 'ceiling_components_score'):.3f}; påvirkes lidt, men MID-positionen gør FWD-cap irrelevant.",
        f"- Laimer: FWD-cap påvirker ham ikke direkte; hvis han stadig slår premium FWD, skyldes det round context/start/strategy score.",
        f"- McTominay: FWD-cap påvirker ham ikke direkte; hans høje next_round-score er drevet af round context, ikke price/value.",
        "",
        "## Formation-aware floor audit",
        "",
        f"- One-FWD low-ceiling flags efter baseline: {low_flags.get('baseline', 0)}.",
        f"- Efter combined moderate: {low_flags.get('ceiling_components_plus_fwd_price_value_cap_moderate', 0)}.",
        f"- Efter combined strong: {low_flags.get('ceiling_components_plus_fwd_price_value_cap_strong', 0)}.",
        "- Hvis 4-5-1/5-4-1 stadig vælger lav offensiv upside, peger auditten først på round EV/absolute EV og price/value-lag. Et hard floor kan skjule problemet og bør fortsat undgås, indtil EV-komponenterne er bedre.",
        "",
        "## Anbefalet næste produktionsvariant",
        "",
        "Test først `ceiling_components_plus_fwd_price_value_cap_moderate` i en separat produktionspipeline-run. Den er mest kontrolleret: den tilføjer kun beregnelige Holdet.dk-ceiling-komponenter og reducerer kun ekstrem FWD price/value-effekt, uden at hardcode premiumangribere eller formation floors.",
    ]
    OUT_EXPERIMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_squad_md(formation_rows: list[dict[str, Any]]) -> None:
    focus = [
        row for row in formation_rows
        if row["formation"] in ONE_FWD_FORMATIONS
        and row["variant"] in {"baseline", "ceiling_components_only", "fwd_price_value_cap_moderate", "ceiling_components_plus_fwd_price_value_cap_moderate"}
    ]
    lines = [
        "# Model Experiment Squad Comparison",
        "",
        "Eksperimentelle squads løst i memory med samme constraints som optimizer. Produktionsfiler er ikke overskrevet.",
        "",
        "## One-FWD formationer",
        "",
        *md_table(focus, ["variant", "strategy", "formation", "total_score", "total_ev", "total_price", "premium_fwd_count", "one_fwd_low_ceiling_flag", "player_names"], 80),
    ]
    OUT_SQUAD_COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    players = optimizer.load_players()
    draw_lookup = load_draw_lookup()
    penalty_taker_ids = load_penalty_taker_ids()
    variant_players = build_variant_players(players, draw_lookup)

    player_rows, experiment_rows = build_player_comparison_rows(players, variant_players, draw_lookup, penalty_taker_ids)
    formation_rows = solve_variant_formations(variant_players)
    haaland_kane_rows = build_haaland_kane_rows(players, draw_lookup)

    write_csv(OUT_HAALAND_KANE_CSV, HAALAND_KANE_COLUMNS, haaland_kane_rows)
    write_haaland_kane_md(haaland_kane_rows)
    write_csv(OUT_PLAYER_COMPARISON_CSV, PLAYER_COLUMNS, player_rows)
    write_csv(OUT_EXPERIMENT_CSV, PLAYER_COLUMNS, experiment_rows)
    write_csv(OUT_FORMATION_COMPARISON_CSV, FORMATION_COLUMNS, formation_rows)
    write_experiment_md(player_rows, formation_rows)
    write_squad_md(formation_rows)

    print("Offensive ceiling model experiment")
    print("----------------------------------")
    print(f"Player comparison rows: {len(player_rows)}")
    print(f"Formation comparison rows: {len(formation_rows)}")
    print(f"Haaland/Kane round rows: {len(haaland_kane_rows)}")
    for path in [
        OUT_HAALAND_KANE_CSV,
        OUT_HAALAND_KANE_MD,
        OUT_EXPERIMENT_CSV,
        OUT_EXPERIMENT_MD,
        OUT_PLAYER_COMPARISON_CSV,
        OUT_FORMATION_COMPARISON_CSV,
        OUT_SQUAD_COMPARISON_MD,
    ]:
        print(f"Wrote: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
