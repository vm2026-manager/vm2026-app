from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EV_PATH = DATA / "player_ev_group_stage_v1.csv"
ROOT_CAUSE_PATH = DATA / "no_ev_source_root_cause_audit.csv"
START_SIGNAL_PATH = DATA / "player_start_signal_layer_v1.csv"
MATCH_ODDS_PATH = DATA / "match_odds_probs.csv"
OUT_CSV = DATA / "missing_player_component_source_experiment.csv"
OUT_MD = DATA / "missing_player_component_source_experiment.md"

ROUND_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.90}
GOAL_POINTS = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
ASSIST_POINTS = 3.0
SHOT_ON_TARGET_POINTS = 1.0
CLEAN_SHEET_POINTS = {"GK": 2.8, "DEF": 2.2, "MID": 0.0, "FWD": 0.0}
YELLOW_CARD_POINTS = -1.0
ON_PITCH_POINTS = 7.0
NOT_ON_PITCH_POINTS = -5.0
WIN_POINTS = 25.0
DRAW_POINTS = 5.0
LOSS_POINTS = -8.0
TEAM_SCORES_POINTS = 10.0
OPPONENT_SCORES_POINTS = -8.0
POINT_SCALE = 100.0
TEAM_SHARE_CAP = 0.90

SHARE_COLS = {
    "goal": "goal_share_norm",
    "assist": "assist_share_norm",
    "shots": "sot_share_norm",
}
COMPONENT_COLS = {
    "goal": "goal_ev",
    "assist": "assist_ev",
    "shots": "shots_on_target_ev",
}


def number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def start_band(value: float) -> str:
    if value < 0.25:
        return "reserve"
    if value < 0.50:
        return "rotation"
    if value < 0.70:
        return "contender"
    if value < 0.85:
        return "likely_starter"
    return "strong_starter"


def robust_median(values: list[float], default: float = 0.0) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else default


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(EV_PATH, low_memory=False)
    roots = pd.read_csv(ROOT_CAUSE_PATH)
    signals = pd.read_csv(START_SIGNAL_PATH, low_memory=False)
    odds = pd.read_csv(MATCH_ODDS_PATH, low_memory=False)
    return ev, roots, signals, odds


def complete_reference_population(ev: pd.DataFrame) -> pd.DataFrame:
    has_share = pd.Series(False, index=ev.index)
    for column in SHARE_COLS.values():
        has_share |= pd.to_numeric(ev[column], errors="coerce").fillna(0.0).gt(0)
    has_components = ev["component_source"].fillna("").eq("complete_components")
    has_match_ev = sum(
        pd.to_numeric(ev[f"match_{match_no}_weighted_match_ev"], errors="coerce").fillna(0.0)
        for match_no in (1, 2, 3)
    ).abs().gt(0)
    return ev.loc[has_share & has_components & has_match_ev].copy()


def signal_lookup(signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["player_id"]): row.to_dict()
        for _, row in signals.drop_duplicates("player_id", keep="last").iterrows()
    }


def appearance_probability(row: pd.Series, signals: dict[str, dict[str, Any]]) -> float:
    start = clamp(number(row.get("start_prob")))
    signal = signals.get(str(row.get("player_id")), {})
    explicit = number(signal.get("appearance_prob"), -1.0)
    if explicit > 0:
        return clamp(max(start, explicit))

    conditional = number(signal.get("conditional_start_prob"), -1.0)
    availability = number(signal.get("availability_prob"), -1.0)
    if conditional >= 0 and availability > 0:
        bench_appearance = 0.18 * (1.0 - clamp(conditional))
        return clamp(max(start, availability * (clamp(conditional) + bench_appearance)))

    return clamp(start + 0.18 * (1.0 - start))


def minutes_if_start(row: pd.Series, match_no: int, references: pd.DataFrame) -> float:
    current = number(row.get(f"match_{match_no}_minutes_if_start"))
    if current > 0:
        return current
    same_position = references.loc[references["position"].eq(row.get("position"))]
    values = pd.to_numeric(
        same_position[f"match_{match_no}_minutes_if_start"], errors="coerce"
    ).dropna()
    return float(values.median()) if not values.empty else 70.0


def odds_lookup(odds: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for _, row in odds.iterrows():
        home = str(row.get("home") or "").upper()
        away = str(row.get("away") or "").upper()
        kickoff = str(row.get("kickoff_dk") or "")
        home_win = number(row.get("home_win_prob_fair"))
        draw = number(row.get("draw_prob_fair"))
        away_win = number(row.get("away_win_prob_fair"))
        lookup[(home, away, kickoff)] = {"win": home_win, "draw": draw, "loss": away_win}
        lookup[(away, home, kickoff)] = {"win": away_win, "draw": draw, "loss": home_win}
    return lookup


def estimate_shares(
    row: pd.Series,
    references: pd.DataFrame,
    appearances: dict[str, float],
    *,
    exclude_player_id: str = "",
    allow_documented: bool = True,
) -> tuple[dict[str, float], int, str, str, list[str]]:
    target_id = str(row.get("player_id"))
    candidates = references.loc[references["player_id"].astype(str).ne(exclude_player_id)].copy()
    target_start = clamp(number(row.get("start_prob")))
    target_appearance = appearances.get(target_id, target_start)
    position = str(row.get("position") or "")
    team = str(row.get("team_id") or "")
    band = start_band(target_start)

    team_position = candidates.loc[
        candidates["team_id"].astype(str).eq(team)
        & candidates["position"].astype(str).eq(position)
    ]
    position_band = candidates.loc[
        candidates["position"].astype(str).eq(position)
        & candidates["start_prob"].map(lambda value: start_band(number(value))).eq(band)
    ]
    position_all = candidates.loc[candidates["position"].astype(str).eq(position)]
    fallback = position_band if len(position_band) >= 8 else position_all

    warnings: list[str] = []
    estimates: dict[str, float] = {}
    documented_used = False

    for component, column in SHARE_COLS.items():
        documented = number(row.get(column))
        if allow_documented and documented > 0:
            estimates[component] = documented
            documented_used = True
            continue

        local_values = pd.to_numeric(team_position[column], errors="coerce").dropna()
        fallback_values = pd.to_numeric(fallback[column], errors="coerce").dropna()
        local_median = float(local_values.median()) if not local_values.empty else np.nan
        fallback_median = float(fallback_values.median()) if not fallback_values.empty else 0.0

        if np.isfinite(local_median):
            local_weight = min(0.75, 0.35 + 0.10 * len(local_values))
            base = local_weight * local_median + (1.0 - local_weight) * fallback_median
        else:
            base = fallback_median

        reference_appearances = [
            appearances.get(str(player_id), number(start))
            for player_id, start in zip(fallback["player_id"], fallback["start_prob"])
        ]
        median_appearance = robust_median(reference_appearances, target_appearance or 0.5)
        usage_scale = clamp(
            np.sqrt(max(target_appearance, 0.05) / max(median_appearance, 0.05)),
            0.65,
            1.35,
        )
        estimates[component] = max(0.0, base * usage_scale)

    if documented_used:
        method = "documented_shares_with_reference_component_rates"
    elif len(team_position) >= 3:
        method = "shrunk_team_position_median_plus_position_start_band"
    elif len(team_position) >= 1:
        method = "limited_team_position_blend_plus_position_start_band"
        warnings.append("small_team_position_reference")
    else:
        method = "position_start_band_fallback"
        warnings.append("no_team_position_reference")

    if documented_used and len(team_position) < 1:
        warnings.append("documented_share_without_team_position_reference")

    if documented_used or len(team_position) >= 3:
        confidence = "high"
    elif len(team_position) >= 1 and len(fallback) >= 12:
        confidence = "medium"
    else:
        confidence = "low"

    return estimates, len(team_position), method, confidence, warnings


def apply_team_share_caps(
    proposals: list[dict[str, Any]],
    references: pd.DataFrame,
) -> None:
    for team, team_rows in pd.DataFrame(proposals).groupby("team_id"):
        reference_team = references.loc[references["team_id"].astype(str).eq(str(team))]
        for component, column in SHARE_COLS.items():
            existing = pd.to_numeric(reference_team[column], errors="coerce").fillna(0.0).sum()
            proposed = sum(row["shares"][component] for row in proposals if row["team_id"] == team)
            available = max(0.0, TEAM_SHARE_CAP - existing)
            scale = min(1.0, available / proposed) if proposed > 0 else 1.0
            if scale < 1.0:
                for row in proposals:
                    if row["team_id"] != team:
                        continue
                    row["shares"][component] *= scale
                    row["warnings"].append(f"{component}_team_share_cap_applied")
                    row["team_share_scale"] = min(row.get("team_share_scale", 1.0), scale)


def context_baseline(
    row: pd.Series,
    match_no: int,
    references: pd.DataFrame,
    appearances: dict[str, float],
    odds: dict[tuple[str, str, str], dict[str, float]],
    exclude_player_id: str,
) -> dict[str, float]:
    team = str(row.get("team_id") or "")
    opponent = str(row.get(f"match_{match_no}_opponent_team") or "")
    kickoff = str(row.get(f"match_{match_no}_kickoff") or "")
    peers = references.loc[
        references["team_id"].astype(str).eq(team)
        & references["player_id"].astype(str).ne(exclude_player_id)
    ]

    result_bases: list[float] = []
    team_score_probs: list[float] = []
    opponent_score_probs: list[float] = []
    for _, peer in peers.iterrows():
        app = max(appearances.get(str(peer.get("player_id")), number(peer.get("start_prob"))), 0.01)
        result_ev = number(peer.get(f"match_{match_no}_result_ev"))
        team_scores = number(peer.get(f"match_{match_no}_team_scores_ev"))
        opponent_scores = number(peer.get(f"match_{match_no}_opponent_scores_ev"))
        if result_ev:
            result_bases.append(result_ev / app)
        if team_scores > 0:
            team_score_probs.append(clamp(team_scores * POINT_SCALE / (TEAM_SCORES_POINTS * app)))
        if opponent_scores < 0:
            opponent_score_probs.append(
                clamp(abs(opponent_scores) * POINT_SCALE / (abs(OPPONENT_SCORES_POINTS) * app))
            )

    match_odds = odds.get((team, opponent, kickoff))
    odds_result = 0.0
    if match_odds:
        odds_result = (
            match_odds["win"] * WIN_POINTS
            + match_odds["draw"] * DRAW_POINTS
            + match_odds["loss"] * LOSS_POINTS
        ) / POINT_SCALE

    return {
        "result_base": odds_result or robust_median(result_bases),
        "team_scores_prob": robust_median(team_score_probs),
        "opponent_scores_prob": robust_median(opponent_score_probs),
    }


def offensive_rate(
    row: pd.Series,
    match_no: int,
    component: str,
    references: pd.DataFrame,
    appearances: dict[str, float],
    exclude_player_id: str,
) -> tuple[float, int]:
    share_column = SHARE_COLS[component]
    component_column = COMPONENT_COLS[component]
    team = str(row.get("team_id") or "")
    position = str(row.get("position") or "")
    candidates = references.loc[references["player_id"].astype(str).ne(exclude_player_id)].copy()

    def rates(frame: pd.DataFrame) -> list[float]:
        output = []
        for _, peer in frame.iterrows():
            share = number(peer.get(share_column))
            app = appearances.get(str(peer.get("player_id")), number(peer.get("start_prob")))
            value = number(peer.get(f"match_{match_no}_{component_column}"))
            denominator = share * max(app, 0.05)
            if denominator > 0 and value >= 0:
                output.append(value / denominator)
        return output

    same_team = candidates.loc[candidates["team_id"].astype(str).eq(team)]
    team_rates = rates(same_team)
    if len(team_rates) >= 4:
        return robust_median(team_rates), len(team_rates)

    same_position = candidates.loc[candidates["position"].astype(str).eq(position)]
    position_rates = rates(same_position)
    if team_rates:
        blended = 0.45 * robust_median(team_rates) + 0.55 * robust_median(position_rates)
        return blended, len(team_rates)
    return robust_median(position_rates), 0


def card_component(
    row: pd.Series,
    match_no: int,
    references: pd.DataFrame,
    appearances: dict[str, float],
    exclude_player_id: str,
) -> float:
    peers = references.loc[
        references["position"].astype(str).eq(str(row.get("position") or ""))
        & references["player_id"].astype(str).ne(exclude_player_id)
    ]
    rates = []
    for _, peer in peers.iterrows():
        app = max(appearances.get(str(peer.get("player_id")), number(peer.get("start_prob"))), 0.05)
        rates.append(number(peer.get(f"match_{match_no}_card_ev")) / app)
    return robust_median(rates) * appearances.get(str(row.get("player_id")), number(row.get("start_prob")))


def estimate_components(
    row: pd.Series,
    shares: dict[str, float],
    references: pd.DataFrame,
    appearances: dict[str, float],
    odds: dict[tuple[str, str, str], dict[str, float]],
    *,
    exclude_player_id: str = "",
) -> tuple[dict[str, float], list[str]]:
    position = str(row.get("position") or "")
    start = clamp(number(row.get("start_prob")))
    appearance = appearances.get(str(row.get("player_id")), start)
    output: dict[str, float] = {}
    warnings: list[str] = []

    for match_no in (1, 2, 3):
        minutes = minutes_if_start(row, match_no, references)
        context = context_baseline(
            row, match_no, references, appearances, odds, exclude_player_id
        )

        for component in ("goal", "assist", "shots"):
            rate, team_rate_size = offensive_rate(
                row, match_no, component, references, appearances, exclude_player_id
            )
            output[f"match_{match_no}_{COMPONENT_COLS[component]}"] = (
                shares[component] * appearance * rate
            )
            if team_rate_size == 0:
                warnings.append(f"match_{match_no}_{component}_position_rate_fallback")

        clean_prob = number(row.get(f"match_{match_no}_clean_sheet_prob"), -1.0)
        clean_eligibility = start * clamp(minutes / 60.0) if minutes > 0 else start
        clean_sheet = (
            clean_prob * clean_eligibility
            if position in {"GK", "DEF"} and clean_prob >= 0
            else 0.0
        )
        result = context["result_base"] * appearance
        team_scores = (
            context["team_scores_prob"] * TEAM_SCORES_POINTS * appearance / POINT_SCALE
        )
        opponent_scores = (
            context["opponent_scores_prob"]
            * OPPONENT_SCORES_POINTS
            * appearance
            / POINT_SCALE
        )
        on_pitch = (
            appearance * ON_PITCH_POINTS + (1.0 - appearance) * NOT_ON_PITCH_POINTS
        ) / POINT_SCALE
        card = card_component(
            row, match_no, references, appearances, exclude_player_id
        )

        output[f"match_{match_no}_clean_sheet_ev"] = clean_sheet
        output[f"match_{match_no}_card_ev"] = card
        output[f"match_{match_no}_result_ev"] = result
        output[f"match_{match_no}_team_scores_ev"] = team_scores
        output[f"match_{match_no}_opponent_scores_ev"] = opponent_scores
        output[f"match_{match_no}_on_pitch_ev"] = on_pitch

        total = (
            output[f"match_{match_no}_goal_ev"] * GOAL_POINTS.get(position, 5.0)
            + output[f"match_{match_no}_assist_ev"] * ASSIST_POINTS
            + output[f"match_{match_no}_shots_on_target_ev"] * SHOT_ON_TARGET_POINTS
            + clean_sheet * CLEAN_SHEET_POINTS.get(position, 0.0)
            + card * YELLOW_CARD_POINTS
            + result
            + team_scores
            + opponent_scores
            + on_pitch
        )
        output[f"match_{match_no}_total_ev_next_match"] = total
        output[f"match_{match_no}_weighted_match_ev"] = total * ROUND_WEIGHTS[match_no]

    output["estimated_base_ev"] = sum(
        output[f"match_{match_no}_weighted_match_ev"] for match_no in (1, 2, 3)
    )
    return output, warnings


def validation_experiment(
    references: pd.DataFrame,
    appearances: dict[str, float],
    odds: dict[tuple[str, str, str], dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for _, player in references.iterrows():
        player_id = str(player.get("player_id"))
        shares, size, method, confidence, warnings = estimate_shares(
            player,
            references,
            appearances,
            exclude_player_id=player_id,
            allow_documented=False,
        )

        # A leave-one-out cap prevents the hidden player's estimate from exceeding
        # the remaining team share while leaving all other documented players intact.
        team_peers = references.loc[
            references["team_id"].astype(str).eq(str(player.get("team_id")))
            & references["player_id"].astype(str).ne(player_id)
        ]
        for component, column in SHARE_COLS.items():
            existing = pd.to_numeric(team_peers[column], errors="coerce").fillna(0.0).sum()
            available = max(0.0, TEAM_SHARE_CAP - existing)
            if shares[component] > available:
                shares[component] = available
                warnings.append(f"{component}_team_share_cap_applied")

        estimated, component_warnings = estimate_components(
            player,
            shares,
            references,
            appearances,
            odds,
            exclude_player_id=player_id,
        )
        actual = number(player.get("weighted_group_stage_ev_before_price_quality"))
        if actual <= 0:
            actual = sum(
                number(player.get(f"match_{match_no}_weighted_match_ev"))
                for match_no in (1, 2, 3)
            )
        error = estimated["estimated_base_ev"] - actual
        relative_error = abs(error) / abs(actual) if abs(actual) > 1e-9 else np.nan
        rows.append(
            {
                "player_id": player_id,
                "player_name": player.get("player_name"),
                "team_id": player.get("team_id"),
                "position": player.get("position"),
                "start_prob": number(player.get("start_prob")),
                "start_band": start_band(number(player.get("start_prob"))),
                "reference_population_size": size,
                "reference_method": method,
                "confidence": confidence,
                "actual_base_ev": actual,
                "estimated_base_ev": estimated["estimated_base_ev"],
                "signed_error": error,
                "absolute_error": abs(error),
                "relative_error": relative_error,
                "warning_flags": ";".join(sorted(set(warnings + component_warnings))),
            }
        )
    return pd.DataFrame(rows)


def target_experiment(
    ev: pd.DataFrame,
    roots: pd.DataFrame,
    references: pd.DataFrame,
    appearances: dict[str, float],
    odds: dict[tuple[str, str, str], dict[str, float]],
) -> pd.DataFrame:
    root_columns = [
        "player_id",
        "root_cause",
        "fantasy_relevance_score",
        "historical_recovery_status",
    ]
    targets = roots[root_columns].merge(ev, on="player_id", how="left")
    proposals: list[dict[str, Any]] = []

    for _, row in targets.iterrows():
        identity_blocked = row["root_cause"] == "missing_from_ev_master_at_holdet_rebase"
        if identity_blocked:
            proposals.append(
                {
                    "row": row,
                    "team_id": str(row.get("team_id") or ""),
                    "shares": {"goal": 0.0, "assist": 0.0, "shots": 0.0},
                    "size": 0,
                    "method": "identity_review_required_no_ev_generated",
                    "confidence": "blocked",
                    "warnings": ["unsafe_rebase_identity"],
                    "identity_blocked": True,
                }
            )
            continue

        shares, size, method, confidence, warnings = estimate_shares(
            row, references, appearances, allow_documented=True
        )
        proposals.append(
            {
                "row": row,
                "team_id": str(row.get("team_id") or ""),
                "shares": shares,
                "size": size,
                "method": method,
                "confidence": confidence,
                "warnings": warnings,
                "identity_blocked": False,
            }
        )

    eligible = [proposal for proposal in proposals if not proposal["identity_blocked"]]
    apply_team_share_caps(eligible, references)

    rows = []
    for proposal in proposals:
        row = proposal["row"]
        output = {
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "team_id": row.get("team_id"),
            "position": row.get("position"),
            "price": int(number(row.get("price"))),
            "start_prob": number(row.get("start_prob")),
            "root_cause": row.get("root_cause"),
            "reference_population_size": proposal["size"],
            "reference_method": proposal["method"],
            "estimated_goal_share": proposal["shares"]["goal"],
            "estimated_assist_share": proposal["shares"]["assist"],
            "estimated_shots_share": proposal["shares"]["shots"],
            "confidence": proposal["confidence"],
            "fantasy_relevance_score": number(row.get("fantasy_relevance_score")),
            "identity_safe_for_generation": "no" if proposal["identity_blocked"] else "yes",
        }

        if proposal["identity_blocked"]:
            for match_no in (1, 2, 3):
                for component in (
                    "goal_ev",
                    "assist_ev",
                    "shots_on_target_ev",
                    "clean_sheet_ev",
                    "card_ev",
                    "result_ev",
                    "team_scores_ev",
                    "opponent_scores_ev",
                    "on_pitch_ev",
                    "total_ev_next_match",
                    "weighted_match_ev",
                ):
                    output[f"estimated_match_{match_no}_{component}"] = np.nan
            output["estimated_match_1_weighted_ev"] = np.nan
            output["estimated_match_2_weighted_ev"] = np.nan
            output["estimated_match_3_weighted_ev"] = np.nan
            output["estimated_base_ev"] = np.nan
        else:
            estimates, component_warnings = estimate_components(
                row, proposal["shares"], references, appearances, odds
            )
            proposal["warnings"].extend(component_warnings)
            if estimates["estimated_base_ev"] < 0:
                proposal["warnings"].append("negative_base_ev_requires_zero_floor")
            for key, value in estimates.items():
                if key == "estimated_base_ev":
                    output[key] = value
                elif key.startswith("match_"):
                    output[f"estimated_{key}"] = value
            for match_no in (1, 2, 3):
                output[f"estimated_match_{match_no}_weighted_ev"] = estimates[
                    f"match_{match_no}_weighted_match_ev"
                ]

        output["warning_flags"] = ";".join(sorted(set(proposal["warnings"])))
        rows.append(output)

    return pd.DataFrame(rows).sort_values(
        ["identity_safe_for_generation", "fantasy_relevance_score"],
        ascending=[False, False],
    )


def metric_rows(validation: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, frame in validation.groupby(group_column, dropna=False):
        relative = frame["relative_error"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                group_column: group,
                "players": len(frame),
                "median_absolute_error": frame["absolute_error"].median(),
                "mean_absolute_error": frame["absolute_error"].mean(),
                "median_signed_error": frame["signed_error"].median(),
                "median_relative_error": relative.median(),
                "over_25_pct": int(relative.gt(0.25).sum()),
                "over_50_pct": int(relative.gt(0.50).sum()),
                "over_100_pct": int(relative.gt(1.00).sum()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> list[str]:
    shown = frame.head(limit) if limit else frame
    if shown.empty:
        return ["(ingen)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            elif isinstance(value, (float, np.floating)):
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(targets: pd.DataFrame, validation: pd.DataFrame, reference_count: int) -> None:
    valid_relative = validation["relative_error"].replace([np.inf, -np.inf], np.nan).dropna()
    position_metrics = metric_rows(validation, "position")
    band_metrics = metric_rows(validation, "start_band")
    confidence_counts = targets["confidence"].value_counts()
    warning_counts = Counter(
        warning
        for flags in targets["warning_flags"].fillna("")
        for warning in str(flags).split(";")
        if warning
    )

    top_missing = targets.loc[targets["identity_safe_for_generation"].eq("yes")].sort_values(
        "fantasy_relevance_score", ascending=False
    ).head(25)
    identity_blocked = targets.loc[targets["identity_safe_for_generation"].eq("no")]
    high_confidence = targets.loc[targets["confidence"].eq("high")]
    medium_confidence = targets.loc[targets["confidence"].eq("medium")]
    low_confidence = targets.loc[targets["confidence"].eq("low")]

    median_abs = validation["absolute_error"].median()
    mean_abs = validation["absolute_error"].mean()
    median_rel = valid_relative.median()
    over_25 = int(valid_relative.gt(0.25).sum())
    over_50 = int(valid_relative.gt(0.50).sum())
    over_100 = int(valid_relative.gt(1.00).sum())
    position_bias = {
        str(row["position"]): float(row["median_signed_error"])
        for _, row in position_metrics.iterrows()
    }
    band_bias = {
        str(row["start_band"]): float(row["median_signed_error"])
        for _, row in band_metrics.iterrows()
    }
    position_bias_text = ", ".join(
        f"{name} {'over' if value > 0 else 'under'} {abs(value):.3f}"
        for name, value in position_bias.items()
    )
    band_bias_text = ", ".join(
        f"{name} {'over' if value > 0 else 'under'} {abs(value):.3f}"
        for name, value in band_bias.items()
    )

    if median_rel <= 0.20 and over_50 / max(len(valid_relative), 1) <= 0.20:
        conclusion = "egnet kun med konservativ cap/floor"
    elif median_rel <= 0.35 and over_100 / max(len(valid_relative), 1) <= 0.20:
        conclusion = "egnet kun til manuel review"
    else:
        conclusion = "ikke præcis nok"

    lines = [
        "# Missing Player Component Source Experiment",
        "",
        "## Metode",
        "",
        f"- Referencepopulation: {reference_count} spillere med komplette komponenter og dokumenterede offensive shares.",
        "- Shares estimeres med shrinkage mellem samme hold/position og samme position/startniveau.",
        "- Dokumenterede shares anvendes direkte, når de findes.",
        "- Offensive komponentrater estimeres robust fra samme holds komplette spillere med positionsfallback.",
        "- Clean sheet, resultat, team-score, opponent-score og on-pitch følger de eksisterende generelle modelregler.",
        f"- Samlede nye shares skaleres mod et konservativt holdloft på {TEAM_SHARE_CAP:.2f}. Pris bruges kun til prioritering, aldrig som performance-input.",
        "- De 12 usikre Holdet-rebase-identiteter får ingen estimeret EV.",
        "",
        "## Leave-One-Out Validation",
        "",
        f"- Spillere testet: {len(validation)}",
        f"- Median absolut fejl: {median_abs:.4f}",
        f"- Gennemsnitlig absolut fejl: {mean_abs:.4f}",
        f"- Median relativ fejl: {median_rel:.1%}",
        f"- Afvigelse over 25%: {over_25}",
        f"- Afvigelse over 50%: {over_50}",
        f"- Afvigelse over 100%: {over_100}",
        "",
        "### Fejl pr. position",
        "",
        *markdown_table(
            position_metrics,
            [
                "position",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_signed_error",
                "median_relative_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "### Fejl pr. startniveau",
        "",
        *markdown_table(
            band_metrics,
            [
                "start_band",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_signed_error",
                "median_relative_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "Positiv median signed error betyder systematisk overvurdering; negativ betyder undervurdering.",
        "",
        f"Positionsbias: {position_bias_text}.",
        "",
        f"Startniveau-bias: {band_bias_text}. Metoden undervurderer især `strong_starter` i absolut EV og har den højeste relative fejl for `reserve`.",
        "",
        "## Sikkerhed",
        "",
        *[f"- `{name}`: {int(count)}" for name, count in confidence_counts.items()],
        "",
        "Hyppigste advarsler:",
        "",
        *([f"- `{name}`: {count}" for name, count in warning_counts.most_common(12)] or ["- Ingen"]),
        "",
        "### Høj sikkerhed",
        "",
        ", ".join(high_confidence["player_name"].astype(str).tolist()) or "(ingen)",
        "",
        "### Medium sikkerhed",
        "",
        ", ".join(medium_confidence["player_name"].astype(str).tolist()) or "(ingen)",
        "",
        "### Lav sikkerhed",
        "",
        ", ".join(low_confidence["player_name"].astype(str).tolist()) or "(ingen)",
        "",
        "## 12 Holdet-Rebase Identitetsproblemer",
        "",
        *markdown_table(
            identity_blocked,
            [
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "reference_method",
                "warning_flags",
            ],
        ),
        "",
        "## Top 25 Manglende Spillere",
        "",
        *markdown_table(
            top_missing,
            [
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "root_cause",
                "reference_population_size",
                "confidence",
                "estimated_base_ev",
                "warning_flags",
            ],
        ),
        "",
        "## Konklusion",
        "",
        f"**{conclusion}**",
        "",
        "Konklusionen er baseret på leave-one-out-fejlen, ikke på om de estimerede værdier ser plausible ud enkeltvis.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ev, roots, signals, odds_frame = load_inputs()
    references = complete_reference_population(ev)
    signals_by_id = signal_lookup(signals)
    appearances = {
        str(row.get("player_id")): appearance_probability(row, signals_by_id)
        for _, row in ev.iterrows()
    }
    odds = odds_lookup(odds_frame)

    validation = validation_experiment(references, appearances, odds)
    targets = target_experiment(ev, roots, references, appearances, odds)
    targets.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(targets, validation, len(references))

    print(f"Reference players: {len(references)}")
    print(f"Validation players: {len(validation)}")
    print(f"Target players: {len(targets)}")
    print(f"Median absolute error: {validation['absolute_error'].median():.6f}")
    print(f"Mean absolute error: {validation['absolute_error'].mean():.6f}")
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
