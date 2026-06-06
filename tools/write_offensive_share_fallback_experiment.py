from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EV_PATH = DATA / "player_ev_group_stage_v1.csv"
FIXTURE_PATH = DATA / "fixture_strength_multipliers.csv"
SET_PIECE_PATH = DATA / "set_piece_takers_matched.csv"
TM_USAGE_PATH = DATA / "transfermarkt_national_team" / "player_national_team_usage_transfermarkt.csv"

MISSING_AUDIT_CSV = DATA / "offensive_share_missing_audit.csv"
MISSING_AUDIT_MD = DATA / "offensive_share_missing_audit.md"
EXPERIMENT_CSV = DATA / "offensive_share_fallback_experiment.csv"
EXPERIMENT_MD = DATA / "offensive_share_fallback_experiment.md"

SHARE_COLS = {
    "goal": "goal_share_norm",
    "assist": "assist_share_norm",
    "sot": "sot_share_norm",
}
EV_COLS = {
    "goal": "goal_ev",
    "assist": "assist_ev",
    "sot": "shots_on_target_ev",
}
POINTS = {
    "goal": {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0},
    "assist": {"GK": 3.0, "DEF": 3.0, "MID": 3.0, "FWD": 3.0},
    "sot": {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0},
}
ROUND_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.90}
PRICE_QUALITY_WEIGHT = 0.45
TEAM_SHARE_HARD_CAP = 0.90

SANITY_NAMES = [
    "Raphinha",
    "Mahmoud Trezeguet",
    "Neymar Jr.",
    "Kenan Yildiz",
    "Christian Pulisic",
    "Viktor Gyökeres",
    "Patrik Schick",
    "Hakan Calhanoglu",
    "Salem Al-Dawsari",
    "Federico Valverde",
    "Bruno Guimaraes",
    "Romelu Lukaku",
    "Tomas Soucek",
    "Antonio Nusa",
    "Brian Gutierrez",
]


def number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def norm(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(text.split())


def price_band(price_pct: float) -> str:
    if price_pct < 0.25:
        return "budget"
    if price_pct < 0.50:
        return "lower_mid"
    if price_pct < 0.75:
        return "upper_mid"
    return "premium"


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    limit: int | None = None,
) -> list[str]:
    shown = frame.head(limit) if limit else frame
    if shown.empty:
        return ["(ingen)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            elif isinstance(value, (float, np.floating)):
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(EV_PATH, low_memory=False)
    fixture = pd.read_csv(FIXTURE_PATH, low_memory=False)
    set_piece = (
        pd.read_csv(SET_PIECE_PATH, low_memory=False)
        if SET_PIECE_PATH.exists()
        else pd.DataFrame()
    )
    tm_usage = (
        pd.read_csv(TM_USAGE_PATH, low_memory=False)
        if TM_USAGE_PATH.exists()
        else pd.DataFrame()
    )
    return ev, fixture, set_piece, tm_usage


def enrich_features(ev: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    work = ev.copy()
    numeric = [
        "price",
        "start_prob",
        "minute_share",
        "weighted_group_stage_ev_before_price_quality",
        "optimizer_ev",
        "price_quality_ev",
        *SHARE_COLS.values(),
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")

    work["position"] = work["position"].fillna("").astype(str).str.upper()
    work["team_id"] = work["team_id"].fillna("").astype(str).str.upper()
    work["price_pct_position"] = work.groupby("position")["price"].rank(
        pct=True, method="average"
    )
    position_minute_median = work.groupby("position")["minute_share"].transform("median")
    work["minute_proxy"] = (
        work["minute_share"]
        .div(position_minute_median.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(work["start_prob"])
        .clip(0.0, 1.5)
    )

    environment_rows: list[dict[str, Any]] = []
    for _, row in fixture.iterrows():
        home = str(row.get("home") or "").upper()
        away = str(row.get("away") or "").upper()
        environment_rows.extend(
            [
                {
                    "team_id": home,
                    "attack_environment": (
                        0.6 * number(row.get("home_goal_multiplier"), 1.0)
                        + 0.4 * number(row.get("home_assist_multiplier"), 1.0)
                    ),
                },
                {
                    "team_id": away,
                    "attack_environment": (
                        0.6 * number(row.get("away_goal_multiplier"), 1.0)
                        + 0.4 * number(row.get("away_assist_multiplier"), 1.0)
                    ),
                },
            ]
        )
    environment = (
        pd.DataFrame(environment_rows)
        .groupby("team_id", as_index=False)["attack_environment"]
        .mean()
    )
    work = work.merge(environment, on="team_id", how="left")
    work["attack_environment"] = work["attack_environment"].fillna(1.0)
    work["price_band"] = work["price_pct_position"].map(price_band)
    return work


def reference_population(ev: pd.DataFrame) -> pd.DataFrame:
    share_valid = ev[list(SHARE_COLS.values())].notna().all(axis=1)
    component_valid = pd.Series(True, index=ev.index)
    for match_no in (1, 2, 3):
        for suffix in EV_COLS.values():
            component_valid &= pd.to_numeric(
                ev.get(f"match_{match_no}_{suffix}"), errors="coerce"
            ).notna()
    base_valid = pd.to_numeric(
        ev["weighted_group_stage_ev_before_price_quality"], errors="coerce"
    ).notna()
    references = ev.loc[share_valid & component_valid & base_valid].copy()
    references["_appearance_proxy"] = references.apply(appearance_proxy, axis=1)
    for component, share_col in SHARE_COLS.items():
        denominator = (
            pd.to_numeric(references[share_col], errors="coerce")
            * references["_appearance_proxy"].clip(lower=0.05)
        )
        for match_no in (1, 2, 3):
            event = pd.to_numeric(
                references[f"match_{match_no}_{EV_COLS[component]}"],
                errors="coerce",
            )
            references[f"_rate_{match_no}_{component}"] = event.div(
                denominator.replace(0, np.nan)
            )
    references["_actual_offensive_points"] = references.apply(
        actual_offensive_points, axis=1
    )
    references["_nonoffensive_base"] = (
        pd.to_numeric(
            references["weighted_group_stage_ev_before_price_quality"],
            errors="coerce",
        )
        - references["_actual_offensive_points"]
    )
    return references


def missing_population(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.loc[ev[list(SHARE_COLS.values())].isna().any(axis=1)].copy()


def feature_distance(target: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    return (
        1.45
        * (
            pd.to_numeric(candidates["price_pct_position"], errors="coerce")
            - number(target.get("price_pct_position"), 0.5)
        ).abs()
        + 1.25
        * (
            pd.to_numeric(candidates["start_prob"], errors="coerce")
            - number(target.get("start_prob"), 0.5)
        ).abs()
        + 0.75
        * (
            pd.to_numeric(candidates["minute_proxy"], errors="coerce")
            - number(target.get("minute_proxy"), 0.5)
        ).abs()
        + 0.30
        * (
            pd.to_numeric(candidates["attack_environment"], errors="coerce")
            - number(target.get("attack_environment"), 1.0)
        ).abs()
    )


def nearest_references(
    target: pd.Series,
    references: pd.DataFrame,
    exclude_player_id: str = "",
    count: int = 40,
) -> pd.DataFrame:
    candidates = references.loc[
        references["position"].eq(str(target.get("position") or ""))
        & references["player_id"].astype(str).ne(exclude_player_id)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["_distance"] = feature_distance(target, candidates)
    return candidates.nsmallest(min(count, len(candidates)), "_distance")


def variant_a_shares(
    target: pd.Series,
    references: pd.DataFrame,
    exclude_player_id: str = "",
    nearest: pd.DataFrame | None = None,
) -> tuple[dict[str, float], int, list[str]]:
    nearest = (
        nearest.head(40).copy()
        if nearest is not None
        else nearest_references(target, references, exclude_player_id)
    )
    warnings: list[str] = []
    if len(nearest) < 12:
        warnings.append("small_position_reference")
    if nearest.empty:
        return {component: 0.0 for component in SHARE_COLS}, 0, [
            "no_position_reference"
        ]

    distance = pd.to_numeric(nearest["_distance"], errors="coerce").fillna(2.0)
    weights = (1.0 / (0.08 + distance).pow(2)).to_numpy()
    estimates: dict[str, float] = {}
    position_reference = references.loc[
        references["position"].eq(str(target.get("position") or ""))
        & references["player_id"].astype(str).ne(exclude_player_id)
    ]
    for component, column in SHARE_COLS.items():
        values = pd.to_numeric(nearest[column], errors="coerce").to_numpy()
        estimate = weighted_average(values, weights)
        position_values = pd.to_numeric(
            position_reference[column], errors="coerce"
        ).dropna()
        if not position_values.empty:
            estimate = clamp(
                estimate,
                max(0.0, float(position_values.quantile(0.02))),
                float(position_values.quantile(0.98)),
            )
        estimates[component] = estimate
    return estimates, len(nearest), warnings


def team_share_targets(references: pd.DataFrame) -> dict[str, float]:
    targets: dict[str, float] = {}
    for component, column in SHARE_COLS.items():
        totals = references.groupby("team_id")[column].sum()
        targets[component] = min(
            TEAM_SHARE_HARD_CAP,
            max(0.70, float(totals.quantile(0.90))),
        )
    return targets


def position_share_caps(references: pd.DataFrame) -> dict[tuple[str, str], float]:
    caps: dict[tuple[str, str], float] = {}
    for position, group in references.groupby("position"):
        for component, column in SHARE_COLS.items():
            caps[(position, component)] = min(
                TEAM_SHARE_HARD_CAP,
                max(0.01, float(group[column].quantile(0.95))),
            )
    return caps


def allocate_variant_b(
    targets: pd.DataFrame,
    references: pd.DataFrame,
    variant_a: dict[str, dict[str, float]],
    *,
    hidden_player_id: str = "",
    precomputed_team_targets: dict[str, float] | None = None,
    precomputed_position_caps: dict[tuple[str, str], float] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    team_targets = precomputed_team_targets or team_share_targets(references)
    position_caps = precomputed_position_caps or position_share_caps(references)
    output = {
        str(row["player_id"]): {component: 0.0 for component in SHARE_COLS}
        for _, row in targets.iterrows()
    }
    warnings = {str(row["player_id"]): [] for _, row in targets.iterrows()}

    for team, team_missing in targets.groupby("team_id"):
        known = references.loc[
            references["team_id"].eq(team)
            & references["player_id"].astype(str).ne(hidden_player_id)
        ]
        for component, column in SHARE_COLS.items():
            existing = float(pd.to_numeric(known[column], errors="coerce").fillna(0).sum())
            residual = max(
                0.0,
                min(TEAM_SHARE_HARD_CAP - existing, team_targets[component] - existing),
            )
            ids = team_missing["player_id"].astype(str).tolist()
            raw = np.array(
                [max(variant_a[player_id][component], 0.002) for player_id in ids]
            )
            allocation = residual * raw / raw.sum() if raw.sum() > 0 else np.zeros(len(ids))
            for index, (_, row) in enumerate(team_missing.iterrows()):
                player_id = str(row["player_id"])
                profile_cap = min(
                    position_caps.get((str(row["position"]), component), 0.10),
                    variant_a[player_id][component] * 1.50 + 0.01,
                )
                value = min(float(allocation[index]), profile_cap)
                output[player_id][component] = value
                if value + 1e-12 < float(allocation[index]):
                    warnings[player_id].append(f"{component}_individual_cap")
                if residual <= 1e-12:
                    warnings[player_id].append(f"{component}_no_team_residual")
    return output, warnings


def appearance_proxy(row: pd.Series) -> float:
    start = clamp(number(row.get("start_prob")))
    minute_share = number(row.get("minute_share"), -1.0)
    if minute_share > 0:
        minute_based = clamp(minute_share / 0.091)
        return clamp(0.60 * start + 0.40 * minute_based)
    return clamp(start + 0.15 * (1.0 - start))


def offensive_components(
    target: pd.Series,
    shares: dict[str, float],
    references: pd.DataFrame,
    exclude_player_id: str = "",
    nearest: pd.DataFrame | None = None,
) -> tuple[dict[str, float], float, list[str]]:
    output: dict[str, float] = {}
    warnings: list[str] = []
    appearance = appearance_proxy(target)
    position = str(target.get("position") or "")
    nearest_frame = (
        nearest
        if nearest is not None
        else nearest_references(target, references, exclude_player_id, count=60)
    )
    team_frame = references.loc[
        references["team_id"].eq(str(target.get("team_id") or ""))
        & references["player_id"].astype(str).ne(exclude_player_id)
    ]
    weighted_points = 0.0
    for match_no in (1, 2, 3):
        match_points = 0.0
        for component in SHARE_COLS:
            rate_column = f"_rate_{match_no}_{component}"
            team_rates = (
                pd.to_numeric(team_frame[rate_column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            position_rates = (
                pd.to_numeric(nearest_frame[rate_column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(team_rates) >= 4:
                rate = float(team_rates.median())
                source = "team_rate"
            elif not team_rates.empty and not position_rates.empty:
                rate = (
                    0.40 * float(team_rates.median())
                    + 0.60 * float(position_rates.median())
                )
                source = "blended_team_position_rate"
            elif not position_rates.empty:
                rate = float(position_rates.median())
                source = "position_rate"
            else:
                rate = 0.0
                source = "missing_rate"
            value = shares[component] * appearance * rate
            output[f"match_{match_no}_{EV_COLS[component]}"] = value
            match_points += value * POINTS[component].get(position, 1.0)
            if source != "team_rate":
                warnings.append(f"match_{match_no}_{component}_{source}")
        weighted_points += match_points * ROUND_WEIGHTS[match_no]
    return output, weighted_points, warnings


def actual_offensive_points(row: pd.Series) -> float:
    position = str(row.get("position") or "")
    total = 0.0
    for match_no in (1, 2, 3):
        match_total = 0.0
        for component, suffix in EV_COLS.items():
            match_total += number(row.get(f"match_{match_no}_{suffix}")) * POINTS[
                component
            ].get(position, 1.0)
        total += match_total * ROUND_WEIGHTS[match_no]
    return total


def estimate_nonoffensive_base(
    target: pd.Series,
    references: pd.DataFrame,
    exclude_player_id: str = "",
    nearest: pd.DataFrame | None = None,
) -> tuple[float, list[str]]:
    nearest = (
        nearest.head(45).copy()
        if nearest is not None
        else nearest_references(target, references, exclude_player_id, count=45)
    )
    if nearest.empty:
        return 0.0, ["missing_nonoffensive_reference"]
    bases = pd.to_numeric(nearest["_nonoffensive_base"], errors="coerce")
    distance = pd.to_numeric(nearest["_distance"], errors="coerce").fillna(2.0)
    weights = (1.0 / (0.10 + distance).pow(2)).to_numpy()
    estimate = weighted_average(bases.clip(lower=-0.5).to_numpy(), weights)
    return estimate, []


def optimizer_estimate(base_ev: float, price_quality_ev: float) -> float:
    return (
        (1.0 - PRICE_QUALITY_WEIGHT) * max(base_ev, 0.0)
        + PRICE_QUALITY_WEIGHT * max(price_quality_ev, 0.0)
    )


def validation_rows(references: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    precomputed_team_targets = team_share_targets(references)
    precomputed_position_caps = position_share_caps(references)
    for _, target in references.iterrows():
        player_id = str(target["player_id"])
        nearest = nearest_references(
            target, references, exclude_player_id=player_id, count=60
        )
        shares_a, size, warnings_a = variant_a_shares(
            target, references, exclude_player_id=player_id, nearest=nearest
        )
        hidden = target.to_frame().T
        shares_b_map, warnings_b_map = allocate_variant_b(
            hidden,
            references,
            {player_id: shares_a},
            hidden_player_id=player_id,
            precomputed_team_targets=precomputed_team_targets,
            precomputed_position_caps=precomputed_position_caps,
        )
        shares_b = shares_b_map[player_id]
        components_a, offense_a, component_warnings_a = offensive_components(
            target,
            shares_a,
            references,
            exclude_player_id=player_id,
            nearest=nearest,
        )
        components_b, offense_b, component_warnings_b = offensive_components(
            target,
            shares_b,
            references,
            exclude_player_id=player_id,
            nearest=nearest,
        )
        nonoffensive, nonoffensive_warnings = estimate_nonoffensive_base(
            target,
            references,
            exclude_player_id=player_id,
            nearest=nearest,
        )
        actual_base = number(target.get("weighted_group_stage_ev_before_price_quality"))
        estimated_base_a = max(0.0, nonoffensive + offense_a)
        estimated_base_b = max(0.0, nonoffensive + offense_b)
        result: dict[str, Any] = {
            "player_id": player_id,
            "player_name": target.get("player_name"),
            "team_id": target.get("team_id"),
            "position": target.get("position"),
            "price_band": target.get("price_band"),
            "reference_population_size": size,
            "actual_base_ev": actual_base,
            "estimated_base_ev_variant_a": estimated_base_a,
            "estimated_base_ev_variant_b": estimated_base_b,
            "base_abs_error_variant_a": abs(estimated_base_a - actual_base),
            "base_abs_error_variant_b": abs(estimated_base_b - actual_base),
            "base_relative_error_variant_a": (
                abs(estimated_base_a - actual_base) / abs(actual_base)
                if abs(actual_base) > 1e-9
                else np.nan
            ),
            "base_relative_error_variant_b": (
                abs(estimated_base_b - actual_base) / abs(actual_base)
                if abs(actual_base) > 1e-9
                else np.nan
            ),
            "base_signed_error_variant_a": estimated_base_a - actual_base,
            "base_signed_error_variant_b": estimated_base_b - actual_base,
            "warning_flags_variant_a": ";".join(
                sorted(set(warnings_a + component_warnings_a + nonoffensive_warnings))
            ),
            "warning_flags_variant_b": ";".join(
                sorted(
                    set(
                        warnings_b_map[player_id]
                        + component_warnings_b
                        + nonoffensive_warnings
                    )
                )
            ),
        }
        for component, column in SHARE_COLS.items():
            actual_share = number(target.get(column))
            result[f"actual_{component}_share"] = actual_share
            result[f"{component}_share_abs_error_variant_a"] = abs(
                shares_a[component] - actual_share
            )
            result[f"{component}_share_abs_error_variant_b"] = abs(
                shares_b[component] - actual_share
            )
            for match_no in (1, 2, 3):
                actual_component = number(
                    target.get(f"match_{match_no}_{EV_COLS[component]}")
                )
                result[
                    f"match_{match_no}_{component}_abs_error_variant_a"
                ] = abs(
                    components_a[f"match_{match_no}_{EV_COLS[component]}"]
                    - actual_component
                )
                result[
                    f"match_{match_no}_{component}_abs_error_variant_b"
                ] = abs(
                    components_b[f"match_{match_no}_{EV_COLS[component]}"]
                    - actual_component
                )
        rows.append(result)
    return pd.DataFrame(rows)


def fallback_experiment(
    missing: pd.DataFrame,
    references: pd.DataFrame,
    set_piece: pd.DataFrame,
    tm_usage: pd.DataFrame,
) -> pd.DataFrame:
    variant_a: dict[str, dict[str, float]] = {}
    reference_sizes: dict[str, int] = {}
    warning_a: dict[str, list[str]] = {}
    nearest_by_id: dict[str, pd.DataFrame] = {}
    for _, target in missing.iterrows():
        player_id = str(target["player_id"])
        nearest = nearest_references(target, references, count=60)
        nearest_by_id[player_id] = nearest
        shares, size, warnings = variant_a_shares(
            target, references, nearest=nearest
        )
        variant_a[player_id] = shares
        reference_sizes[player_id] = size
        warning_a[player_id] = warnings

    variant_b, warning_b = allocate_variant_b(missing, references, variant_a)
    for team, team_missing in missing.groupby("team_id"):
        known = references.loc[references["team_id"].eq(team)]
        for component, column in SHARE_COLS.items():
            projected_a = float(
                pd.to_numeric(known[column], errors="coerce").fillna(0).sum()
            ) + sum(
                variant_a[str(player_id)][component]
                for player_id in team_missing["player_id"]
            )
            if projected_a > TEAM_SHARE_HARD_CAP + 1e-9:
                for player_id in team_missing["player_id"].astype(str):
                    warning_a[player_id].append(
                        f"variant_a_{component}_team_share_over_cap"
                    )
    set_piece_ids = (
        set(set_piece["matched_player_id"].dropna().astype(str))
        if not set_piece.empty and "matched_player_id" in set_piece
        else set()
    )
    tm_ids = (
        set(tm_usage["player_id"].dropna().astype(str))
        if not tm_usage.empty and "player_id" in tm_usage
        else set()
    )

    rows: list[dict[str, Any]] = []
    for _, target in missing.iterrows():
        player_id = str(target["player_id"])
        shares_a = variant_a[player_id]
        shares_b = variant_b[player_id]
        _, offense_a, component_warnings_a = offensive_components(
            target, shares_a, references, nearest=nearest_by_id[player_id]
        )
        _, offense_b, component_warnings_b = offensive_components(
            target, shares_b, references, nearest=nearest_by_id[player_id]
        )
        nonoffensive, nonoffensive_warnings = estimate_nonoffensive_base(
            target, references, nearest=nearest_by_id[player_id]
        )
        fallback_base_a = max(0.0, nonoffensive + offense_a)
        fallback_base_b = max(0.0, nonoffensive + offense_b)
        current_base = number(
            target.get("weighted_group_stage_ev_before_price_quality")
        )
        price_quality = number(target.get("price_quality_ev"))
        flags = set(
            warning_a[player_id]
            + warning_b[player_id]
            + component_warnings_a
            + component_warnings_b
            + nonoffensive_warnings
        )
        if str(target.get("round_context_source") or "") == (
            "distributed_from_existing_optimizer_ev"
        ):
            flags.add("legacy_aggregate_round_context")
        if player_id in set_piece_ids:
            flags.add("set_piece_role_available_not_used_as_direct_share")
        if player_id not in tm_ids:
            flags.add("no_transfermarkt_usage_match")
        if target["position"] in {"MID", "FWD"} and number(target["start_prob"]) >= 0.70:
            flags.add("high_priority_offensive_starter")

        team_known = references.loc[references["team_id"].eq(target["team_id"])]
        confidence = "high"
        if reference_sizes[player_id] < 20 or len(team_known) < 10:
            confidence = "medium"
        if reference_sizes[player_id] < 10 or any(
            flag.endswith("no_team_residual") for flag in flags
        ):
            confidence = "low"

        rows.append(
            {
                "player_id": player_id,
                "player_name": target.get("player_name"),
                "team_id": target.get("team_id"),
                "position": target.get("position"),
                "price": int(number(target.get("price"))),
                "start_prob": number(target.get("start_prob")),
                "current_goal_share_norm": target.get("goal_share_norm"),
                "current_assist_share_norm": target.get("assist_share_norm"),
                "current_sot_share_norm": target.get("sot_share_norm"),
                "fallback_goal_share_norm_variant_a": shares_a["goal"],
                "fallback_assist_share_norm_variant_a": shares_a["assist"],
                "fallback_sot_share_norm_variant_a": shares_a["sot"],
                "fallback_goal_share_norm_variant_b": shares_b["goal"],
                "fallback_assist_share_norm_variant_b": shares_b["assist"],
                "fallback_sot_share_norm_variant_b": shares_b["sot"],
                "current_base_ev": current_base,
                "fallback_base_ev_variant_a": fallback_base_a,
                "fallback_base_ev_variant_b": fallback_base_b,
                "current_optimizer_ev": number(target.get("optimizer_ev")),
                "fallback_optimizer_ev_variant_a_estimate": optimizer_estimate(
                    fallback_base_a, price_quality
                ),
                "fallback_optimizer_ev_variant_b_estimate": optimizer_estimate(
                    fallback_base_b, price_quality
                ),
                "reference_population_size": reference_sizes[player_id],
                "price_band": target.get("price_band"),
                "round_context_source": target.get("round_context_source"),
                "confidence": confidence,
                "reason_flags": ";".join(sorted(flags)),
            }
        )
    return pd.DataFrame(rows)


def missing_audit(
    missing: pd.DataFrame,
    experiment: pd.DataFrame,
    set_piece: pd.DataFrame,
) -> pd.DataFrame:
    set_piece_ids = (
        set(set_piece["matched_player_id"].dropna().astype(str))
        if not set_piece.empty and "matched_player_id" in set_piece
        else set()
    )
    work = missing.merge(
        experiment[
            [
                "player_id",
                "fallback_base_ev_variant_a",
                "fallback_base_ev_variant_b",
                "confidence",
                "reason_flags",
            ]
        ],
        on="player_id",
        how="left",
    )
    max_price = max(number(work["price"].max()), 1.0)
    max_optimizer = max(number(work["optimizer_ev"].max()), 1.0)
    work["offensive_role_score"] = (
        work["position"].map({"GK": 0.0, "DEF": 0.08, "MID": 0.30, "FWD": 0.45}).fillna(0)
        + 0.30 * work["start_prob"].fillna(0)
        + 0.15 * work["price"].fillna(0) / max_price
        + 0.10 * work["optimizer_ev"].fillna(0) / max_optimizer
        + 0.12 * work["player_id"].astype(str).isin(set_piece_ids).astype(float)
    )
    output = pd.DataFrame(
        {
            "player_id": work["player_id"],
            "player_name": work["player_name"],
            "team_id": work["team_id"],
            "position": work["position"],
            "price": work["price"],
            "start_prob": work["start_prob"],
            "missing_goal_share_norm": work["goal_share_norm"].isna(),
            "missing_assist_share_norm": work["assist_share_norm"].isna(),
            "missing_sot_share_norm": work["sot_share_norm"].isna(),
            "round_context_source": work["round_context_source"],
            "current_weighted_group_stage_ev_before_price_quality": work[
                "weighted_group_stage_ev_before_price_quality"
            ],
            "current_optimizer_ev": work["optimizer_ev"],
            "price_band": work["price_band"],
            "offensive_role_score": work["offensive_role_score"],
            "fallback_base_ev_variant_a": work["fallback_base_ev_variant_a"],
            "fallback_base_ev_variant_b": work["fallback_base_ev_variant_b"],
            "confidence": work["confidence"],
            "reason_flags": work["reason_flags"],
        }
    )
    return output.sort_values(
        ["offensive_role_score", "start_prob", "price", "current_optimizer_ev"],
        ascending=False,
    )


def metric_summary(
    validation: pd.DataFrame,
    variant: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    groups = (
        validation.groupby(group_column, dropna=False)
        if group_column
        else [("all", validation)]
    )
    rows: list[dict[str, Any]] = []
    for group, frame in groups:
        relative = frame[f"base_relative_error_variant_{variant}"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        rows.append(
            {
                group_column or "group": group,
                "players": len(frame),
                "median_absolute_error": frame[
                    f"base_abs_error_variant_{variant}"
                ].median(),
                "mean_absolute_error": frame[
                    f"base_abs_error_variant_{variant}"
                ].mean(),
                "median_relative_error": relative.median(),
                "median_signed_error": frame[
                    f"base_signed_error_variant_{variant}"
                ].median(),
                "over_25_pct": int(relative.gt(0.25).sum()),
                "over_50_pct": int(relative.gt(0.50).sum()),
                "over_100_pct": int(relative.gt(1.00).sum()),
            }
        )
    return pd.DataFrame(rows)


def share_validation_summary(
    validation: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    rows = []
    for component in SHARE_COLS:
        column = f"{component}_share_abs_error_variant_{variant}"
        actual = validation[f"actual_{component}_share"].abs()
        relative = validation[column].div(actual.replace(0, np.nan))
        rows.append(
            {
                "component": component,
                "median_absolute_error": validation[column].median(),
                "mean_absolute_error": validation[column].mean(),
                "median_relative_error": relative.median(),
                "over_25_pct": int(relative.gt(0.25).sum()),
                "over_50_pct": int(relative.gt(0.50).sum()),
                "over_100_pct": int(relative.gt(1.00).sum()),
            }
        )
    return pd.DataFrame(rows)


def component_validation_summary(
    validation: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    rows = []
    for component in SHARE_COLS:
        columns = [
            f"match_{match_no}_{component}_abs_error_variant_{variant}"
            for match_no in (1, 2, 3)
        ]
        values = validation[columns].to_numpy().ravel()
        rows.append(
            {
                "component": component,
                "median_match_component_absolute_error": float(np.nanmedian(values)),
                "mean_match_component_absolute_error": float(np.nanmean(values)),
            }
        )
    return pd.DataFrame(rows)


def sanity_rows(
    ev: pd.DataFrame,
    experiment: pd.DataFrame,
) -> pd.DataFrame:
    ev_by_name = {norm(row["player_name"]): row for _, row in ev.iterrows()}
    experiment_by_name = {
        norm(row["player_name"]): row for _, row in experiment.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for requested_name in SANITY_NAMES:
        key = norm(requested_name)
        source = ev_by_name.get(key)
        estimate = experiment_by_name.get(key)
        if source is None:
            rows.append(
                {
                    "requested_name": requested_name,
                    "status": "not_found",
                }
            )
            continue
        has_missing = any(pd.isna(source.get(column)) for column in SHARE_COLS.values())
        status = "missing_shares_estimated" if estimate is not None else "shares_present"
        rows.append(
            {
                "requested_name": requested_name,
                "player_name": source.get("player_name"),
                "team_id": source.get("team_id"),
                "position": source.get("position"),
                "start_prob": source.get("start_prob"),
                "status": status if has_missing else "shares_present_no_fallback_needed",
                "current_base_ev": source.get(
                    "weighted_group_stage_ev_before_price_quality"
                ),
                "fallback_base_ev_variant_a": (
                    estimate.get("fallback_base_ev_variant_a")
                    if estimate is not None
                    else np.nan
                ),
                "fallback_base_ev_variant_b": (
                    estimate.get("fallback_base_ev_variant_b")
                    if estimate is not None
                    else np.nan
                ),
                "current_optimizer_ev": source.get("optimizer_ev"),
                "fallback_optimizer_ev_variant_a_estimate": (
                    estimate.get("fallback_optimizer_ev_variant_a_estimate")
                    if estimate is not None
                    else np.nan
                ),
                "fallback_optimizer_ev_variant_b_estimate": (
                    estimate.get("fallback_optimizer_ev_variant_b_estimate")
                    if estimate is not None
                    else np.nan
                ),
                "confidence": estimate.get("confidence") if estimate is not None else "",
            }
        )
    return pd.DataFrame(rows)


def choose_recommendation(
    summary_a: pd.DataFrame,
    summary_b: pd.DataFrame,
) -> tuple[str, str]:
    a = summary_a.iloc[0]
    b = summary_b.iloc[0]
    chosen = "A" if a["median_absolute_error"] <= b["median_absolute_error"] else "B"
    best = a if chosen == "A" else b
    over_50_rate = number(best["over_50_pct"]) / max(number(best["players"]), 1.0)
    median_relative = number(best["median_relative_error"])
    if median_relative <= 0.20 and over_50_rate <= 0.20:
        conclusion = "egnet kun med konservativ cap/floor"
    elif median_relative <= 0.35 and over_50_rate <= 0.35:
        conclusion = "egnet kun til manuel review"
    else:
        conclusion = "ikke præcis nok"
    return chosen, conclusion


def write_missing_report(
    missing: pd.DataFrame,
    audit: pd.DataFrame,
    sanity: pd.DataFrame,
) -> None:
    likely = audit["start_prob"].ge(0.70)
    mid_fwd = audit["position"].isin(["MID", "FWD"])
    distributed = audit["round_context_source"].fillna("").eq(
        "distributed_from_existing_optimizer_ev"
    )
    low_base = pd.to_numeric(
        audit["current_weighted_group_stage_ev_before_price_quality"], errors="coerce"
    ).lt(1.0)
    position_counts = audit.groupby("position").size().reset_index(name="players")
    team_counts = (
        audit.groupby("team_id")
        .size()
        .reset_index(name="players")
        .sort_values("players", ascending=False)
    )
    top = audit.head(35)
    lines = [
        "# Offensive Share Missing Audit",
        "",
        "## Omfang",
        "",
        f"- Spillere uden `goal_share_norm`: {int(missing['goal_share_norm'].isna().sum())}",
        f"- Spillere uden `assist_share_norm`: {int(missing['assist_share_norm'].isna().sum())}",
        f"- Spillere uden `sot_share_norm`: {int(missing['sot_share_norm'].isna().sum())}",
        f"- Spillere uden mindst én offensiv share: {len(audit)}",
        f"- Sandsynlige startere (`start_prob >= 0.70`) uden shares: {int(likely.sum())}",
        f"- MID/FWD med `start_prob >= 0.70` uden shares: {int((likely & mid_fwd).sum())}",
        f"- Med `round_context_source = distributed_from_existing_optimizer_ev`: {int(distributed.sum())}",
        f"- Disse med base-EV under 1.00: {int((distributed & low_base).sum())}",
        "",
        "Alle rækker i denne audit har alle tre shares manglende; der er ingen delvist udfyldte share-rækker.",
        "",
        "## Fordeling pr. position",
        "",
        *markdown_table(position_counts, ["position", "players"]),
        "",
        "## Lande med flest gaps",
        "",
        *markdown_table(team_counts, ["team_id", "players"], 20),
        "",
        "## Top offensive gaps",
        "",
        "Rangeringen bruger startchance, prisrang, nuværende optimizer-EV, position og dokumenteret dødboldsrolle som audit-prioritering. Pris anvendes ikke som direkte EV.",
        "",
        *markdown_table(
            top,
            [
                "player_name",
                "team_id",
                "position",
                "price",
                "start_prob",
                "current_weighted_group_stage_ev_before_price_quality",
                "current_optimizer_ev",
                "offensive_role_score",
                "confidence",
            ],
        ),
        "",
        "## Sanity-listen",
        "",
        *markdown_table(
            sanity,
            [
                "requested_name",
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "status",
                "current_base_ev",
            ],
        ),
    ]
    MISSING_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_report(
    references: pd.DataFrame,
    experiment: pd.DataFrame,
    validation: pd.DataFrame,
    sanity: pd.DataFrame,
) -> tuple[str, str]:
    overall_a = metric_summary(validation, "a")
    overall_b = metric_summary(validation, "b")
    position_a = metric_summary(validation, "a", "position")
    position_b = metric_summary(validation, "b", "position")
    price_a = metric_summary(validation, "a", "price_band")
    price_b = metric_summary(validation, "b", "price_band")
    shares_a = share_validation_summary(validation, "a")
    shares_b = share_validation_summary(validation, "b")
    components_a = component_validation_summary(validation, "a")
    components_b = component_validation_summary(validation, "b")
    recommended, conclusion = choose_recommendation(overall_a, overall_b)
    confidence = (
        experiment.groupby("confidence").size().reset_index(name="players")
    )
    warnings = Counter(
        flag
        for flags in experiment["reason_flags"].fillna("")
        for flag in str(flags).split(";")
        if flag
    )
    warning_frame = pd.DataFrame(
        warnings.most_common(15), columns=["warning_flag", "players"]
    )

    lines = [
        "# Offensive Share Fallback Experiment",
        "",
        "## Metode",
        "",
        f"- Referencepopulation: {len(references)} spillere med alle tre shares og kampkomponenter.",
        "- Variant A: positionsspecifik nearest-reference-model med prisrang inden for position, startchance, minutproxy og holdets offensive fixturemiljø.",
        "- Variant B: team-residualmodel. Variant A bruges kun som fordelingsvægt; kendte shares bevares, residualen fordeles op til et empirisk teammål og et hårdt loft på 0.90.",
        "- Variant B har desuden positionsbaserede individuelle caps, så én spiller ikke absorberer en stor rest-share alene.",
        "- Kampkomponenter beregnes fra robuste teamrater med positionsfallback og de eksisterende Holdet.dk-pointregler.",
        "- Base-EV består af en reference-estimeret ikke-offensiv baseline plus de estimerede offensive komponenter.",
        "- Auditens optimizer-estimat bruger den eksisterende formel `0.55 * base_ev + 0.45 * price_quality_ev`; produktionsfelter ændres ikke.",
        "- Transfermarkt-usage indeholder caps/startbrug, men ikke en stabil kamp-for-kamp goal/assist/SOT-rate for hele populationen. Derfor er der ikke bygget en variant C.",
        "",
        "## Leave-One-Out-validering",
        "",
        "Hver referencespillers shares skjules, spilleren fjernes fra sit eget referencegrundlag, og fallbacken sammenlignes med de faktiske shares, kampkomponenter og base-EV.",
        "",
        "### Samlet base-EV-fejl",
        "",
        "Variant A:",
        "",
        *markdown_table(
            overall_a,
            [
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "Variant B:",
        "",
        *markdown_table(
            overall_b,
            [
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "### Share-fejl",
        "",
        "Variant A:",
        "",
        *markdown_table(
            shares_a,
            [
                "component",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "Variant B:",
        "",
        *markdown_table(
            shares_b,
            [
                "component",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "### Kampkomponent-fejl",
        "",
        "Variant A:",
        "",
        *markdown_table(
            components_a,
            [
                "component",
                "median_match_component_absolute_error",
                "mean_match_component_absolute_error",
            ],
        ),
        "",
        "Variant B:",
        "",
        *markdown_table(
            components_b,
            [
                "component",
                "median_match_component_absolute_error",
                "mean_match_component_absolute_error",
            ],
        ),
        "",
        "### Base-EV-fejl pr. position",
        "",
        "Variant A:",
        "",
        *markdown_table(
            position_a,
            [
                "position",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "Variant B:",
        "",
        *markdown_table(
            position_b,
            [
                "position",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "### Base-EV-fejl pr. prisniveau",
        "",
        "Variant A:",
        "",
        *markdown_table(
            price_a,
            [
                "price_band",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "Variant B:",
        "",
        *markdown_table(
            price_b,
            [
                "price_band",
                "players",
                "median_absolute_error",
                "mean_absolute_error",
                "median_relative_error",
                "median_signed_error",
                "over_25_pct",
                "over_50_pct",
                "over_100_pct",
            ],
        ),
        "",
        "## Missing-share-estimater",
        "",
        "Sikkerhed:",
        "",
        *markdown_table(confidence, ["confidence", "players"]),
        "",
        "Hyppigste advarsler:",
        "",
        *markdown_table(warning_frame, ["warning_flag", "players"]),
        "",
        "## Sanity",
        "",
        *markdown_table(
            sanity,
            [
                "requested_name",
                "player_name",
                "team_id",
                "position",
                "start_prob",
                "status",
                "current_base_ev",
                "fallback_base_ev_variant_a",
                "fallback_base_ev_variant_b",
                "current_optimizer_ev",
                "fallback_optimizer_ev_variant_a_estimate",
                "fallback_optimizer_ev_variant_b_estimate",
                "confidence",
            ],
        ),
        "",
        "## Anbefaling",
        "",
        f"- Bedste validerede variant: **Variant {recommended}**.",
        f"- Produktionsvurdering: **{conclusion}**.",
        "- Variant B er den stærkeste sikkerhedsbarriere mod dobbeltallokering, men kan give nul/lav residual på hold, hvor kendte shares allerede fylder teammålet.",
        "- Spillere med `low` confidence eller `*_no_team_residual` bør ikke auto-opdateres uden manuel review.",
        "- Dette eksperiment skriver ikke shares eller EV tilbage til produktionsdata.",
    ]
    EXPERIMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommended, conclusion


def main() -> int:
    ev_raw, fixture, set_piece, tm_usage = load_inputs()
    ev = enrich_features(ev_raw, fixture)
    references = reference_population(ev)
    missing = missing_population(ev)

    validation = validation_rows(references)
    experiment = fallback_experiment(missing, references, set_piece, tm_usage)
    audit = missing_audit(missing, experiment, set_piece)
    sanity = sanity_rows(ev, experiment)

    audit.to_csv(MISSING_AUDIT_CSV, index=False, encoding="utf-8-sig")
    experiment.to_csv(EXPERIMENT_CSV, index=False, encoding="utf-8-sig")
    write_missing_report(missing, audit, sanity)
    recommended, conclusion = write_experiment_report(
        references, experiment, validation, sanity
    )

    print(f"Reference players: {len(references)}")
    print(f"Missing-share players: {len(missing)}")
    print(
        "Likely starters missing shares: "
        f"{int(pd.to_numeric(missing['start_prob'], errors='coerce').ge(0.70).sum())}"
    )
    print(
        "MID/FWD likely starters missing shares: "
        f"{int((missing['position'].isin(['MID', 'FWD']) & missing['start_prob'].ge(0.70)).sum())}"
    )
    for variant in ("a", "b"):
        summary = metric_summary(validation, variant).iloc[0]
        print(
            f"Variant {variant.upper()}: median_abs={summary['median_absolute_error']:.6f} "
            f"mean_abs={summary['mean_absolute_error']:.6f} "
            f"median_rel={summary['median_relative_error']:.2%}"
        )
    print(f"Recommended variant: {recommended}")
    print(f"Conclusion: {conclusion}")
    print(f"Wrote: {MISSING_AUDIT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {MISSING_AUDIT_MD.relative_to(ROOT)}")
    print(f"Wrote: {EXPERIMENT_CSV.relative_to(ROOT)}")
    print(f"Wrote: {EXPERIMENT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
