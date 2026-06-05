from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TM_DIR = DATA_DIR / "transfermarkt_national_team"

OUT_MATCHES = TM_DIR / "player_national_team_matches_classified.csv"
OUT_SUMMARY = TM_DIR / "player_national_team_usage_by_competition.csv"
OUT_PLAYER_SUMMARY = TM_DIR / "player_national_team_usage_competitive_summary.csv"
AUDIT_CSV = TM_DIR / "competition_classifier_audit.csv"
AUDIT_MD = TM_DIR / "competition_classifier_audit.md"


COMPETITION_CODE_MAP = {
    # Senior friendlies.
    "FS": ("venskabskamp", "Venskabskamp", "International friendly"),
    # FIFA / confederation qualification.
    "WMQ1": ("kvalifikation", "Kvalifikation", "World Cup qualification, AFC"),
    "WMQ2": ("kvalifikation", "Kvalifikation", "World Cup qualification, CAF"),
    "WMQ3": ("kvalifikation", "Kvalifikation", "World Cup qualification, CONCACAF"),
    "WMQ4": ("kvalifikation", "Kvalifikation", "World Cup qualification, CONMEBOL"),
    "WMQ5": ("kvalifikation", "Kvalifikation", "World Cup qualification, OFC"),
    "WMQ6": ("kvalifikation", "Kvalifikation", "World Cup qualification, UEFA"),
    "EMQ": ("kvalifikation", "Kvalifikation", "European Championship qualification"),
    "AFCQ": ("kvalifikation", "Kvalifikation", "Africa Cup of Nations qualification"),
    "ACQU": ("kvalifikation", "Kvalifikation", "Asian Cup qualification"),
    "GCQU": ("kvalifikation", "Kvalifikation", "Gold Cup qualification"),
    "GCQ5": ("kvalifikation", "Kvalifikation", "Gold Cup qualification"),
    "CARQ": ("kvalifikation", "Kvalifikation", "Caribbean / CONCACAF qualification"),
    "FARQ": ("kvalifikation", "Kvalifikation", "Regional championship qualification"),
    "POWM": ("kvalifikation", "Kvalifikation", "World Cup qualification play-off"),
    "POEM": ("kvalifikation", "Kvalifikation", "European qualification play-off"),
    "CACP": ("kvalifikation", "Kvalifikation", "Copa America Centenario qualification play-off"),
    "CCPL": ("kvalifikation", "Kvalifikation", "CONCACAF Cup qualification play-off"),
    # Nations League, including finals and play-offs.
    "UNLA": ("nations_league", "Nations League", "UEFA Nations League A"),
    "UNLB": ("nations_league", "Nations League", "UEFA Nations League B"),
    "UNLC": ("nations_league", "Nations League", "UEFA Nations League C"),
    "UNLD": ("nations_league", "Nations League", "UEFA Nations League D"),
    "UNPO": ("nations_league", "Nations League", "UEFA Nations League play-off"),
    "UNFI": ("nations_league_finals", "Nations League-slutrunde", "UEFA Nations League finals"),
    "CNLA": ("nations_league", "Nations League", "CONCACAF Nations League A"),
    "CNLB": ("nations_league", "Nations League", "CONCACAF Nations League B"),
    "CNLQ": ("nations_league", "Nations League", "CONCACAF Nations League qualification"),
    "CNNF": ("nations_league_finals", "Nations League-slutrunde", "CONCACAF Nations League finals"),
    # Senior finals and continental tournaments.
    "FIWC": ("slutrunde", "Slutrunde", "FIFA World Cup"),
    "EURO": ("slutrunde", "Slutrunde", "UEFA European Championship"),
    "AFCN": ("slutrunde", "Slutrunde", "Africa Cup of Nations"),
    "AFAC": ("slutrunde", "Slutrunde", "AFC Asian Cup"),
    "GOCU": ("slutrunde", "Slutrunde", "CONCACAF Gold Cup"),
    "COPA": ("slutrunde", "Slutrunde", "Copa America"),
    "ARCP": ("slutrunde", "Slutrunde", "FIFA Arab Cup"),
    "CHAN": ("slutrunde", "Slutrunde", "African Nations Championship"),
    "AGUC": ("slutrunde", "Slutrunde", "Regional senior Gold Cup"),
    "CONC": ("slutrunde", "Slutrunde", "FIFA Confederations Cup"),
    "CA16": ("slutrunde", "Slutrunde", "Copa America Centenario"),
    "EAFC": ("slutrunde", "Slutrunde", "EAFF Championship"),
    "CAFA": ("slutrunde", "Slutrunde", "CAFA Nations Cup"),
    "WAF1": ("slutrunde", "Slutrunde", "WAFF Championship"),
    "CENC": ("slutrunde", "Slutrunde", "Central American Championship"),
    "OFCN": ("slutrunde", "Slutrunde", "OFC Nations Cup"),
    "AFT": ("slutrunde", "Slutrunde", "Regional senior national-team tournament"),
    "TRIN": ("oevrig_turnering", "Øvrig turnering", "Tri-nation / invitational tournament"),
    # Youth competitions. Kept separate from senior finals/qualification.
    "U21Q": ("ungdom_kvalifikation", "Ungdomskvalifikation", "UEFA U21 qualification"),
    "A23Q": ("ungdom_kvalifikation", "Ungdomskvalifikation", "AFC U23 qualification"),
    "20WC": ("ungdom_slutrunde", "Ungdomsslutrunde", "FIFA U20 World Cup"),
    "W23C": ("ungdom_slutrunde", "Ungdomsslutrunde", "U23 World championship"),
    "21EU": ("ungdom_slutrunde", "Ungdomsslutrunde", "UEFA U21 Championship"),
    "23AF": ("ungdom_slutrunde", "Ungdomsslutrunde", "CAF U23 Championship"),
    "20AC": ("ungdom_slutrunde", "Ungdomsslutrunde", "AFC U20 Championship"),
    "CA17": ("ungdom_slutrunde", "Ungdomsslutrunde", "U17 continental championship"),
    "C220": ("ungdom_slutrunde", "Ungdomsslutrunde", "U20 continental championship"),
    "2SAM": ("ungdom_slutrunde", "Ungdomsslutrunde", "South American U20 Championship"),
    "SAM2": ("ungdom_slutrunde", "Ungdomsslutrunde", "South American U20 Championship"),
}

COMPETITION_WEIGHTS = {
    "slutrunde": 1.00,
    "nations_league_finals": 0.95,
    "kvalifikation": 0.90,
    "nations_league": 0.78,
    "oevrig_turnering": 0.62,
    "venskabskamp": 0.52,
    "ungdom_slutrunde": 0.35,
    "ungdom_kvalifikation": 0.30,
    "uklar_turnering": 0.45,
    "ukendt": 0.40,
}


FINALS_KEYWORDS = [
    "world cup",
    "uefa euro",
    "european championship",
    "copa america",
    "afcon",
    "africa cup",
    "asian cup",
    "concacaf gold cup",
    "gold cup",
    "finals",
]

QUALIFIER_KEYWORDS = [
    "world cup qualification",
    "world cup qualifier",
    "wc qualification",
    "wc qualifier",
    "euro qualification",
    "euro qualifier",
    "european qualifiers",
    "qualification",
    "qualifier",
    "qualifying",
]

NATIONS_LEAGUE_KEYWORDS = [
    "nations league",
    "uefa nations",
    "concacaf nations",
]

FRIENDLY_KEYWORDS = [
    "friendly",
    "friendlies",
    "international friendly",
]


def clean_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def detect_competition_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        low = clean_text(col)
        if any(word in low for word in ["competition", "compet", "tournament", "comp"]):
            candidates.append(col)

    if candidates:
        return candidates[0]

    return None


def classify_competition(row: pd.Series) -> tuple[str, str]:
    code = str(row.get("competition_raw") or row.get("competition") or "").strip().upper()
    mapped = COMPETITION_CODE_MAP.get(code)
    if mapped:
        return mapped[0], mapped[1]

    # General code families protect the classifier when Transfermarkt adds
    # another regional World Cup or Nations League division code.
    if code.startswith("WMQ"):
        return "kvalifikation", "Kvalifikation"
    if re.fullmatch(r"UNL[A-D]", code) or re.fullmatch(r"CNL[A-D]", code):
        return "nations_league", "Nations League"
    if code.endswith("Q") and code:
        return "kvalifikation", "Kvalifikation"

    parts = []

    for col in row.index:
        value = clean_text(row.get(col))
        if value:
            parts.append(value)

    text = " | ".join(parts)

    if any(keyword in text for keyword in FINALS_KEYWORDS):
        return "slutrunde", "Slutrunde"

    if any(keyword in text for keyword in QUALIFIER_KEYWORDS):
        return "kvalifikation", "Kvalifikation"

    if any(keyword in text for keyword in NATIONS_LEAGUE_KEYWORDS):
        return "nations_league", "Nations League"

    if any(keyword in text for keyword in FRIENDLY_KEYWORDS):
        return "venskabskamp", "Venskabskamp"

    # Transfermarkt-tabellen mister nogle gange selve turneringsnavnet, men
    # gruppenavn/matchday kan stadig indikere kval/slutrunde. Derfor laver vi
    # en forsigtig ekstraheuristik:
    matchday = clean_text(row.get("matchday", ""))
    if "group" in matchday or "qual" in matchday:
        return "uklar_turnering", "Uklar turnering/kval/slutrunde"

    return "ukendt", "Ukendt"




def classify_absence_reason(row: pd.Series) -> tuple[str, str]:
    state = clean_text(row.get("participation_state", ""))
    text = f"{state} {clean_text(row.get('row_text', ''))}"

    if not any(marker in text for marker in ["not in squad", "on the bench", "injured", "suspended"]):
        return "", ""

    injury_words = [
        "injury",
        "injured",
        "muscle",
        "hamstring",
        "knee",
        "ankle",
        "fitness",
        "ill",
        "illness",
        "virus",
        "knock",
        "strain",
        "tear",
        "rehab",
        "surgery",
    ]

    suspension_words = [
        "suspended",
        "suspension",
        "yellow cards",
        "red card",
        "ban",
    ]

    personal_words = [
        "personal reasons",
        "family reasons",
        "private reasons",
    ]

    if any(word in text for word in injury_words):
        return "injury_or_fitness", "Skade/fitness"

    if any(word in text for word in suspension_words):
        return "suspension", "Karantæne"

    if any(word in text for word in personal_words):
        return "personal", "Personlige årsager"

    if "on the bench" in text:
        return "bench", "Bænk"

    if "not in squad" in text:
        return "not_selected_or_unknown", "Ikke i trup / ukendt årsag"

    return "", ""


def infer_appearance_status(row: pd.Series) -> tuple[bool, bool, bool]:
    state = clean_text(row.get("participation_state", ""))
    text = f"{state} {clean_text(row.get('row_text', ''))}"

    def boolean(value: Any) -> bool:
        return clean_text(value) in {"true", "1", "yes"}

    was_not_in_squad = boolean(row.get("was_not_in_squad", False))
    was_on_bench = boolean(row.get("was_on_bench", False))
    started = boolean(row.get("started_estimate", False))

    if "not in squad" in text:
        was_not_in_squad = True
    if "on the bench" in text:
        was_on_bench = True
    if state == "in squad":
        started = False
        was_on_bench = False
        was_not_in_squad = False
    if state in {"injured", "suspended"}:
        started = False
        was_on_bench = False
        was_not_in_squad = False

    return started, was_on_bench, was_not_in_squad


def load_match_files() -> pd.DataFrame:
    files = sorted(TM_DIR.glob("*_national_matches.csv"))

    rows = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            print(f"Springer fil over: {file.name} ({exc})")
            continue

        if df.empty:
            continue

        df["source_file"] = file.name

        comp_col = detect_competition_column(df)
        if comp_col:
            df["competition_raw"] = df[comp_col].astype(str)
        elif "competition_raw" not in df.columns:
            df["competition_raw"] = ""

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)




def add_recency_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        out["date_parsed"] = pd.NaT
        out["months_ago"] = None
        out["recency_weight"] = 0.25
        return out

    out["date_parsed"] = pd.to_datetime(out["date"], format="%d/%m/%y", errors="coerce")

    # Python/pandas interprets e.g. 62 as 2062. Transfermarkt includes historic
    # matches, so dates implausibly after the current run date belong to 19xx.
    today = pd.Timestamp(datetime.now().date())
    future_mask = out["date_parsed"] > today + pd.Timedelta(days=366)
    out.loc[future_mask, "date_parsed"] = out.loc[future_mask, "date_parsed"] - pd.DateOffset(years=100)

    reference_date = min(today, out["date_parsed"].max())
    if pd.isna(reference_date):
        out["months_ago"] = None
        out["recency_weight"] = 0.25
        return out

    out["months_ago"] = ((reference_date - out["date_parsed"]).dt.days / 30.44).clip(lower=0).round(1)

    def weight(months: float) -> float:
        if pd.isna(months):
            return 0.10
        if months <= 1:
            return 1.00
        if months <= 3:
            return 0.92
        if months <= 6:
            return 0.85
        if months <= 12:
            return 0.68
        if months <= 24:
            return 0.48
        if months <= 36:
            return 0.32
        if months <= 60:
            return 0.20
        return 0.10

    out["recency_weight"] = out["months_ago"].map(weight)

    state = out.get("participation_state", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    out["injury_or_suspension"] = state.isin(["injured", "suspended"]) | out["absence_reason"].isin(
        ["injury_or_fitness", "suspension"]
    )
    out["selection_observation"] = (~out["injury_or_suspension"]).astype(float)
    out["unused_in_squad"] = (
        state.eq("in squad")
        & ~out["started_estimate_clean"].astype(bool)
        & pd.to_numeric(out.get("minutes_estimate"), errors="coerce").fillna(0).eq(0)
    )

    out["recent_weighted_start"] = out["started_estimate_clean"].astype(float) * out["recency_weight"]
    out["recent_weighted_bench"] = out["was_on_bench_clean"].astype(float) * out["recency_weight"]
    out["recent_weighted_not_in_squad"] = out["was_not_in_squad_clean"].astype(float) * out["recency_weight"]
    out["competition_weight"] = out["competition_category"].map(COMPETITION_WEIGHTS).fillna(
        COMPETITION_WEIGHTS["ukendt"]
    )
    out["selection_signal_weight"] = (
        out["selection_observation"] * out["recency_weight"] * out["competition_weight"]
    )
    out["start_signal_numerator"] = (
        out["started_estimate_clean"].astype(float) * out["selection_signal_weight"]
    )

    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "player_id",
        "player_name",
        "team_id",
        "position_model",
        "competition_category",
        "competition_category_label",
    ]

    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("player_name", "size"),
            starts=("started_estimate_clean", "sum"),
            bench=("was_on_bench_clean", "sum"),
            not_in_squad=("was_not_in_squad_clean", "sum"),
            injury_or_fitness_absences=("absence_reason", lambda s: int((s == "injury_or_fitness").sum())),
            suspension_absences=("absence_reason", lambda s: int((s == "suspension").sum())),
            not_selected_or_unknown_absences=("absence_reason", lambda s: int((s == "not_selected_or_unknown").sum())),
            unused_in_squad=("unused_in_squad", "sum"),
            selection_observations=("selection_observation", "sum"),
            recency_weight_sum=("recency_weight", "sum"),
            recency_selection_weight_sum=("selection_signal_weight", "sum"),
            recency_start_numerator=("start_signal_numerator", "sum"),
            recent_weighted_starts=("recent_weighted_start", "sum"),
            recent_weighted_bench=("recent_weighted_bench", "sum"),
            recent_weighted_not_in_squad=("recent_weighted_not_in_squad", "sum"),
        )
        .reset_index()
    )

    summary["start_share"] = (
        summary["starts"] / summary["selection_observations"].replace(0, pd.NA)
    ).fillna(0).round(3)
    summary["bench_share"] = (summary["bench"] / summary["rows"]).round(3)
    summary["not_in_squad_share"] = (summary["not_in_squad"] / summary["rows"]).round(3)

    summary["recency_weighted_start_share"] = (
        summary["recency_start_numerator"] / summary["recency_selection_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    summary["recency_weighted_bench_share"] = (
        summary["recent_weighted_bench"] / summary["recency_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    summary["recency_weighted_not_in_squad_share"] = (
        summary["recent_weighted_not_in_squad"] / summary["recency_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    summary["competition_weight"] = summary["competition_category"].map(COMPETITION_WEIGHTS).fillna(
        COMPETITION_WEIGHTS["ukendt"]
    )
    summary["weighted_start_signal"] = (
        summary["start_share"] * summary["competition_weight"]
    ).round(4)

    summary["recency_competition_weighted_start_signal"] = (
        summary["recency_weighted_start_share"] * summary["competition_weight"]
    ).round(4)

    return summary.sort_values(
        ["player_name", "competition_weight", "rows"],
        ascending=[True, False, False],
    )


def build_player_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    player_cols = ["player_id", "player_name", "team_id", "position_model"]
    rows = []
    for keys, group in df.groupby(player_cols, dropna=False):
        total_starts = float(group["started_estimate_clean"].astype(float).sum())
        selection_observations = float(group["selection_observation"].sum())
        weighted_numerator = float(group["start_signal_numerator"].sum())
        weighted_denominator = float(group["selection_signal_weight"].sum())
        historical_score = total_starts / selection_observations if selection_observations else 0.0
        full_recency_score = weighted_numerator / weighted_denominator if weighted_denominator else 0.0

        recent_available = (
            group.loc[group["selection_observation"].gt(0)]
            .sort_values("date_parsed", ascending=False)
            .head(3)
        )
        recent_denominator = float(recent_available["selection_signal_weight"].sum())
        recent_numerator = float(recent_available["start_signal_numerator"].sum())
        recent_three_score = (
            recent_numerator / recent_denominator
            if recent_denominator
            else full_recency_score
        )
        blended_recency_score = 0.70 * recent_three_score + 0.30 * full_recency_score

        rows.append(
            {
                **dict(zip(player_cols, keys)),
                "tm_competition_rows": len(group),
                "tm_competition_starts": int(total_starts),
                "tm_selection_observations": selection_observations,
                "tm_unused_in_squad": int(group["unused_in_squad"].sum()),
                "tm_weighted_start_signal_sum": total_starts,
                "tm_recency_competition_weighted_start_signal_sum": weighted_numerator,
                "tm_competition_weight_sum": weighted_denominator,
                "tm_weighted_competitive_start_score": round(historical_score, 4),
                "tm_full_history_recency_start_score": round(full_recency_score, 4),
                "tm_recent_3_available_start_score": round(recent_three_score, 4),
                "tm_recency_weighted_competitive_start_score": round(blended_recency_score, 4),
            }
        )

    return pd.DataFrame(rows)


def load_before_state() -> tuple[dict[str, int], dict[str, Any]]:
    category_counts: dict[str, int] = {}
    rudiger: dict[str, Any] = {}

    if AUDIT_CSV.exists():
        previous_audit = pd.read_csv(AUDIT_CSV, low_memory=False)
        category_rows = previous_audit.loc[previous_audit["record_type"].eq("category_count")]
        if not category_rows.empty:
            category_counts = {
                str(row["key"]): int(number)
                for _, row in category_rows.iterrows()
                if pd.notna(number := pd.to_numeric(row.get("before_count"), errors="coerce"))
            }
        signal_rows = previous_audit.loc[previous_audit["record_type"].eq("rudiger_signal")]
        if not signal_rows.empty:
            rudiger["tm_recency_weighted_competitive_start_score"] = signal_rows.iloc[0].get(
                "before_value", ""
            )

    if not category_counts and OUT_MATCHES.exists():
        before_matches = pd.read_csv(
            OUT_MATCHES,
            usecols=lambda column: column in {"competition_category_label"},
            low_memory=False,
        )
        if "competition_category_label" in before_matches:
            category_counts = {
                str(key): int(value)
                for key, value in before_matches["competition_category_label"]
                .fillna("Ukendt")
                .value_counts()
                .items()
            }

    if not rudiger and OUT_PLAYER_SUMMARY.exists():
        before_player = pd.read_csv(OUT_PLAYER_SUMMARY, low_memory=False)
        rows = before_player.loc[
            before_player.get("player_id", pd.Series("", index=before_player.index))
            .astype(str)
            .eq("antonio_r_diger__ger")
        ]
        if not rows.empty:
            rudiger = rows.iloc[0].to_dict()

    return category_counts, rudiger


def audit_outputs(
    df: pd.DataFrame,
    player_summary: pd.DataFrame,
    before_counts: dict[str, int],
    rudiger_before: dict[str, Any],
) -> None:
    after_counts = {
        str(key): int(value)
        for key, value in df["competition_category_label"].fillna("Ukendt").value_counts().items()
    }
    total = len(df)
    before_unknown = before_counts.get("Ukendt", 0)
    after_unknown = after_counts.get("Ukendt", 0)

    audit_rows: list[dict[str, Any]] = []
    for label in sorted(set(before_counts) | set(after_counts)):
        audit_rows.append(
            {
                "record_type": "category_count",
                "key": label,
                "description": "",
                "category": label,
                "before_count": before_counts.get(label, 0),
                "after_count": after_counts.get(label, 0),
                "row_count": after_counts.get(label, 0),
                "share": after_counts.get(label, 0) / total if total else 0.0,
            }
        )

    code_counts = df["competition_raw"].fillna("").astype(str).str.upper().value_counts()
    for code, count in code_counts.items():
        mapped = COMPETITION_CODE_MAP.get(code)
        sample = df.loc[df["competition_raw"].astype(str).str.upper().eq(code)].iloc[0]
        category, label = classify_competition(sample)
        audit_rows.append(
            {
                "record_type": "competition_mapping",
                "key": code,
                "description": mapped[2] if mapped else "Classified by general family rule or unknown fallback",
                "category": label,
                "before_count": count if before_unknown else 0,
                "after_count": count,
                "row_count": count,
                "share": count / total if total else 0.0,
            }
        )

    rudiger_matches = df.loc[df["player_id"].astype(str).eq("antonio_r_diger__ger")].copy()
    rudiger_matches = rudiger_matches.sort_values("date_parsed", ascending=False).head(10)
    for _, row in rudiger_matches.iterrows():
        audit_rows.append(
            {
                "record_type": "rudiger_recent_match",
                "key": row.get("date"),
                "description": row.get("participation_state"),
                "category": row.get("competition_category_label"),
                "competition_code": row.get("competition_raw"),
                "date": row.get("date"),
                "participation_state": row.get("participation_state"),
                "minutes_estimate": row.get("minutes_estimate"),
                "started": row.get("started_estimate_clean"),
                "recency_weight": row.get("recency_weight"),
            }
        )

    rudiger_after_rows = player_summary.loc[
        player_summary["player_id"].astype(str).eq("antonio_r_diger__ger")
    ]
    rudiger_after = rudiger_after_rows.iloc[0].to_dict() if not rudiger_after_rows.empty else {}
    audit_rows.append(
        {
            "record_type": "rudiger_signal",
            "key": "tm_recency_weighted_competitive_start_score",
            "description": "Before/after player-level recency and competition weighted start score",
            "before_value": rudiger_before.get("tm_recency_weighted_competitive_start_score", ""),
            "after_value": rudiger_after.get("tm_recency_weighted_competitive_start_score", ""),
            "full_history_recency_value": rudiger_after.get("tm_full_history_recency_start_score", ""),
            "recent_3_available_value": rudiger_after.get("tm_recent_3_available_start_score", ""),
        }
    )

    june = df.loc[
        df["date_parsed"].between(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30"))
        & df["player_id"].astype(str).ne("antonio_r_diger__ger")
    ].copy()
    june_latest = (
        june.sort_values("date_parsed", ascending=False)
        .drop_duplicates("player_id")
        .head(5)
    )
    player_scores = player_summary.set_index("player_id")[
        "tm_recency_weighted_competitive_start_score"
    ].to_dict()
    for _, row in june_latest.iterrows():
        audit_rows.append(
            {
                "record_type": "june_2026_sanity",
                "key": row.get("player_id"),
                "description": row.get("player_name"),
                "category": row.get("competition_category_label"),
                "competition_code": row.get("competition_raw"),
                "date": row.get("date"),
                "participation_state": row.get("participation_state"),
                "minutes_estimate": row.get("minutes_estimate"),
                "started": row.get("started_estimate_clean"),
                "after_value": player_scores.get(str(row.get("player_id")), ""),
            }
        )

    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

    def md_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
        if frame.empty:
            return ["(ingen)"]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row.get(column, "")
                if pd.isna(value):
                    value = ""
                elif isinstance(value, float):
                    value = f"{value:.4f}".rstrip("0").rstrip(".")
                values.append(str(value).replace("|", "/"))
            lines.append("| " + " | ".join(values) + " |")
        return lines

    category_frame = pd.DataFrame(
        [
            {
                "kategori": label,
                "før": before_counts.get(label, 0),
                "efter": after_counts.get(label, 0),
            }
            for label in sorted(set(before_counts) | set(after_counts))
        ]
    )
    mapping_frame = pd.DataFrame(
        [
            {
                "kode": code,
                "rækker": int(count),
                "kategori": classify_competition(
                    df.loc[df["competition_raw"].astype(str).str.upper().eq(code)].iloc[0]
                )[1],
                "mapping": COMPETITION_CODE_MAP.get(code, ("", "", "General family rule / unknown"))[2],
            }
            for code, count in code_counts.items()
        ]
    )
    unknown_frame = mapping_frame.loc[mapping_frame["kategori"].eq("Ukendt")].head(15)
    rudiger_frame = rudiger_matches[
        [
            "date",
            "competition_raw",
            "competition_category_label",
            "participation_state",
            "minutes_estimate",
            "started_estimate_clean",
            "recency_weight",
        ]
    ].copy()
    june_frame = pd.DataFrame(
        [
            {
                "spiller": row.get("player_name"),
                "dato": row.get("date"),
                "kode": row.get("competition_raw"),
                "kategori": row.get("competition_category_label"),
                "status": row.get("participation_state"),
                "start": row.get("started_estimate_clean"),
                "recency_score": player_scores.get(str(row.get("player_id")), ""),
            }
            for _, row in june_latest.iterrows()
        ]
    )

    lines = [
        "# Transfermarkt Competition Classifier Audit",
        "",
        "## Rodårsag",
        "",
        "Det nye CEAPI-format leverer korte turneringskoder i `competition`. Den tidligere classifier søgte kun engelske turneringsnavne i rækketeksten, så alle koder faldt gennem til `Ukendt`.",
        "",
        "Derudover blev historiske to-cifrede årstal som `62` fortolket som 2062. Det skubbede reference-datoen ud i fremtiden og gjorde 2026-kampe kunstigt gamle.",
        "",
        "## Før/Efter",
        "",
        *md_table(category_frame, ["kategori", "før", "efter"]),
        "",
        f"- Ukendt før: {before_unknown}/{total} ({before_unknown / total:.2%})",
        f"- Ukendt efter: {after_unknown}/{total} ({after_unknown / total:.2%})",
        "",
        "## Konkurrencekoder",
        "",
        *md_table(mapping_frame, ["kode", "rækker", "kategori", "mapping"]),
        "",
        "## Top ukendte koder",
        "",
        *md_table(unknown_frame, ["kode", "rækker", "kategori", "mapping"]),
        "",
        "## Antonio Rüdiger",
        "",
        f"- Recency/startsignal før: {rudiger_before.get('tm_recency_weighted_competitive_start_score', '')}",
        f"- Recency/startsignal efter: {rudiger_after.get('tm_recency_weighted_competitive_start_score', '')}",
        f"- Fuld recency-historik efter: {rudiger_after.get('tm_full_history_recency_start_score', '')}",
        f"- Tre seneste tilgængelige observationer: {rudiger_after.get('tm_recent_3_available_start_score', '')}",
        "- Endeligt recency/startsignal = 70% seneste tre tilgængelige observationer + 30% fuld recency-/konkurrencehistorik.",
        "- `injured` er udelukket fra udvælgelsesnævneren. `in squad` uden minutter er en tilgængelig nul-start-observation.",
        "",
        *md_table(
            rudiger_frame,
            [
                "date",
                "competition_raw",
                "competition_category_label",
                "participation_state",
                "minutes_estimate",
                "started_estimate_clean",
                "recency_weight",
            ],
        ),
        "",
        "## Juni 2026 sanity",
        "",
        *md_table(
            june_frame,
            ["spiller", "dato", "kode", "kategori", "status", "start", "recency_score"],
        ),
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not TM_DIR.exists():
        raise FileNotFoundError(f"Mangler mappe: {TM_DIR}")

    before_counts, rudiger_before = load_before_state()
    df = load_match_files()

    print("TRANSFERMARKT COMPETITION CLASSIFIER")
    print("=" * 80)
    print(f"Match-rækker læst: {len(df)}")

    if df.empty:
        print("Ingen matchfiler fundet endnu.")
        return

    categories = []
    labels = []
    started_clean = []
    bench_clean = []
    nis_clean = []
    absence_reason = []
    absence_reason_label = []

    for _, row in df.iterrows():
        category, label = classify_competition(row)
        categories.append(category)
        labels.append(label)

        started, bench, nis = infer_appearance_status(row)
        started_clean.append(started)
        bench_clean.append(bench)
        nis_clean.append(nis)

        absence_code, absence_label = classify_absence_reason(row)
        absence_reason.append(absence_code)
        absence_reason_label.append(absence_label)

    df["competition_category"] = categories
    df["competition_category_label"] = labels
    df["started_estimate_clean"] = started_clean
    df["was_on_bench_clean"] = bench_clean
    df["was_not_in_squad_clean"] = nis_clean
    df["absence_reason"] = absence_reason
    df["absence_reason_label"] = absence_reason_label

    # Robust sikkerhedsregel:
    # Rækker med skade, operation, karantæne/suspension eller "Not in squad"
    # må aldrig tælle som startere, selv hvis Transfermarkt-tabellen har tekst i
    # positionskolonnen. "On the bench" må heller ikke tælle som start.
    status_text = pd.Series("", index=df.index, dtype="string")

    for status_col in ["position", "row_text", "participation_state"]:
        if status_col in df.columns:
            status_text = (status_text.fillna("") + " " + df[status_col].astype(str).str.lower()).astype("string")

    injury_pattern = (
        r"injury|injured|hamstring|ankle|foot|knee|groin|pubalgia|surgery|"
        r"fitness|illness|ill|muscle|strain|knock|problems"
    )
    suspension_pattern = r"suspension|suspended|yellow card suspension|red card suspension|indirect card suspension"
    not_in_squad_pattern = r"not in squad"
    bench_pattern = r"on the bench"

    injury_mask = status_text.str.contains(injury_pattern, regex=True, na=False)
    suspension_mask = status_text.str.contains(suspension_pattern, regex=True, na=False)
    not_in_squad_text_mask = status_text.str.contains(not_in_squad_pattern, regex=True, na=False)
    bench_text_mask = status_text.str.contains(bench_pattern, regex=True, na=False)

    state_text = df.get("participation_state", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    injury_mask |= state_text.eq("injured")
    suspension_mask |= state_text.eq("suspended")
    in_squad_mask = state_text.eq("in squad")
    hard_absence_mask = injury_mask | suspension_mask | not_in_squad_text_mask

    # Bænk: ikke starter, men heller ikke "not in squad".
    if "started_estimate_clean" in df.columns:
        df.loc[bench_text_mask, "started_estimate_clean"] = False
    if "was_on_bench_clean" in df.columns:
        df.loc[bench_text_mask, "was_on_bench_clean"] = True
    if "was_not_in_squad_clean" in df.columns:
        df.loc[bench_text_mask, "was_not_in_squad_clean"] = False
    if "minutes_estimate" in df.columns:
        df.loc[bench_text_mask, "minutes_estimate"] = 0.0

    # "In squad" without minutes is an available, unused non-start observation.
    if "started_estimate_clean" in df.columns:
        df.loc[in_squad_mask, "started_estimate_clean"] = False
    if "was_on_bench_clean" in df.columns:
        df.loc[in_squad_mask, "was_on_bench_clean"] = False
    if "was_not_in_squad_clean" in df.columns:
        df.loc[in_squad_mask, "was_not_in_squad_clean"] = False

    # Injury/suspension is unavailable, not an ordinary bench/non-selection signal.
    if "started_estimate_clean" in df.columns:
        df.loc[hard_absence_mask, "started_estimate_clean"] = False
    if "was_on_bench_clean" in df.columns:
        df.loc[hard_absence_mask, "was_on_bench_clean"] = False
    if "was_not_in_squad_clean" in df.columns:
        df.loc[injury_mask | suspension_mask, "was_not_in_squad_clean"] = False
        df.loc[not_in_squad_text_mask & ~injury_mask & ~suspension_mask, "was_not_in_squad_clean"] = True
    if "minutes_estimate" in df.columns:
        df.loc[hard_absence_mask, "minutes_estimate"] = 0.0
    if "has_position" in df.columns:
        df.loc[hard_absence_mask, "has_position"] = False

    if "absence_reason" in df.columns:
        df.loc[injury_mask, "absence_reason"] = "injury_or_fitness"
        df.loc[suspension_mask, "absence_reason"] = "suspension"
        df.loc[not_in_squad_text_mask & ~injury_mask & ~suspension_mask, "absence_reason"] = "not_selected_or_unknown"

    if "absence_reason_label" in df.columns:
        df.loc[injury_mask, "absence_reason_label"] = "Skade/fitness"
        df.loc[suspension_mask, "absence_reason_label"] = "Karantæne/suspension"
        df.loc[not_in_squad_text_mask & ~injury_mask & ~suspension_mask, "absence_reason_label"] = "Ikke i trup/ukendt"

    df = add_recency_weights(df)

    df.to_csv(OUT_MATCHES, index=False, encoding="utf-8-sig")

    summary = build_summary(df)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    player_summary = build_player_level_summary(df)
    player_summary.to_csv(OUT_PLAYER_SUMMARY, index=False, encoding="utf-8-sig")
    audit_outputs(df, player_summary, before_counts, rudiger_before)

    print("")
    print("Kategorier:")
    print(df["competition_category_label"].value_counts(dropna=False).to_string())

    print("")
    print("Top preview:")
    preview_cols = [
        "player_name",
        "team_id",
        "competition_category_label",
        "rows",
        "starts",
        "start_share",
        "weighted_start_signal",
    ]
    print(summary[preview_cols].head(40).to_string(index=False))

    print("")
    print(f"Skrev: {OUT_MATCHES}")
    print(f"Skrev: {OUT_SUMMARY}")
    print(f"Skrev: {OUT_PLAYER_SUMMARY}")
    print(f"Skrev: {AUDIT_CSV}")
    print(f"Skrev: {AUDIT_MD}")


if __name__ == "__main__":
    main()
