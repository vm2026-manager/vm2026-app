from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TM_DIR = DATA_DIR / "transfermarkt_national_team"

OUT_MATCHES = TM_DIR / "player_national_team_matches_classified.csv"
OUT_SUMMARY = TM_DIR / "player_national_team_usage_by_competition.csv"


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
    text = clean_text(row.get("row_text", ""))

    if "not in squad" not in text and "on the bench" not in text:
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
    text = clean_text(row.get("row_text", ""))

    was_not_in_squad = bool(row.get("was_not_in_squad", False))
    was_on_bench = bool(row.get("was_on_bench", False))
    started = bool(row.get("started_estimate", False))

    if "not in squad" in text:
        was_not_in_squad = True
    if "on the bench" in text:
        was_on_bench = True

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

    max_date = out["date_parsed"].max()
    if pd.isna(max_date):
        out["months_ago"] = None
        out["recency_weight"] = 0.25
        return out

    out["months_ago"] = ((max_date - out["date_parsed"]).dt.days / 30.44).round(1)

    def weight(months: float) -> float:
        if pd.isna(months):
            return 0.25
        if months <= 6:
            return 1.00
        if months <= 12:
            return 0.85
        if months <= 24:
            return 0.65
        if months <= 36:
            return 0.45
        return 0.25

    out["recency_weight"] = out["months_ago"].map(weight)

    # Bruges i summeringer:
    out["recent_weighted_start"] = out["started_estimate_clean"].astype(float) * out["recency_weight"]
    out["recent_weighted_bench"] = out["was_on_bench_clean"].astype(float) * out["recency_weight"]
    out["recent_weighted_not_in_squad"] = out["was_not_in_squad_clean"].astype(float) * out["recency_weight"]

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
            recency_weight_sum=("recency_weight", "sum"),
            recent_weighted_starts=("recent_weighted_start", "sum"),
            recent_weighted_bench=("recent_weighted_bench", "sum"),
            recent_weighted_not_in_squad=("recent_weighted_not_in_squad", "sum"),
        )
        .reset_index()
    )

    summary["start_share"] = (summary["starts"] / summary["rows"]).round(3)
    summary["bench_share"] = (summary["bench"] / summary["rows"]).round(3)
    summary["not_in_squad_share"] = (summary["not_in_squad"] / summary["rows"]).round(3)

    summary["recency_weighted_start_share"] = (
        summary["recent_weighted_starts"] / summary["recency_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    summary["recency_weighted_bench_share"] = (
        summary["recent_weighted_bench"] / summary["recency_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    summary["recency_weighted_not_in_squad_share"] = (
        summary["recent_weighted_not_in_squad"] / summary["recency_weight_sum"].replace(0, pd.NA)
    ).fillna(0).round(3)

    # En foreløbig vægtet usage-score:
    # Slutrunder og kval vægter højest. Venskabskampe lavere.
    weight_map = {
        "slutrunde": 1.00,
        "kvalifikation": 0.85,
        "nations_league": 0.65,
        "uklar_turnering": 0.60,
        "venskabskamp": 0.35,
        "ukendt": 0.45,
    }

    summary["competition_weight"] = summary["competition_category"].map(weight_map).fillna(0.45)
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


def build_player_level_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    player_cols = ["player_id", "player_name", "team_id", "position_model"]

    player = (
        summary.groupby(player_cols, dropna=False)
        .agg(
            tm_competition_rows=("rows", "sum"),
            tm_competition_starts=("starts", "sum"),
            tm_weighted_start_signal_sum=("weighted_start_signal", "sum"),
            tm_recency_competition_weighted_start_signal_sum=("recency_competition_weighted_start_signal", "sum"),
            tm_competition_weight_sum=("competition_weight", "sum"),
        )
        .reset_index()
    )

    player["tm_weighted_competitive_start_score"] = (
        player["tm_weighted_start_signal_sum"] / player["tm_competition_weight_sum"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0).round(4)

    player["tm_recency_weighted_competitive_start_score"] = (
        player["tm_recency_competition_weighted_start_signal_sum"] / player["tm_competition_weight_sum"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0).round(4)

    return player


def main() -> None:
    if not TM_DIR.exists():
        raise FileNotFoundError(f"Mangler mappe: {TM_DIR}")

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

    for status_col in ["position", "row_text"]:
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

    if "absence_reason" in df.columns:
        absence_reason_text = df["absence_reason"].astype(str).str.lower()
        absence_reason_mask = absence_reason_text.isin(
            ["injury_or_fitness", "suspension", "not_selected_or_unknown"]
        )
    else:
        absence_reason_mask = pd.Series(False, index=df.index)

    hard_absence_mask = injury_mask | suspension_mask | not_in_squad_text_mask | absence_reason_mask

    # Bænk: ikke starter, men heller ikke "not in squad".
    if "started_estimate_clean" in df.columns:
        df.loc[bench_text_mask, "started_estimate_clean"] = False
    if "was_on_bench_clean" in df.columns:
        df.loc[bench_text_mask, "was_on_bench_clean"] = True
    if "was_not_in_squad_clean" in df.columns:
        df.loc[bench_text_mask, "was_not_in_squad_clean"] = False
    if "minutes_estimate" in df.columns:
        df.loc[bench_text_mask, "minutes_estimate"] = 0.0

    # Hårdt fravær: ikke starter, ikke bænk, ikke i trup.
    if "started_estimate_clean" in df.columns:
        df.loc[hard_absence_mask, "started_estimate_clean"] = False
    if "was_on_bench_clean" in df.columns:
        df.loc[hard_absence_mask, "was_on_bench_clean"] = False
    if "was_not_in_squad_clean" in df.columns:
        df.loc[hard_absence_mask, "was_not_in_squad_clean"] = True
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

    player_summary = build_player_level_summary(summary)
    player_summary_path = TM_DIR / "player_national_team_usage_competitive_summary.csv"
    player_summary.to_csv(player_summary_path, index=False, encoding="utf-8-sig")

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
    print(f"Skrev: {player_summary_path}")


if __name__ == "__main__":
    main()