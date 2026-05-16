from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

API_URL_TEMPLATE = "https://nexus-app-fantasy-fargate.holdet.dk/api/games/{game_id}/players"

POSITION_MAP = {
    "keeper": "GK",
    "goalkeeper": "GK",
    "goalie": "GK",
    "gk": "GK",
    "defense": "DEF",
    "defender": "DEF",
    "defence": "DEF",
    "back": "DEF",
    "midfield": "MID",
    "midfielder": "MID",
    "mid": "MID",
    "attack": "FWD",
    "attacker": "FWD",
    "forward": "FWD",
    "striker": "FWD",
    "fwd": "FWD",
}


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def get_person_name(person: dict[str, Any]) -> str | None:
    direct = first_non_empty(
        person.get("name"),
        person.get("fullName"),
        person.get("full_name"),
        person.get("displayName"),
        person.get("display_name"),
        person.get("shortName"),
        person.get("short_name"),
        person.get("knownName"),
        person.get("known_name"),
        person.get("commonName"),
        person.get("common_name"),
    )
    if direct:
        return direct

    first = first_non_empty(
        person.get("firstName"),
        person.get("first_name"),
        person.get("firstname"),
        person.get("givenName"),
        person.get("given_name"),
    )
    last = first_non_empty(
        person.get("lastName"),
        person.get("last_name"),
        person.get("lastname"),
        person.get("familyName"),
        person.get("family_name"),
        person.get("surname"),
    )

    if first and last:
        return f"{first} {last}"
    if last:
        return last
    if first:
        return first

    return None


def get_generic_name(obj: dict[str, Any]) -> str | None:
    return first_non_empty(
        obj.get("name"),
        obj.get("fullName"),
        obj.get("full_name"),
        obj.get("displayName"),
        obj.get("display_name"),
        obj.get("shortName"),
        obj.get("short_name"),
        obj.get("title"),
        obj.get("label"),
    )


def normalize_position(position_raw: Any) -> str | None:
    if position_raw is None:
        return None
    text = str(position_raw).strip()
    if not text:
        return None
    return POSITION_MAP.get(text.lower(), text)


def fetch_holdet_players(game_id: int) -> dict[str, Any]:
    url = API_URL_TEMPLATE.format(game_id=game_id)
    response = requests.get(
        url,
        timeout=30,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )
    response.raise_for_status()
    return response.json()


def flatten_holdet_payload(payload: dict[str, Any]) -> pd.DataFrame:
    items = payload.get("items", [])
    embedded = payload.get("_embedded", {})

    persons = embedded.get("persons", {}) or {}
    teams = embedded.get("teams", {}) or {}
    positions = embedded.get("positions", {}) or {}

    rows: list[dict[str, Any]] = []

    for item in items:
        person_id = str(item.get("personId", ""))
        team_id = str(item.get("teamId", ""))
        position_id = str(item.get("positionId", ""))

        person = persons.get(person_id, {}) or {}
        team = teams.get(team_id, {}) or {}
        position = positions.get(position_id, {}) or {}

        player_name = get_person_name(person)
        team_name = get_generic_name(team)
        position_name = get_generic_name(position)

        raw_position = first_non_empty(
            item.get("position"),
            position_name,
            position.get("type"),
            position.get("slug"),
        )

        rows.append(
            {
                "holdet_player_id": item.get("id"),
                "holdet_person_id": item.get("personId"),
                "player_name": player_name,
                "holdet_team_id": item.get("teamId"),
                "team_name": team_name,
                "holdet_position_id": item.get("positionId"),
                "position_raw": raw_position,
                "position": normalize_position(raw_position),
                "start_price": item.get("startPrice"),
                "price": item.get("price"),
                "points": item.get("points"),
                "popularity": item.get("popularity"),
                "is_out": item.get("isOut"),
                "person_raw_keys": "|".join(sorted(person.keys())) if isinstance(person, dict) else "",
                "team_raw_keys": "|".join(sorted(team.keys())) if isinstance(team, dict) else "",
                "position_raw_keys": "|".join(sorted(position.keys())) if isinstance(position, dict) else "",
            }
        )

    df = pd.DataFrame(rows)

    preferred_cols = [
        "holdet_player_id",
        "holdet_person_id",
        "player_name",
        "holdet_team_id",
        "team_name",
        "holdet_position_id",
        "position_raw",
        "position",
        "start_price",
        "price",
        "points",
        "popularity",
        "is_out",
        "person_raw_keys",
        "team_raw_keys",
        "position_raw_keys",
    ]

    existing_cols = [col for col in preferred_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_cols]

    return df[existing_cols + other_cols]


def print_payload_summary(payload: dict[str, Any], df: pd.DataFrame) -> None:
    print("TOP-LEVEL:")
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"{key}: list len = {len(value)}")
            if value and isinstance(value[0], dict):
                print(f"  first item keys = {list(value[0].keys())}")
        elif isinstance(value, dict):
            print(f"{key}: dict with keys = {list(value.keys())[:20]}")
        else:
            print(f"{key}: {type(value).__name__}")

    embedded = payload.get("_embedded", {})
    if isinstance(embedded, dict):
        print("\n_embedded:")
        for key, value in embedded.items():
            if isinstance(value, dict):
                print(f"  {key}: dict with keys = {list(value.keys())[:20]}")
            else:
                print(f"  {key}: {type(value).__name__}")

    print("\nFlat kolonner:")
    print(list(df.columns))

    print("\nFørste 20 rækker:")
    if df.empty:
        print("(ingen rækker)")
    else:
        print(df.head(20).to_string(index=False))

    print("\nDatakvalitet:")
    if "player_name" in df.columns:
        print(f"Mangler player_name: {int(df['player_name'].isna().sum())} / {len(df)}")
    if "team_name" in df.columns:
        print(f"Mangler team_name: {int(df['team_name'].isna().sum())} / {len(df)}")
    if "position" in df.columns:
        print(f"Mangler position: {int(df['position'].isna().sum())} / {len(df)}")

    if "position" in df.columns:
        print("\nPositioner:")
        print(df["position"].value_counts(dropna=False).to_string())

    if "team_name" in df.columns:
        print("\nAntal hold:")
        print(df["team_name"].nunique(dropna=True))

    if "is_out" in df.columns:
        print("\nAktive/deaktiverede:")
        print(df["is_out"].value_counts(dropna=False).to_string())

    if "person_raw_keys" in df.columns:
        print("\nPerson-key eksempel:")
        examples = df["person_raw_keys"].dropna().astype(str)
        examples = examples[examples != ""].drop_duplicates().head(5)
        if len(examples) == 0:
            print("(ingen person keys fundet)")
        else:
            for value in examples:
                print(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=int, required=True)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    game_id = args.game_id

    raw_json_path = DATA_DIR / f"holdet_players_game_{game_id}_raw.json"
    flat_csv_path = DATA_DIR / f"holdet_players_game_{game_id}_flat.csv"
    flat_json_path = DATA_DIR / f"holdet_players_game_{game_id}_flat.json"

    payload = fetch_holdet_players(game_id)
    df = flatten_holdet_payload(payload)

    raw_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df.to_csv(flat_csv_path, index=False, encoding="utf-8-sig")

    flat_json_path.write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_payload_summary(payload, df)

    print("\nFiler skrevet:")
    print(f"Raw JSON:  {raw_json_path}")
    print(f"Flat CSV:  {flat_csv_path}")
    print(f"Flat JSON: {flat_json_path}")


if __name__ == "__main__":
    main()