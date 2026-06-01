from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLAYER_POOL_PATH = DATA_DIR / "player_pool_v1.json"
PREFERRED_HOLDET_FILE = DATA_DIR / "holdet_players_game_616_flat.csv"
LOCAL_FALLBACK_DATA_DIR = Path(r"C:\Users\Administrator\Desktop\vm2026_local_app\data")

NEW_PLAYERS_REVIEW_PATH = DATA_DIR / "holdet_new_players_review.csv"
MATCH_REPORT_PATH = DATA_DIR / "holdet_existing_players_match_report.csv"
MISMATCH_REPORT_PATH = DATA_DIR / "holdet_possible_name_or_team_mismatches.csv"

POSITION_MAP = {
    "keeper": "GK",
    "goalkeeper": "GK",
    "goalie": "GK",
    "gk": "GK",
    "defense": "DEF",
    "defender": "DEF",
    "defence": "DEF",
    "back": "DEF",
    "forsvar": "DEF",
    "midfield": "MID",
    "midfielder": "MID",
    "mid": "MID",
    "midtbanespiller": "MID",
    "midtbane": "MID",
    "attack": "FWD",
    "attacker": "FWD",
    "forward": "FWD",
    "striker": "FWD",
    "fwd": "FWD",
    "angriber": "FWD",
}

POSITION_OUTPUT = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

TEAM_ALIASES = {
    "HOLDET_584": "CZE",
    "HOLDET_767": "CIV",
}

TEAM_NAME_TO_CANONICAL = {
    "algeriet": "ALG",
    "argentina": "ARG",
    "australien": "AUS",
    "belgien": "BEL",
    "bosnien hercegovina": "BIH",
    "brasilien": "BRA",
    "canada": "CAN",
    "colombia": "COL",
    "congo dr": "COD",
    "dr congo": "COD",
    "curacao": "CUW",
    "ecuador": "ECU",
    "egypten": "EGY",
    "elfenbenskysten": "CIV",
    "england": "ENG",
    "frankrig": "FRA",
    "ghana": "GHA",
    "haiti": "HAI",
    "holland": "NED",
    "irak": "IRQ",
    "iran": "IRN",
    "japan": "JPN",
    "jordan": "JOR",
    "kap verde": "CPV",
    "kroatien": "CRO",
    "marokko": "MAR",
    "mexico": "MEX",
    "new zealand": "NZL",
    "norge": "NOR",
    "panama": "PAN",
    "paraguay": "PAR",
    "portugal": "POR",
    "qatar": "QAT",
    "saudi arabien": "KSA",
    "schweiz": "SUI",
    "senegal": "SEN",
    "skotland": "SCO",
    "spanien": "ESP",
    "sverige": "SWE",
    "sydafrika": "RSA",
    "sydkorea": "KOR",
    "tjekkiet": "CZE",
    "tunesien": "TUN",
    "tyrkiet": "TUR",
    "tyskland": "GER",
    "uruguay": "URU",
    "usa": "USA",
    "usbekistan": "UZB",
    "ostrig": "AUT",
}

HOLDET_TEAM_ID_TO_CANONICAL = {
    "761": "ALG",
    "762": "ARG",
    "763": "AUS",
    "884": "BEL",
    "882": "BIH",
    "764": "BRA",
    "1059": "CAN",
    "879": "COL",
    "1145": "COD",
    "1128": "CUW",
    "881": "ECU",
    "973": "EGY",
    "767": "CIV",
    "769": "ENG",
    "592": "FRA",
    "770": "GHA",
    "1129": "HAI",
    "593": "NED",
    "947": "IRQ",
    "883": "IRN",
    "772": "JPN",
    "1123": "JOR",
    "1125": "CPV",
    "588": "CRO",
    "974": "MAR",
    "775": "MEX",
    "776": "NZL",
    "1126": "NOR",
    "971": "PAN",
    "778": "PAR",
    "585": "POR",
    "1009": "QAT",
    "977": "KSA",
    "583": "SUI",
    "975": "SEN",
    "1022": "SCO",
    "595": "ESP",
    "598": "SWE",
    "782": "RSA",
    "774": "KOR",
    "584": "CZE",
    "976": "TUN",
    "586": "TUR",
    "589": "GER",
    "784": "USA",
    "783": "URU",
    "1122": "UZB",
    "587": "AUT",
}


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm_text(value: Any) -> str:
    text = txt(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = txt(value)
        if text:
            return text
    return ""


def normalize_position(value: Any) -> str:
    text = txt(value)
    if not text:
        return ""
    return POSITION_OUTPUT.get(text.upper(), POSITION_MAP.get(text.lower(), text.upper()))


def normalize_team(value: Any) -> str:
    raw = txt(value)
    if not raw:
        return ""
    upper = raw.upper()
    return TEAM_ALIASES.get(upper, upper)


def canonical_holdet_team(row: dict[str, Any]) -> str:
    team_name = first_non_empty(row.get("team_name"), row.get("team"), row.get("country"))
    by_name = TEAM_NAME_TO_CANONICAL.get(norm_text(team_name))
    if by_name:
        return by_name

    holdet_team_id = first_non_empty(row.get("holdet_team_id"), row.get("team_id"))
    by_id = HOLDET_TEAM_ID_TO_CANONICAL.get(txt(holdet_team_id))
    if by_id:
        return by_id

    return normalize_team(first_non_empty(row.get("team_id"), row.get("holdet_team_id"), row.get("country_code"), team_name))


def parse_price(value: Any) -> int:
    text = txt(value)
    if not text:
        return 0
    cleaned = re.sub(r"[^\d]", "", text)
    return int(cleaned) if cleaned else 0


def get_person_name(person: dict[str, Any]) -> str:
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
    return " ".join(part for part in [first, last] if part)


def get_generic_name(obj: dict[str, Any]) -> str:
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


def find_holdet_source_file() -> Path:
    if PREFERRED_HOLDET_FILE.exists():
        return PREFERRED_HOLDET_FILE

    local_preferred = LOCAL_FALLBACK_DATA_DIR / PREFERRED_HOLDET_FILE.name
    if local_preferred.exists():
        return local_preferred

    candidates = [
        path
        for search_dir in [DATA_DIR, LOCAL_FALLBACK_DATA_DIR]
        if search_dir.exists()
        for path in search_dir.glob("holdet_*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json"}
        and "player" in path.name.lower()
        and not path.name.endswith("_review.csv")
        and "match_report" not in path.name.lower()
        and "mismatch" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Ingen Holdet.dk spillerfil fundet i data/. Kør først: python tools/holdet_players_api.py --game-id 616"
        )

    def priority(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        if "flat" in name and path.suffix.lower() == ".csv":
            rank = 3
        elif "flat" in name and path.suffix.lower() == ".json":
            rank = 2
        elif "raw" in name and path.suffix.lower() == ".json":
            rank = 1
        else:
            rank = 0
        return rank, path.stat().st_mtime

    return max(candidates, key=priority)


def load_flat_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_flat_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} er ikke en flat JSON-liste")
    return data


def load_raw_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload.get("items", [])
    embedded = payload.get("_embedded", {}) or {}
    persons = embedded.get("persons", {}) or {}
    teams = embedded.get("teams", {}) or {}
    positions = embedded.get("positions", {}) or {}

    rows: list[dict[str, Any]] = []
    for item in items:
        person_id = txt(item.get("personId"))
        team_id = txt(item.get("teamId"))
        position_id = txt(item.get("positionId"))
        person = persons.get(person_id, {}) or {}
        team = teams.get(team_id, {}) or {}
        position = positions.get(position_id, {}) or {}
        position_raw = first_non_empty(item.get("position"), get_generic_name(position), position.get("type"), position.get("slug"))
        rows.append(
            {
                "holdet_player_id": item.get("id"),
                "holdet_person_id": item.get("personId"),
                "player_name": get_person_name(person),
                "holdet_team_id": item.get("teamId"),
                "team_name": get_generic_name(team),
                "holdet_position_id": item.get("positionId"),
                "position_raw": position_raw,
                "position": normalize_position(position_raw),
                "start_price": item.get("startPrice"),
                "price": item.get("price"),
                "is_out": item.get("isOut"),
            }
        )
    return rows


def load_holdet_rows(path: Path) -> list[dict[str, Any]]:
    name = path.name.lower()
    if path.suffix.lower() == ".csv":
        rows = load_flat_csv(path)
    elif "raw" in name:
        rows = load_raw_json(path)
    else:
        rows = load_flat_json(path)
    if not rows:
        raise ValueError(f"{path} indeholder ingen spillerrækker")
    return rows


def normalize_holdet_row(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    raw_price = first_non_empty(row.get("price"), row.get("holdet_price"), row.get("start_price"), row.get("startPrice"))
    position = normalize_position(first_non_empty(row.get("position"), row.get("holdet_position"), row.get("position_raw")))
    team_id = canonical_holdet_team(row)
    return {
        "holdet_player_id": first_non_empty(row.get("holdet_player_id"), row.get("id"), row.get("player_id")),
        "player_name": first_non_empty(row.get("player_name"), row.get("name"), row.get("fullName"), row.get("displayName")),
        "team_id": team_id,
        "team_name": first_non_empty(row.get("team_name"), row.get("team"), row.get("country")),
        "position": position,
        "price": parse_price(raw_price),
        "raw_price": raw_price,
        "source_file": source_file,
    }


def normalize_pool_player(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_player_id": txt(player.get("player_id")),
        "pool_player_name": txt(player.get("player_name")),
        "pool_team_id": normalize_team(first_non_empty(player.get("team_id"), player.get("holdet_team_id"), player.get("team_name"))),
        "pool_position": normalize_position(first_non_empty(player.get("position"), player.get("holdet_position"))),
    }


def name_team_position_key(name: str, team: str, position: str) -> tuple[str, str, str]:
    return norm_text(name), normalize_team(team), normalize_position(position)


def name_team_key(name: str, team: str) -> tuple[str, str]:
    return norm_text(name), normalize_team(team)


def closest_pool_match(holdet_player: dict[str, Any], pool_players: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float, str]:
    h_name = norm_text(holdet_player["player_name"])
    h_team = normalize_team(holdet_player["team_id"])
    h_position = normalize_position(holdet_player["position"])

    best: dict[str, Any] | None = None
    best_score = 0.0
    best_reason = ""

    for pool in pool_players:
        p_name = norm_text(pool["pool_player_name"])
        score = SequenceMatcher(None, h_name, p_name).ratio()
        same_team = h_team and h_team == pool["pool_team_id"]
        same_position = h_position and h_position == pool["pool_position"]
        if same_team:
            score += 0.20
        if same_position:
            score += 0.05
        if score > best_score:
            best = pool
            best_score = score
            best_reason = "similar_name"
            if same_team:
                best_reason += "+same_team"
            if same_position:
                best_reason += "+same_position"

    return best, best_score, best_reason


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        source_path = find_holdet_source_file()
        holdet_rows_raw = load_holdet_rows(source_path)
    except Exception as exc:
        print(f"FEJL: {exc}")
        return 1

    if not PLAYER_POOL_PATH.exists():
        print(f"FEJL: Mangler {PLAYER_POOL_PATH}")
        return 1

    with PLAYER_POOL_PATH.open(encoding="utf-8-sig") as f:
        player_pool = json.load(f)

    try:
        source_file = str(source_path.relative_to(PROJECT_ROOT))
    except ValueError:
        source_file = str(source_path)
    holdet_players = [normalize_holdet_row(row, source_file) for row in holdet_rows_raw]
    holdet_players = [player for player in holdet_players if player["player_name"]]
    pool_players = [normalize_pool_player(player) for player in player_pool]

    pool_by_ntp: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    pool_by_nt: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for player in pool_players:
        pool_by_ntp.setdefault(
            name_team_position_key(player["pool_player_name"], player["pool_team_id"], player["pool_position"]),
            [],
        ).append(player)
        pool_by_nt.setdefault(
            name_team_key(player["pool_player_name"], player["pool_team_id"]),
            [],
        ).append(player)

    new_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    secure_matches = 0
    possible_matches = 0

    for player in holdet_players:
        ntp = name_team_position_key(player["player_name"], player["team_id"], player["position"])
        nt = name_team_key(player["player_name"], player["team_id"])

        match_type = ""
        pool_match = None
        if ntp in pool_by_ntp:
            pool_match = pool_by_ntp[ntp][0]
            match_type = "name_team_position"
            secure_matches += 1
        elif nt in pool_by_nt:
            pool_match = pool_by_nt[nt][0]
            match_type = "name_team"
            possible_matches += 1

        if pool_match:
            match_rows.append(
                {
                    "holdet_player_id": player["holdet_player_id"],
                    "holdet_player_name": player["player_name"],
                    "holdet_team_id": player["team_id"],
                    "holdet_position": player["position"],
                    "pool_player_id": pool_match["pool_player_id"],
                    "pool_player_name": pool_match["pool_player_name"],
                    "pool_team_id": pool_match["pool_team_id"],
                    "pool_position": pool_match["pool_position"],
                    "match_type": match_type,
                    "source_file": source_file,
                }
            )
            continue

        closest, score, reason = closest_pool_match(player, pool_players)
        if closest and score >= 0.82:
            possible_matches += 1
            mismatch_rows.append(
                {
                    "holdet_player_id": player["holdet_player_id"],
                    "holdet_player_name": player["player_name"],
                    "holdet_team_id": player["team_id"],
                    "holdet_position": player["position"],
                    "closest_pool_player_id": closest["pool_player_id"],
                    "closest_pool_player_name": closest["pool_player_name"],
                    "closest_pool_team_id": closest["pool_team_id"],
                    "closest_pool_position": closest["pool_position"],
                    "reason": f"{reason}; score={score:.3f}",
                    "source_file": source_file,
                }
            )
            status = "possible_existing"
            new_reason = f"closest_match_score={score:.3f}"
        else:
            status = "potential_new"
            new_reason = "no_name_team_match"

        new_rows.append(
            {
                "holdet_player_id": player["holdet_player_id"],
                "player_name": player["player_name"],
                "team_id": player["team_id"],
                "team_name": player["team_name"],
                "position": player["position"],
                "price": player["price"],
                "raw_price": player["raw_price"],
                "status": status,
                "reason": new_reason,
                "source_file": source_file,
            }
        )

    new_rows.sort(key=lambda row: (parse_price(row["price"]), row["player_name"]), reverse=True)

    write_csv(
        NEW_PLAYERS_REVIEW_PATH,
        ["holdet_player_id", "player_name", "team_id", "team_name", "position", "price", "raw_price", "status", "reason", "source_file"],
        new_rows,
    )
    write_csv(
        MATCH_REPORT_PATH,
        [
            "holdet_player_id",
            "holdet_player_name",
            "holdet_team_id",
            "holdet_position",
            "pool_player_id",
            "pool_player_name",
            "pool_team_id",
            "pool_position",
            "match_type",
            "source_file",
        ],
        match_rows,
    )
    write_csv(
        MISMATCH_REPORT_PATH,
        [
            "holdet_player_id",
            "holdet_player_name",
            "holdet_team_id",
            "holdet_position",
            "closest_pool_player_id",
            "closest_pool_player_name",
            "closest_pool_team_id",
            "closest_pool_position",
            "reason",
            "source_file",
        ],
        mismatch_rows,
    )

    potential_new_count = sum(1 for row in new_rows if row["status"] == "potential_new")
    print(f"Holdet-fil brugt: {source_file}")
    print(f"Spillere i Holdet-data: {len(holdet_players)}")
    print(f"Spillere i player_pool_v1.json: {len(player_pool)}")
    print(f"Sikre eksisterende matches: {secure_matches}")
    print(f"Mulige/usikre matches: {possible_matches}")
    print(f"Potentielt nye spillere: {potential_new_count}")
    print("Top 50 potentielt nye spillere efter pris:")
    for row in [row for row in new_rows if row["status"] == "potential_new"][:50]:
        print(f"- {row['player_name']} | {row['team_id']} | {row['position']} | {row['price']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
