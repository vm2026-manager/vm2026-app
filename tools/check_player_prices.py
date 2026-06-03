import json
import unicodedata
from pathlib import Path

names = [
    "Cristiano Ronaldo",
    "Raul Jimenez",
    "Raúl Jiménez",
    "Jonathan David",
    "Kerem Akturkoglu",
    "Kerem Aktürkoğlu",
    "Andreas Schjelderup",
    "Roberto Alvarado",
    "Oscar Bobb",
    "Malik Tillman",
    "Antonio Nusa",
    "Konrad Laimer",
    "Mohamed Oyarzabal",
    "Dani Olmo",
    "Renato Veiga",
]

def norm(s):
    s = str(s or "").lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    return (
        s.replace("ø", "o")
         .replace("æ", "ae")
         .replace("å", "a")
         .replace("ü", "u")
    )

pool = json.loads(Path("data/player_pool_v1.json").read_text(encoding="utf-8"))
wanted = {norm(x) for x in names}

for p in pool:
    n = p.get("player_name") or p.get("name") or ""
    if norm(n) in wanted:
        team = p.get("team_id", "")
        pos = p.get("position", "")
        price = p.get("price", "")
        start = p.get("start_prob", "")
        cond = p.get("conditional_start_prob", "")
        risk = p.get("availability_risk", "")
        print(f"{n:28} | {team:3} | {pos:3} | price={price} | start={start} | cond={cond} | risk={risk}")