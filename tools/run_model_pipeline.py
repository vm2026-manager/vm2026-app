from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SCRIPTS = [
    "tools/build_market_odds_layer.py",
    "tools/audit_market_odds_layer.py",
    "tools/build_clean_market_optimizer_squads.py",
    "tools/write_model_status_snapshot.py",
    "tools/audit_app_data_consistency.py",
    "tools/sanity_check_active_json.py",
]


def run_script(script: str) -> None:
    path = PROJECT_ROOT / script

    if not path.exists():
        raise FileNotFoundError(f"Mangler script: {path}")

    print("")
    print("=" * 80)
    print(f"KØRER: {script}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script fejlede: {script}")


def main() -> None:
    print("VM 2026 MODEL PIPELINE")
    print(f"Projektmappe: {PROJECT_ROOT}")

    for script in SCRIPTS:
        run_script(script)

    print("")
    print("=" * 80)
    print("FÆRDIG")
    print("=" * 80)
    print("Pipeline er kørt færdig.")
    print("Tjek især:")
    print("- Bedste formation")
    print("- Bedste hold")
    print("- Audit: OK")
    print("- Manglende odds-kolonner i audit_market_odds_layer")


if __name__ == "__main__":
    main()
