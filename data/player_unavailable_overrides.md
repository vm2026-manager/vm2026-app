# Player unavailable overrides

`player_unavailable_overrides.csv` is the persistent manual source for confirmed
out/unavailable players when upstream Holdet data is missing or stale.

Run `python tools/apply_unavailable_players_safe.py` to apply exact `player_id`
matches into `player_pool_v1.json`, `player_ev_group_stage_v1.csv`, and
`player_start_security_nt.csv`. The script creates backups, zeroes useful model
and display values, runs the optimizer, and writes `player_unavailable_audit.csv`.

Only confirmed out/unavailable decisions belong in this file.
