# Warehouse (dbt-duckdb)

Staging layer that reads the raw NBA JSON pulls in `../data/raw/{pbp,box_traditional,box_advanced}/`
directly with DuckDB's `read_json_auto` and materializes typed tables into
`courtiq.duckdb`. Scoped to 2023-24 by default (regular season + playoffs).

## Run

```bash
source ../.venv/bin/activate          # deps are in the repo-level .venv
cd warehouse
dbt build --profiles-dir .            # run models + tests
```

`profiles.yml` lives here, so `--profiles-dir .` keeps the run self-contained.
The DuckDB file and dbt `target/` are reproducible build artifacts and are
gitignored.

## Coverage check

`dbt build` enforces coverage via schema tests: a `relationships` pair pins
`stg_pbp_actions` and `stg_box_team_advanced` to an identical set of game_ids,
another `relationships` test guarantees every pbp game has box-player rows, and
an `expected_row_count` test pins `dim_games` to 1312 under the 2023-24 scope.

For a single green/red signal over all three staging sources at once:

```bash
python checks/coverage_check.py            # prints PASS / FAIL, exits 0 / 1
```

It prints `PASS` only when `stg_pbp_actions`, `stg_box_player`, and
`stg_box_team_advanced` cover the exact same 1312 game_ids (set difference empty
in both directions). Pass `--expected` alongside a widened `game_id_patterns`
var when fanning out to more seasons.

## Vars

| Var | Default | Purpose |
|---|---|---|
| `raw_root` | `../data/raw` | Root of the raw JSON pulls (run from `warehouse/`). Pass an absolute path if the data lives elsewhere. |
| `game_id_patterns` | `["00223%", "00423%"]` | `game_id` LIKE patterns that scope every model. `002` = Regular Season, `004` = Playoffs; the next two digits are the season. |
| `rapm_ratings_path` | `../data/rapm/bayes_ratings.parquet` | Exact-posterior player-season ratings used by the opt-in game-state mart. |
| `enable_winprob` | `false` | Enables `fct_game_states` and its hard gate. `./game_states.sh` sets this true. |

Fan out to more seasons without touching SQL — just widen the var:

```bash
dbt build --profiles-dir . \
  --vars '{game_id_patterns: ["00221%","00421%","00222%","00422%","00223%","00423%"]}'
```

## Models

| Model | Grain |
|---|---|
| `dim_games` | one row per `game_id` (season, season_type, home/away team ids, backfilled game date) |
| `stg_pbp_actions` | one row per play-by-play action (`clock` parsed to `seconds_remaining`; string scores parsed, `''` → NULL) |
| `stg_box_player` | one row per `(game_id, team_id, person_id)` (`minutes` `MM:SS` → `minutes_seconds`) |
| `stg_box_team_advanced` | one row per `(game_id, team_id)` (possessions, pace, offensive_rating) |
| `fct_possessions` | one row per reconstructed possession, including start/end time and pre-action feed score |
| `fct_game_states` | one row per possession boundary; opt-in leakage-safe win-probability features with prior-season RAPM |
