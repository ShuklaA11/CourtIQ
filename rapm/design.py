"""Build the RAPM design matrix from fct_possessions.

The model (locked Sprint 2 decisions)
-------------------------------------
Offense/Defense RAPM at the possession grain, rating unit = **player-season**:
each (player, season) with enough floor time gets one offensive column and one
defensive column. For a possession, the five offensive players get +1 in their
O-columns, the five defensive players +1 in their D-columns, plus a single
global home-court term (+1 when the offense is the home team). The response is
points scored by the offense on that possession.

Coefficient signs: a positive O-coefficient means the player *adds* points on
offense; a negative D-coefficient means the player *allows* fewer points on
defense (good defense reads as negative here — we do not flip it, so downstream
"net" = O - D).

Replacement level: a (player, season) seen in fewer than `min_possessions`
possessions is too thin to estimate on its own, so its +1s are pooled into a
per-season replacement column (one for offense, one for defense). This is a
ridge-phase device only — the Phase-3 hierarchical prior shrinks fringe players
automatically, so it must not be applied on top of partial pooling.

Everything here is deterministic and leakage-aware: `split_games` assigns whole
games to folds so no game's possessions ever straddle train and test, and the
column layout is a pure function of the possession counts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

# A (player, season) needs at least this many possessions on the floor
# (offensive + defensive) to earn its own columns; below it, pool to replacement.
DEFAULT_MIN_POSSESSIONS = 200

# Columns per possession row in the player block: five on offense, five on
# defense. The upstream exactly-five gate guarantees this, and we assert it.
PLAYERS_PER_SIDE = 5


@dataclass(frozen=True)
class Column:
    """One column of the design matrix."""
    kind: str            # 'player' | 'replacement' | 'home'
    side: str | None     # 'O' | 'D' | None (home)
    player_id: int | None
    season: int | None
    penalized: bool      # whether ridge/prior should shrink this column


@dataclass(frozen=True)
class Design:
    """The assembled design matrix plus row/column metadata."""
    X: csr_matrix                 # (n_possessions, n_columns), 0/1 entries
    y: np.ndarray                 # (n_possessions,) points scored by offense
    columns: list[Column]         # length n_columns, index == column position
    row_game_id: np.ndarray       # (n_possessions,) game id per row (object/str)
    row_season: np.ndarray        # (n_possessions,) season per row
    row_duration: np.ndarray      # (n_possessions,) possession length, seconds
    row_offense_is_home: np.ndarray  # (n_possessions,) bool

    @property
    def n_possessions(self) -> int:
        return self.X.shape[0]

    @property
    def n_columns(self) -> int:
        return self.X.shape[1]


def _build_columns(
    counts: dict[tuple[int, int], int], min_possessions: int
) -> tuple[list[Column], dict, dict, int]:
    """Lay out the columns from per-(player, season) possession counts.

    Returns (columns, player_lookup, replacement_lookup, home_index) where
    player_lookup maps (side, player, season) -> column index for kept players
    and replacement_lookup maps (side, season) -> column index for the pooled
    fallback. Column order is deterministic: by season, then player, then the
    two replacement columns; the global home term is last.
    """
    columns: list[Column] = []
    player_lookup: dict[tuple[str, int, int], int] = {}
    replacement_lookup: dict[tuple[str, int], int] = {}

    def add(col: Column) -> int:
        columns.append(col)
        return len(columns) - 1

    seasons = sorted({season for (_player, season) in counts})
    for season in seasons:
        kept = sorted(
            player
            for (player, s), c in counts.items()
            if s == season and c >= min_possessions
        )
        for player in kept:
            for side in ("O", "D"):
                player_lookup[(side, player, season)] = add(
                    Column("player", side, player, season, penalized=True)
                )
        for side in ("O", "D"):
            replacement_lookup[(side, season)] = add(
                Column("replacement", side, None, season, penalized=True)
            )
    home_index = add(Column("home", None, None, None, penalized=False))
    return columns, player_lookup, replacement_lookup, home_index


def build_design(
    *,
    game_id: np.ndarray,
    season: np.ndarray,
    offense_five: np.ndarray,
    defense_five: np.ndarray,
    points: np.ndarray,
    offense_is_home: np.ndarray,
    duration_seconds: np.ndarray,
    min_possessions: int = DEFAULT_MIN_POSSESSIONS,
) -> Design:
    """Assemble the sparse design matrix. Pure: arrays in, Design out.

    `offense_five` / `defense_five` are (n_possessions, 5) integer arrays of
    person_ids. All arrays are aligned row-for-row. Every row must carry exactly
    five players per side (enforced upstream by the reconstruction gate).
    """
    n = len(game_id)
    if offense_five.shape != (n, PLAYERS_PER_SIDE) or defense_five.shape != (
        n,
        PLAYERS_PER_SIDE,
    ):
        raise ValueError(
            f"expected ({n}, {PLAYERS_PER_SIDE}) lineups, got "
            f"{offense_five.shape} / {defense_five.shape}"
        )

    # Per-(player, season) floor counts: a player is on exactly one side per
    # possession, so offensive + defensive appearances sum to floor time.
    counts: dict[tuple[int, int], int] = {}
    for side_arr in (offense_five, defense_five):
        players = side_arr.reshape(-1)
        seasons_rep = np.repeat(season, PLAYERS_PER_SIDE)
        for p, s in zip(players.tolist(), seasons_rep.tolist()):
            counts[(p, s)] = counts.get((p, s), 0) + 1

    columns, player_lookup, replacement_lookup, home_index = _build_columns(
        counts, min_possessions
    )

    # Build COO triplets. Each row contributes 10 player entries; home rows add
    # one more. Preallocate to avoid Python list churn over millions of rows.
    n_home = int(np.count_nonzero(offense_is_home))
    nnz = n * 2 * PLAYERS_PER_SIDE + n_home
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    cursor = 0
    for side, side_arr in (("O", offense_five), ("D", defense_five)):
        repl = {s: replacement_lookup[(side, s)] for s in {int(x) for x in season}}
        for slot in range(PLAYERS_PER_SIDE):
            slot_players = side_arr[:, slot]
            block = slice(cursor, cursor + n)
            rows[block] = np.arange(n)
            cols[block] = [
                player_lookup.get((side, int(p), int(s)), repl[int(s)])
                for p, s in zip(slot_players.tolist(), season.tolist())
            ]
            cursor += n

    home_rows = np.nonzero(offense_is_home)[0]
    rows[cursor : cursor + n_home] = home_rows
    cols[cursor : cursor + n_home] = home_index
    cursor += n_home

    data = np.ones(nnz, dtype=np.float64)
    X = csr_matrix((data, (rows, cols)), shape=(n, len(columns)))

    return Design(
        X=X,
        y=np.asarray(points, dtype=np.float64),
        columns=columns,
        row_game_id=np.asarray(game_id),
        row_season=np.asarray(season),
        row_duration=np.asarray(duration_seconds, dtype=np.float64),
        row_offense_is_home=np.asarray(offense_is_home, dtype=bool),
    )


def split_games(
    game_id: np.ndarray, season: np.ndarray, n_folds: int, seed: int
) -> np.ndarray:
    """Assign every row a fold in [0, n_folds) by *game*, blocked within season.

    Whole games go to one fold, so no game's possessions straddle folds — the
    leakage guard for grouped cross-validation. Folds are balanced within each
    season, so every season is represented in every fold. Deterministic in seed.
    """
    fold_of_row = np.empty(len(game_id), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for s in np.unique(season):
        mask = season == s
        games = np.unique(game_id[mask])
        shuffled = rng.permutation(games)
        game_to_fold = {g: i % n_folds for i, g in enumerate(shuffled)}
        fold_of_row[mask] = [game_to_fold[g] for g in game_id[mask]]
    return fold_of_row


def corpus_hash(game_id: np.ndarray, n_possessions: int) -> str:
    """Stable fingerprint of the training corpus for reproducibility.

    Hashes the sorted unique game ids plus the possession count, so a rebuilt
    corpus (e.g. after the quarantine set shrinks) yields a different id and
    results can be tied to the exact input they came from.
    """
    h = hashlib.sha256()
    for g in sorted(set(game_id.tolist())):
        h.update(str(g).encode())
    h.update(str(n_possessions).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Materialization from the warehouse
# --------------------------------------------------------------------------

DEFAULT_DB = "warehouse/courtiq.duckdb"
DEFAULT_OUT = Path("data/rapm")
DEFAULT_FOLDS = 5
DEFAULT_TEST_FRAC_SEED = 20260724  # fixed so the split is reproducible


def _fetch_possessions(db_path: str) -> dict[str, np.ndarray]:
    """Pull the possession columns needed for the design out of DuckDB."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = con.execute(
            """
            select
                p.game_id,
                p.season,
                p.offense_five,
                p.defense_five,
                p.points,
                (p.offense_team_id = g.home_team_id) as offense_is_home,
                p.duration_seconds
            from fct_possessions p
            join dim_games g using (game_id)
            order by p.game_id, p.period, p.possession_number
            """
        ).fetchnumpy()
    finally:
        con.close()

    # offense_five / defense_five arrive as object arrays of python lists; stack
    # into (n, 5) int arrays for the vectorized assembler.
    offense = np.array([list(x) for x in cols["offense_five"]], dtype=np.int64)
    defense = np.array([list(x) for x in cols["defense_five"]], dtype=np.int64)
    return {
        "game_id": np.asarray(cols["game_id"]),
        "season": np.asarray(cols["season"], dtype=np.int64),
        "offense_five": offense,
        "defense_five": defense,
        "points": np.asarray(cols["points"], dtype=np.float64),
        "offense_is_home": np.asarray(cols["offense_is_home"], dtype=bool),
        "duration_seconds": np.asarray(cols["duration_seconds"], dtype=np.float64),
    }


def _write_columns(columns: list[Column], path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "index": i,
                    "kind": c.kind,
                    "side": c.side,
                    "player_id": c.player_id,
                    "season": c.season,
                    "penalized": c.penalized,
                }
            )
            for i, c in enumerate(columns)
        )
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the RAPM design matrix")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-possessions", type=int, default=DEFAULT_MIN_POSSESSIONS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching possessions from {args.db} ...")
    data = _fetch_possessions(args.db)
    n = len(data["game_id"])
    print(f"  {n:,} possessions")

    print(f"Assembling design (min_possessions={args.min_possessions}) ...")
    design = build_design(min_possessions=args.min_possessions, **data)
    folds = split_games(design.row_game_id, design.row_season, args.folds, DEFAULT_TEST_FRAC_SEED)
    chash = corpus_hash(design.row_game_id, n)

    n_players = sum(1 for c in design.columns if c.kind == "player") // 2
    print(f"  X: {design.X.shape[0]:,} x {design.X.shape[1]:,}  "
          f"({design.X.nnz:,} nonzeros)")
    print(f"  {n_players:,} player-seasons with own columns; corpus {chash}")

    save_npz(out / "X.npz", design.X)
    np.save(out / "y.npy", design.y)
    np.save(out / "row_game_id.npy", design.row_game_id)
    np.save(out / "row_season.npy", design.row_season)
    np.save(out / "row_duration.npy", design.row_duration)
    np.save(out / "row_offense_is_home.npy", design.row_offense_is_home)
    np.save(out / "fold.npy", folds)
    _write_columns(design.columns, out / "columns.jsonl")

    manifest = {
        "corpus_hash": chash,
        "n_possessions": int(n),
        "n_columns": int(design.n_columns),
        "n_player_seasons": int(n_players),
        "min_possessions": int(args.min_possessions),
        "n_folds": int(args.folds),
        "split_seed": DEFAULT_TEST_FRAC_SEED,
        "seasons": sorted(int(s) for s in np.unique(design.row_season)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote design artifacts to {out}/  (manifest.json pins the corpus)")


if __name__ == "__main__":
    main()
