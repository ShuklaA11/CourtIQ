"""Leakage-safe PRE-TIP player-availability layer.

The market prices *who is on the floor*: a star ruled out an hour before tip
moves the line more than any prior-season rating. This module turns each game's
V3 ``boxScoreTraditional`` payload into a per-(game, team, player) availability
label that a pre-game model can consume as the inactive-list signal that WOULD
have been known at tip-off.

Single source of truth is one pure rule, ``classify_player_availability``:

* A player with ``minutes_played > 0`` is AVAILABLE by construction — being on the
  floor is proof, and this check WINS over any comment marker.
* An empty comment (the player dressed / played) is AVAILABLE.
* A ``Coach's Decision`` comment (substring ``'coach'``) is a healthy scratch —
  AVAILABLE (the player could have played).
* A comment containing any ``UNAVAILABLE_MARKERS`` substring (case-insensitive)
  is UNAVAILABLE.
* Any other non-empty note is unrecognized and defaults to AVAILABLE, so the
  layer only ever flags a player OUT on an explicit, auditable marker.

``parse_box_availability`` applies that rule to both teams of one payload;
``build_game_availability`` folds the raw box_traditional JSONs — keeping ONLY
games present in the possession mart, keyed by the raw file's ``game_id`` stem,
the same convention ``recon.build`` uses — into one tidy frame.
``rollup_unavailable`` collapses it to a per-(game, team) inactive list.

Pure pandas; every function returns a new object and never mutates its input.
``personId`` in the payload is the RAPM ``player_id``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/winprob")
MART_PARQUET_NAME = "fct_game_states.parquet"
AVAILABILITY_PARQUET_NAME = "game_availability.parquet"

# Raw box_traditional directory, RELATIVE to the parent of ``data/winprob`` (i.e.
# ``data/raw/box_traditional``). ``recon.build`` reads the same tree, one file per
# game named ``{game_id}.json`` where the stem IS the mart game_id.
RAW_BOX_SUBDIR = Path("raw") / "box_traditional"

# The payload's top-level box key.
BOX_KEY = "boxScoreTraditional"

# Case-insensitive substrings that mark a DNP comment as UNAVAILABLE. Derived from
# the enumerated DNP/DND/NWT reasons in the V3 feed; this list is the SOLE
# unavailable path (a comment matching none of these is treated as available).
UNAVAILABLE_MARKERS: tuple[str, ...] = (
    "injury",
    "illness",
    "not with team",
    "suspension",
    "suspend",
    "rest",
    "personal",
    "health and safety",
    "return to competition",
)

# Substring identifying a healthy scratch ("DNP - Coach's Decision") — AVAILABLE.
COACH_DECISION_MARKER: str = "coach"

# The availability frame's exact columns, in fixed order.
AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "player_id",
    "available",
    "comment",
)

# The per-(game, team) rollup's exact columns, in fixed order.
ROLLUP_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "unavailable_player_ids",
    "n_unavailable",
)


@dataclass(frozen=True)
class AvailabilityRecord:
    """One (game, team, player) availability label — an immutable value object."""

    game_id: str
    team_id: int
    player_id: int
    available: bool
    comment: str


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], ctx: str) -> None:
    """Fail fast if any required column is absent, naming the offenders."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{ctx}: missing columns {missing}")


# --------------------------------------------------------------------------
# The availability rule (single source of truth).
# --------------------------------------------------------------------------

def classify_player_availability(comment: str, minutes_played: float) -> bool:
    """Return ``True`` iff the player was available to play (pure, deterministic).

    Order matters: ``minutes_played > 0`` is checked FIRST so that being on the
    floor always wins over any unavailable marker. Then an empty comment and a
    ``Coach's Decision`` note both mean AVAILABLE; only an explicit unavailable
    marker returns ``False``. An unrecognized non-empty note defaults to
    AVAILABLE so the layer never flags a player out without a named reason.
    """
    if minutes_played > 0:
        return True

    text = (comment or "").strip().lower()
    if text == "":
        return True
    if COACH_DECISION_MARKER in text:
        return True
    return not any(marker in text for marker in UNAVAILABLE_MARKERS)


def _minutes_played(raw: object) -> float:
    """``'MM:SS'`` box minutes -> whole minutes as a float; blank / DNP -> ``0.0``.

    Only the sign matters downstream (``> 0`` means the player was on the floor),
    so a coarse minute count is sufficient and a malformed value degrades to
    ``0.0`` rather than raising.
    """
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text or ":" not in text:
        return 0.0
    minutes, _, seconds = text.partition(":")
    try:
        return int(minutes) + float(seconds) / 60.0
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# Per-payload parsing.
# --------------------------------------------------------------------------

def parse_box_availability(box_json: dict) -> list[AvailabilityRecord]:
    """Availability records for BOTH teams of one ``boxScoreTraditional`` payload.

    ``personId`` is the RAPM ``player_id``; ``gameId`` is the mart game_id string.
    Returns one record per player across ``homeTeam`` and ``awayTeam``; the input
    is never mutated.
    """
    if BOX_KEY not in box_json:
        raise ValueError(f"payload missing {BOX_KEY!r} key")
    box = box_json[BOX_KEY]
    game_id = str(box["gameId"])

    records: list[AvailabilityRecord] = []
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        team_id = int(team["teamId"])
        for player in team["players"]:
            comment = str(player.get("comment") or "")
            minutes = _minutes_played((player.get("statistics") or {}).get("minutes"))
            records.append(
                AvailabilityRecord(
                    game_id=game_id,
                    team_id=team_id,
                    player_id=int(player["personId"]),
                    available=classify_player_availability(comment, minutes),
                    comment=comment,
                )
            )
    return records


def _with_availability_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the availability frame to its fixed dtypes (a NEW frame)."""
    typed = frame.copy()
    typed["game_id"] = typed["game_id"].astype(str)
    typed["team_id"] = typed["team_id"].astype("int64")
    typed["player_id"] = typed["player_id"].astype("int64")
    typed["available"] = typed["available"].astype(bool)
    typed["comment"] = typed["comment"].astype(str)
    return typed


# --------------------------------------------------------------------------
# Mart-scoped build over the raw box_traditional tree.
# --------------------------------------------------------------------------

def build_game_availability(
    raw_dir: Path, mart_game_ids: Iterable[str]
) -> pd.DataFrame:
    """Fold the raw box_traditional JSONs into one availability frame.

    Iterates ``{game_id}.json`` files under ``raw_dir`` (the ``recon.build`` file
    convention: the stem IS the mart game_id) and keeps ONLY games whose id is in
    ``mart_game_ids``. Emits one row per (game_id, team_id, player_id) with the
    ``available`` label and its source ``comment``. Returns a NEW frame with
    exactly ``AVAILABILITY_COLUMNS``.
    """
    raw_dir = Path(raw_dir)
    keep = {str(g) for g in mart_game_ids}

    records: list[AvailabilityRecord] = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.stem not in keep:
            continue
        with open(path) as handle:
            payload = json.load(handle)
        records.extend(parse_box_availability(payload))

    frame = pd.DataFrame(
        [
            (r.game_id, r.team_id, r.player_id, r.available, r.comment)
            for r in records
        ],
        columns=list(AVAILABILITY_COLUMNS),
    )
    return _with_availability_dtypes(frame)


def rollup_unavailable(availability: pd.DataFrame) -> pd.DataFrame:
    """Collapse an availability frame to one row per (game_id, team_id).

    Emits ``unavailable_player_ids`` (the sorted list of that team's out players)
    and ``n_unavailable`` for EVERY (game, team) present in ``availability`` — a
    team with nobody out gets an empty list and ``0``. Returns a NEW frame with
    exactly ``ROLLUP_COLUMNS``; the input is never mutated.
    """
    _require_columns(
        availability,
        ("game_id", "team_id", "player_id", "available"),
        "rollup_unavailable",
    )
    avail = availability.copy()
    avail["available"] = avail["available"].astype(bool)

    keys = (
        avail[["game_id", "team_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "team_id"])
        .reset_index(drop=True)
    )
    out_players = avail.loc[~avail["available"]]
    grouped = (
        out_players.groupby(["game_id", "team_id"])["player_id"]
        .apply(lambda ids: sorted(int(p) for p in ids))
        .to_dict()
    )

    ids_column = [
        grouped.get((row.game_id, int(row.team_id)), [])
        for row in keys.itertuples(index=False)
    ]
    out = keys.copy()
    out["unavailable_player_ids"] = ids_column
    out["n_unavailable"] = [len(ids) for ids in ids_column]
    return out.loc[:, list(ROLLUP_COLUMNS)]


# --------------------------------------------------------------------------
# Artifact build + CLI.
# --------------------------------------------------------------------------

def run(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Build ``game_availability.parquet`` for every game in the possession mart.

    Reads the mart's game_ids, folds the matching raw box_traditional payloads,
    writes the availability frame to ``data_dir/game_availability.parquet``, and
    returns it.
    """
    data_dir = Path(data_dir)
    mart_path = data_dir / MART_PARQUET_NAME
    if not mart_path.exists():
        raise FileNotFoundError(
            f"missing game-state mart at {mart_path}; run ./game_states.sh first"
        )
    raw_dir = data_dir.parent / RAW_BOX_SUBDIR
    if not raw_dir.exists():
        raise FileNotFoundError(f"missing raw box directory at {raw_dir}")

    mart = pd.read_parquet(mart_path, columns=["game_id"])
    mart_game_ids = mart["game_id"].astype(str).unique().tolist()

    availability = build_game_availability(raw_dir, mart_game_ids)
    availability.to_parquet(data_dir / AVAILABILITY_PARQUET_NAME, index=False)
    return availability


def _print_summary(availability: pd.DataFrame) -> None:
    games = availability["game_id"].nunique()
    players = len(availability)
    unavailable = int((~availability["available"]).sum())
    print(
        f"availability: {games:,} games, {players:,} player-rows, "
        f"{unavailable:,} unavailable ({unavailable / max(players, 1):.2%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the leakage-safe pre-tip player-availability artifact"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    availability = run(Path(args.data_dir))
    _print_summary(availability)


if __name__ == "__main__":
    main()
