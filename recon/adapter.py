"""Bridge raw NBA-API play-by-play JSON into the segmenter's action log.

`recon.possessions.recon_possessions` consumes a *normalised, lineup-tagged*
action log (monotone `seconds_elapsed`, `event` in a small closed vocabulary,
`points`, and free-throw `ft_index`/`ft_total`/`ft_kind`). The raw pull under
`data/raw/pbp` is nothing like that: it is NBA-API v3 format — `actionType`/
`subType` strings, an ISO-8601 clock (`"PT12M00.00S"`) that counts *down*,
`scoreHome`/`scoreAway` as strings that are blank except on scoring plays, and
no lineups at all. This module is the adapter that closes that gap.

What it deliberately does NOT do
--------------------------------
Reconstruct who is on the floor. Five-man lineups come from the upstream
`onfloor-minutes-recon` stage (its own pipeline step); wiring real fives in is
that stage's job, injected here via the optional `lineups_at` hook. With no
hook, we still attach a **two-team lineup map** (`{home_id: (), away_id: ()}`)
so the segmenter can resolve `offense_team_id`/`defense_team_id` and the balance
gate is meaningful — only the player identities inside `offense_five`/
`defense_five` are left empty. Keeping lineup reconstruction out of this module
is a scope boundary, not an oversight.

Ground truth is independent of the walk
---------------------------------------
`final_score_from_box` and `final_score_from_pbp` both derive the official score
without ever calling the segmenter — one from the box-score endpoint, one from
the running `scoreHome`/`scoreAway` maxima. The hard reconciliation gate compares
the possession walk against these, so a bug in the walk cannot hide by also being
present in the ground truth.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable

# ISO-8601 duration on the game clock, e.g. "PT12M00.00S" => 12:00 remaining.
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")

# Period length in seconds: 12:00 regulation, 5:00 for any overtime (period 5+).
_REGULATION_SECONDS = 12 * 60
_OVERTIME_SECONDS = 5 * 60

# "Free Throw 2 of 3", "Free Throw Flagrant 1 of 3", etc. -> (index, total).
_FT_INDEX_RE = re.compile(r"(\d+)\s+of\s+(\d+)")

# Raw NBA-API actionType -> our normalised event vocabulary. Anything not listed
# (Foul, Substitution, Timeout, Jump Ball, Violation, Ejection, Instant Replay,
# and the blank-actionType STEAL/BLOCK annotations) is boundary-neutral: it is
# emitted as "delay" so the and-1 look-ahead skips it (see _BOUNDARY_NEUTRAL in
# recon.possessions) but it never opens or closes a possession.
_EVENT_MAP = {
    "Made Shot": "made_shot",
    "Missed Shot": "missed_shot",
    "Rebound": "rebound",
    "Turnover": "turnover",
    "Free Throw": "free_throw",
}
_NEUTRAL_EVENT = "delay"


def parse_clock(clock: str, period: int) -> float:
    """Seconds elapsed *within the period* from the countdown clock.

    The raw clock counts down from the period length, so elapsed =
    period_length - remaining. Monotone non-decreasing within a period, which is
    exactly the ordering the segmenter relies on for start/end seconds.
    """
    match = _CLOCK_RE.fullmatch(clock.strip()) if clock else None
    if not match:
        return 0.0
    remaining = int(match.group(1)) * 60 + float(match.group(2))
    period_length = _REGULATION_SECONDS if period <= 4 else _OVERTIME_SECONDS
    return period_length - remaining


def _parse_free_throw(subtype: str) -> tuple[int, int, str]:
    """(ft_index, ft_total, ft_kind) from a free-throw subType string.

    Technicals ("Free Throw Technical") are a single dead-ball attempt — index 1
    of 1, kind "technical". Flagrant and clear-path free throws let the shooting
    team retain the ball, so both map to kind "flagrant" (the segmenter's
    retain-possession branch). Everything else is a regular trip whose index/
    total come from the "N of M" in the label.
    """
    if "Technical" in subtype:
        return 1, 1, "technical"
    kind = "flagrant" if ("Flagrant" in subtype or "Clear Path" in subtype) else "regular"
    match = _FT_INDEX_RE.search(subtype)
    if match:
        return int(match.group(1)), int(match.group(2)), kind
    return 1, 1, kind


def _foul_retains_possession(foul_subtype: str | None) -> bool:
    """Does the free throw awarded by this foul leave the ball with the shooter?

    Away-from-play fouls and transition take fouls ("Personal Take") award one
    free throw *and* return possession to the fouled team — the offense never
    spent a shot. Treating that made free throw as a normal trip-ender would
    close the possession and hand the same team a second, phantom one right after
    (a per-period balance break). These map to the retain branch instead.
    """
    if not foul_subtype:
        return False
    lowered = foul_subtype.lower()
    return "away from play" in lowered or "take" in lowered


def _is_missed_free_throw(description: str) -> bool:
    """Made vs missed free throw. Missed FTs are prefixed "MISS" in the raw feed
    (and, cross-checked over 60+ games, are exactly the ones that leave the score
    fields blank); made FTs are not."""
    return description.strip().startswith("MISS")


def team_ids_from_pbp(pbp: dict) -> tuple[int, ...]:
    """The distinct real team ids appearing in the log (period markers use 0)."""
    seen: dict[int, None] = {}
    for action in pbp["game"]["actions"]:
        team = action.get("teamId")
        if team:
            seen.setdefault(team, None)
    return tuple(seen)


def team_tokens_from_box(box_traditional: dict) -> dict[int, tuple[str, ...]]:
    """team_id -> lowercase identifying tokens (name, city, tricode).

    Team-credited events (team rebounds, shot-clock turnovers) arrive with
    `teamId == 0` and a blank tricode, but the description leads with the team
    name ("Nets Rebound", "TRAIL BLAZERS Rebound"). These tokens let us recover
    the team by prefix-matching that description.
    """
    box = box_traditional["boxScoreTraditional"]
    tokens: dict[int, tuple[str, ...]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        tokens[team["teamId"]] = (
            team["teamName"].lower(),
            team["teamCity"].lower(),
            team["teamTricode"].lower(),
        )
    return tokens


def _resolve_team_from_description(description: str,
                                   team_tokens: dict[int, tuple[str, ...]]) -> int | None:
    """Team id whose name/city/tricode *prefixes* the description, else None.

    Prefix (not substring) matching is deliberate: "Hornets Rebound" must not
    match the Nets via the "nets" substring, and "Trail Blazers" must not match
    the Clippers' "LA" city hiding inside "Blazers".
    """
    text = description.strip().lower()
    for team_id, tokens in team_tokens.items():
        if any(token and text.startswith(token) for token in tokens):
            return team_id
    return None


def to_action_log(
    pbp: dict,
    lineups_at: Callable[[int, dict], dict] | None = None,
    team_tokens: dict[int, tuple[str, ...]] | None = None,
) -> list[dict]:
    """Normalise one game's raw pbp into the segmenter's action-log contract.

    `lineups_at(action_index, raw_action) -> {team_id: (p1..p5)}` is the optional
    injection point for the upstream on-floor stage. Without it we attach a
    two-team map with empty fives, which is all the segmenter needs for
    offense/defense attribution and the balance gate.

    `team_tokens` (from `team_tokens_from_box`) recovers the team on events that
    the raw feed credits to team id 0 — team rebounds and shot-clock turnovers.
    Without it those events carry no team, which strands whole possessions with a
    null offense and wrecks the balance gate, so a real run must pass it in.
    """
    team_ids = team_ids_from_pbp(pbp)
    default_lineups = {team: () for team in team_ids}

    log: list[dict] = []
    # The subType of the most recent foul, used to spot free throws that return
    # possession to the shooter (away-from-play / take fouls).
    last_foul_subtype: str | None = None
    for index, action in enumerate(pbp["game"]["actions"]):
        action_type = action.get("actionType", "")
        period = int(action["period"])
        seconds = parse_clock(action.get("clock", ""), period)
        team = action.get("teamId") or None
        if team is None and team_tokens and action_type in _EVENT_MAP:
            team = _resolve_team_from_description(action.get("description", ""), team_tokens)
        lineups = lineups_at(index, action) if lineups_at else default_lineups

        if action_type == "Foul":
            last_foul_subtype = action.get("subType", "")

        if action_type == "period":
            if action.get("subType") == "end":
                log.append(_row("period_end", None, period, seconds, 0, lineups))
            continue  # "start" markers carry no segmentation meaning

        event = _EVENT_MAP.get(action_type, _NEUTRAL_EVENT)

        if event == "made_shot":
            points = int(action.get("shotValue") or 2)
            log.append(_row(event, team, period, seconds, points, lineups))
        elif event == "free_throw":
            index_, total, kind = _parse_free_throw(action.get("subType", ""))
            # A one-shot award from an away-from-play or take foul keeps the ball
            # with the shooter: route it through the retain branch, not a trip end.
            if kind == "regular" and total == 1 and _foul_retains_possession(last_foul_subtype):
                kind = "flagrant"
            made = not _is_missed_free_throw(action.get("description", ""))
            log.append(_row(
                event, team, period, seconds, 1 if made else 0, lineups,
                ft_index=index_, ft_total=total, ft_kind=kind, made=made,
            ))
        else:
            # missed_shot, rebound, turnover, and every neutral event: no points.
            log.append(_row(event, team, period, seconds, 0, lineups))

    return log


def _row(event: str, team: int | None, period: int, seconds: float,
         points: int, lineups: dict, **extra: object) -> dict:
    """Assemble one normalised action row."""
    row = {
        "event": event,
        "team_id": team,
        "period": period,
        "seconds_elapsed": seconds,
        "points": points,
        "lineups": lineups,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# Independent ground truth — never touches the possession walk.
# --------------------------------------------------------------------------- #

def final_score_from_box(box_traditional: dict) -> dict[int, int]:
    """Official final score per team id, from the traditional box score."""
    box = box_traditional["boxScoreTraditional"]
    return {
        box["homeTeamId"]: int(box["homeTeam"]["statistics"]["points"]),
        box["awayTeamId"]: int(box["awayTeam"]["statistics"]["points"]),
    }


def final_score_from_pbp(pbp: dict, home_team_id: int, away_team_id: int) -> dict[int, int]:
    """Official final score per team from the running scoreHome/scoreAway maxima.

    A second, endpoint-independent ground truth: the last non-blank score line is
    the final, and taking the max is robust to blank rows. Used to cross-check the
    box-score figures so neither source is trusted blindly.
    """
    home = away = 0
    for action in pbp["game"]["actions"]:
        sh, sa = action.get("scoreHome", ""), action.get("scoreAway", "")
        if sh:
            home = max(home, int(sh))
        if sa:
            away = max(away, int(sa))
    return {home_team_id: home, away_team_id: away}


def box_possessions_estimate(box_advanced: dict) -> dict[int, float]:
    """The formula possession estimate per team from the advanced box score.

    This is `box_advanced.possessions` — the loose-gate reference the segmenter's
    own possession count is sanity-checked against. It is an estimate, not truth.
    """
    box = box_advanced["boxScoreAdvanced"]
    return {
        box["homeTeamId"]: float(box["homeTeam"]["statistics"]["possessions"]),
        box["awayTeamId"]: float(box["awayTeam"]["statistics"]["possessions"]),
    }


# --------------------------------------------------------------------------- #
# Lineup-side parsing: raw pbp -> recon.lineups.Action list.
#
# This is the piece the minutes/lineups gate runs on, and the one the original
# mission got wrong. The real V3 substitution format (verified across every
# season in the corpus) is:
#   * ONE row per swap, actionType "Substitution", subType "" (always blank),
#   * personId == the OUTGOING player (the one leaving the floor),
#   * the INCOMING player appears only in the description: "SUB: <IN> FOR <OUT>".
# So sub_out is read directly from personId; sub_in must be resolved by matching
# the <IN> name against that team's box-score roster.
# --------------------------------------------------------------------------- #

from recon.lineups import SUBSTITUTION, Action  # noqa: E402  (kept near its use)

# "SUB: Ross FOR F. Wagner" -> incoming "Ross", outgoing "F. Wagner".
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")

# actionTypes that never carry a real on-floor player and so must not seed the
# period-starter back-inference (period markers, dead-ball/administrative rows).
_NON_PLAYER_ACTION_TYPES = frozenset({"period", "Timeout", "Instant Replay"})


class AdapterError(ValueError):
    """A raw payload could not be parsed into lineup-reconstruction inputs."""


def parse_team_ids(box_traditional: dict) -> tuple[int, int]:
    """Return ``(home_team_id, away_team_id)`` from a box_traditional payload."""
    box = box_traditional["boxScoreTraditional"]
    return int(box["homeTeamId"]), int(box["awayTeamId"])


_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def _strip_suffix(family: str) -> str:
    """Drop a trailing generational suffix from a folded surname."""
    words = family.split()
    if len(words) >= 2 and words[-1].rstrip(".") in _NAME_SUFFIXES:
        return " ".join(words[:-1])
    return family


def _fold(text: str) -> str:
    """Lowercase and strip diacritics: 'Jokić' -> 'jokic', 'Vučević' -> 'vucevic'.

    The play-by-play description spells names in ASCII while the box-score roster
    keeps the accented spelling, so all name matching happens on this folded form.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_only.strip().lower()


# German umlauts have two accepted transliterations that the feed mixes between:
# the box roster may spell "Poeltl"/"Schroeder" while the description writes
# "Pöltl"/"Schröder". Accent-stripping alone maps ö->o (giving "poltl"), which
# misses the "oe" roster spelling, so we also generate an expanded variant.
_UMLAUT_EXPANSIONS = (("ö", "oe"), ("ü", "ue"), ("ä", "ae"), ("ß", "ss"))


def _name_keys(text: str) -> frozenset[str]:
    """All normalized forms a name might match on: accent-stripped AND umlaut-
    expanded (ö->o and ö->oe), so "Pöltl" and "Poeltl" resolve to each other."""
    lowered = text.lower()
    expanded = lowered
    for umlaut, digraph in _UMLAUT_EXPANSIONS:
        expanded = expanded.replace(umlaut, digraph)
    return frozenset({_fold(lowered), _fold(expanded)}) - {""}


def _bare_keys(keys: frozenset[str]) -> frozenset[str]:
    """The suffix-stripped variants of a key set ('butler iii' -> 'butler')."""
    return frozenset(_strip_suffix(k) for k in keys)


def parse_rosters(box_traditional: dict) -> dict[int, list[dict]]:
    """team_id -> list of folded roster records ``{pid, namei, family, first}``.

    All name parts are accent-folded so ASCII descriptions match the accented
    box-score spelling. Resolution logic lives in ``_resolve_incoming``; keeping
    the roster a flat list (not a pre-merged index) avoids conflating distinct
    surnames like "Williams" and "Williams III".
    """
    box = box_traditional["boxScoreTraditional"]
    rosters: dict[int, list[dict]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        rosters[int(team["teamId"])] = [
            {
                "pid": int(player["personId"]),
                "namei_keys": _name_keys(player.get("nameI") or ""),
                "family_keys": _name_keys(player.get("familyName") or ""),
                "family_bare": _bare_keys(_name_keys(player.get("familyName") or "")),
                "first": _fold(player.get("firstName") or ""),
            }
            for player in team["players"]
        ]
    return rosters


def _resolve_incoming(in_token: str, roster: list[dict],
                      exclude: frozenset[int] = frozenset()) -> int:
    """Resolve a description's incoming-player name to a person_id.

    Matching, in order: exact "I. Surname" (``nameI``); then surname (suffix-
    insensitive on both sides, so "Butler" matches "Butler III" and "Martin Jr."
    matches "Martin"), optionally narrowed by a leading first-name prefix
    ("Ja. Green" -> firstName starts with "ja"). ``exclude`` drops person_ids
    already accounted for (e.g. on-floor players), which resolves an otherwise
    ambiguous shared surname to the one candidate off the floor.

    Raises AdapterError when the name is unknown or still ambiguous, so the caller
    can quarantine the game rather than guess.
    """
    avail = [p for p in roster if p["pid"] not in exclude]
    token_keys = _name_keys(in_token)

    exact = [p for p in avail if p["namei_keys"] & token_keys]
    if len(exact) == 1:
        return exact[0]["pid"]

    # Exact surname match (suffix included, e.g. "Jackson Jr.") disambiguates a
    # Jr./Sr. from a plain surname before we fall back to suffix-insensitive.
    exact_family = [p for p in avail if p["family_keys"] & token_keys]
    if len(exact_family) == 1:
        return exact_family[0]["pid"]

    # "j. surname" / "ja. surname" — a leading token ending in '.' is a first-name
    # prefix used to disambiguate a shared surname; split it off.
    prefix = None
    words = in_token.split()
    if len(words) >= 2 and words[0].endswith("."):
        prefix = _fold(words[0].rstrip("."))
        surname = " ".join(words[1:])
    else:
        surname = in_token
    surname_bare = _bare_keys(_name_keys(surname))

    candidates = [p for p in avail if p["family_bare"] & surname_bare]
    if prefix is not None:
        candidates = [p for p in candidates if p["first"].startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]["pid"]
    if not candidates:
        raise AdapterError(f"incoming sub name {in_token!r} not found on roster")
    raise AdapterError(f"incoming sub name {in_token!r} is ambiguous on roster")


def parse_lineup_actions(pbp: dict, rosters: dict[int, dict]) -> list[Action]:
    """Parse a play-by-play payload into a chronological ``Action`` list.

    Substitutions become one swap Action (sub_out from personId, sub_in resolved
    from the description). Every other player event becomes an Action carrying its
    person_id so the period-starter back-inference can see it; period markers and
    administrative rows are dropped.

    ``order`` is the position in the raw ``actions`` array, NOT ``actionNumber``:
    the array is delivered in true chronological order (clock non-increasing
    within a period) while ``actionNumber`` is not monotonic, so sorting by it
    scrambles the walk and mis-infers period starters.
    """
    actions: list[Action] = []
    for order, raw in enumerate(pbp["game"]["actions"]):
        action_type = raw.get("actionType") or ""
        period = int(raw["period"])
        clock = str(raw.get("clock") or "")

        if action_type == SUBSTITUTION:
            team_id = _opt_int(raw.get("teamId"))
            sub_out = _opt_int(raw.get("personId"))
            match = _SUB_RE.match(raw.get("description") or "")
            if team_id is None or sub_out is None or not match:
                raise AdapterError(
                    f"unparseable substitution (actionNumber {order}): "
                    f"{raw.get('description')!r}"
                )
            roster = rosters.get(team_id)
            if roster is None:
                raise AdapterError(f"substitution for unknown team {team_id}")
            sub_in = _resolve_incoming(match.group(1), roster)
            actions.append(Action(
                order=order, period=period, clock=clock, team_id=team_id,
                action_type=SUBSTITUTION, sub_out=sub_out, sub_in=sub_in,
            ))
        elif action_type in _NON_PLAYER_ACTION_TYPES:
            continue  # no on-floor player to attribute
        else:
            person_id = _opt_int(raw.get("personId"))
            # Some events do not prove a player was on the floor: a technical foul
            # or an ejection can befall a player (or coach) on the BENCH. Null the
            # person on those so they never seed the period-starter back-inference;
            # a genuinely on-floor player still has other events. The walk ignores
            # these action types regardless, so dropping the person is otherwise inert.
            is_technical = action_type == "Foul" and "Technical" in (raw.get("subType") or "")
            if is_technical or action_type == "Ejection":
                person_id = None
            actions.append(Action(
                order=order, period=period, clock=clock,
                team_id=_opt_int(raw.get("teamId")),
                action_type=action_type,
                person_id=person_id,
            ))
    return actions


def parse_box_seconds(box_traditional: dict) -> dict[int, float]:
    """Per-player on-floor seconds from a box_traditional payload (DNPs omitted)."""
    box = box_traditional["boxScoreTraditional"]
    seconds: dict[int, float] = {}
    for side in ("homeTeam", "awayTeam"):
        for player in box[side]["players"]:
            raw_minutes = (player.get("statistics") or {}).get("minutes")
            secs = _parse_box_minutes(raw_minutes)
            if secs is not None:
                seconds[int(player["personId"])] = secs
    return seconds


def _parse_box_minutes(raw: object) -> float | None:
    """'MM:SS' -> seconds; blank/None (a DNP) -> None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or ":" not in text:
        return None
    minutes, _, secs = text.partition(":")
    return int(minutes) * 60 + float(secs)


def _opt_int(value: object) -> int | None:
    """Coerce to int, mapping 0/blank/None to None (team & administrative rows)."""
    if value in (None, "", 0, "0"):
        return None
    return int(value)
