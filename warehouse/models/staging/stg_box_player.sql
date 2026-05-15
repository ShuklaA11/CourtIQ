-- One row per (game_id, team_id, person_id) from the traditional box score.
-- Home and away player lists are unnested and tagged with is_home.
-- Minutes arrive as 'MM:SS' ('' for DNPs) and are parsed to integer seconds.

with raw as (
    select boxScoreTraditional as b
    from read_json_auto(
        '{{ var("raw_root") }}/box_traditional/*.json',
        union_by_name = true
    )
),

home as (
    select
        b.gameId            as game_id,
        b.homeTeam.teamId   as team_id,
        unnest(b.homeTeam.players) as player,
        true                as is_home
    from raw
),

away as (
    select
        b.gameId            as game_id,
        b.awayTeam.teamId   as team_id,
        unnest(b.awayTeam.players) as player,
        false               as is_home
    from raw
),

combined as (
    select * from home
    union all
    select * from away
),

typed as (
    select
        game_id,
        team_id,
        player.personId as person_id,
        player.nameI    as player_name,
        nullif(player.statistics.minutes, '') as minutes_raw,

        -- 'MM:SS' -> integer seconds; '' (DNP) -> NULL.
        case
            when player.statistics.minutes is null
              or player.statistics.minutes = '' then null
            else cast(split_part(player.statistics.minutes, ':', 1) as integer) * 60
               + cast(split_part(player.statistics.minutes, ':', 2) as integer)
        end as minutes_seconds,

        player.statistics.points               as points,
        -- Counting stats the possession-estimate gate needs (box formula).
        player.statistics.fieldGoalsAttempted  as fga,
        player.statistics.freeThrowsAttempted   as fta,
        player.statistics.reboundsOffensive     as oreb,
        player.statistics.turnovers             as tov,
        is_home
    from combined
),

final as (
    select
        *,
        -- Alias used by the minutes-reconciliation mart; NULL DNPs -> 0 seconds.
        coalesce(minutes_seconds, 0) as box_seconds
    from typed
)

select *
from final
where {{ game_id_filter('game_id') }}
