-- One row per (game_id, team_id) from the advanced box score.
-- Home and away team stat blocks are stacked and tagged with is_home.

with raw as (
    select boxScoreAdvanced as b
    from read_json_auto(
        '{{ var("raw_root") }}/box_advanced/*.json',
        union_by_name = true
    )
),

home as (
    select
        b.gameId          as game_id,
        b.homeTeam.teamId as team_id,
        b.homeTeam.statistics as stats,
        true              as is_home
    from raw
),

away as (
    select
        b.gameId          as game_id,
        b.awayTeam.teamId as team_id,
        b.awayTeam.statistics as stats,
        false             as is_home
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
        cast(stats.possessions as double)     as possessions,
        cast(stats.pace as double)            as pace,
        cast(stats.offensiveRating as double) as offensive_rating,
        is_home
    from combined
)

select *
from typed
where {{ game_id_filter('game_id') }}
