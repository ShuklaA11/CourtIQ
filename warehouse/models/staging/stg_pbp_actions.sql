-- One row per play-by-play action (game.actions[]) for in-scope games.
-- Clocks arrive ISO-8601-ish ('PT11M41.00S'); we parse them to float seconds.
-- Scores arrive as strings and are '' before the first bucket -> NULL.

with raw as (
    select
        game.gameId       as game_id,
        unnest(game.actions) as action
    from read_json_auto(
        '{{ var("raw_root") }}/pbp/*.json',
        union_by_name = true
    )
),

typed as (
    select
        game_id,
        action.actionNumber as action_number,
        action.period       as period,
        nullif(action.clock, '') as clock_raw,

        -- 'PT##M##.##S' -> total seconds as a float; empty clock -> NULL.
        case
            when action.clock is null or action.clock = '' then null
            else cast(regexp_extract(action.clock, 'PT([0-9]+)M', 1) as integer) * 60
               + cast(regexp_extract(action.clock, 'M([0-9.]+)S', 1) as double)
        end as seconds_remaining,

        action.teamId    as team_id,
        action.personId  as person_id,
        nullif(action.playerName, '') as player_name,
        action.actionType as action_type,
        nullif(action.subType, '')    as sub_type,
        action.description as description,
        nullif(action.shotResult, '') as shot_result,
        cast(action.isFieldGoal as boolean) as is_field_goal,
        action.shotValue   as shot_value,
        action.pointsTotal as points_total,

        -- Scores are strings; '' (pre-scoring) -> NULL.
        try_cast(nullif(action.scoreHome, '') as integer) as score_home,
        try_cast(nullif(action.scoreAway, '') as integer) as score_away
    from raw
)

select *
from typed
where {{ game_id_filter('game_id') }}
