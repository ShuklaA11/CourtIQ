-- One row per game_id. Team ids come straight from the traditional box score
-- header. season / season_type are decoded from the game_id itself:
--   chars 1-3  -> '002' Regular Season, '004' Playoffs
--   chars 4-5  -> two-digit season start year (e.g. '23' -> 2023-24)
-- game_date is not present in the raw pulls (meta.time is the scrape time, not
-- the tip-off date), so it is carried as NULL until a dated source is added.

with raw as (
    select
        boxScoreTraditional.gameId     as game_id,
        boxScoreTraditional.homeTeamId as home_team_id,
        boxScoreTraditional.awayTeamId as away_team_id,
        boxScoreTraditional.homeTeam.statistics.points as home_final_score,
        boxScoreTraditional.awayTeam.statistics.points as away_final_score
    from read_json_auto(
        '{{ var("raw_root") }}/box_traditional/*.json',
        union_by_name = true
    )
),

decoded as (
    select
        game_id,
        2000 + cast(substr(game_id, 4, 2) as integer) as season,
        case substr(game_id, 1, 3)
            when '002' then 'Regular'
            when '004' then 'Playoffs'
            else 'Unknown'
        end as season_type,
        home_team_id,
        away_team_id,
        cast(home_final_score as integer) as home_final_score,
        cast(away_final_score as integer) as away_final_score,
        cast(null as date) as game_date
    from raw
)

select *
from decoded
where {{ game_id_filter('game_id') }}
