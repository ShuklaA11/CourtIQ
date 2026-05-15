-- Validation: counted vs. estimated possessions per team-game, plus a
-- points-reconciliation flag.
--
--  * counted    = possessions the reconstruction actually assigned to the team.
--  * estimated  = the classic box-score estimate FGA - OREB + TOV + 0.44*FTA.
--    The two should agree within a few possessions; a large gap means the event
--    walk dropped or invented changes of control.
--  * points_reconciliation_ok = do the team's summed possession points equal its
--    official final score? This is exact on clean data, so any FALSE is a real
--    defect (a miscredited or missing bucket).
-- Reconstructed games only: a quarantined game has no possessions, so it would
-- otherwise show 0 points != final score and trip the hard gate spuriously.
with in_scope as (
    select * from {{ ref('dim_games') }}
    where game_id not in (select game_id from {{ source('recon', 'recon_quarantine') }})
),

team_game as (
    select game_id, home_team_id as team_id, home_final_score as final_score
    from in_scope
    union all
    select game_id, away_team_id as team_id, away_final_score as final_score
    from in_scope
),

box_team as (
    select
        game_id,
        team_id,
        sum(fga)  as fga,
        sum(fta)  as fta,
        sum(oreb) as oreb,
        sum(tov)  as tov
    from {{ ref('stg_box_player') }}
    group by 1, 2
),

counted as (
    select
        game_id,
        offense_team_id as team_id,
        count(*)        as counted_possessions,
        sum(points)     as possession_points
    from {{ source('recon', 'recon_possessions') }}
    group by 1, 2
)

select
    tg.game_id,
    tg.team_id,
    coalesce(c.counted_possessions, 0)                       as counted_possessions,
    bt.fga - bt.oreb + bt.tov + 0.44 * bt.fta               as estimated_possessions,
    coalesce(c.possession_points, 0)                        as possession_points,
    tg.final_score,
    coalesce(c.possession_points, 0) = tg.final_score        as points_reconciliation_ok
from team_game tg
left join box_team bt on tg.game_id = bt.game_id and tg.team_id = bt.team_id
left join counted  c  on tg.game_id = c.game_id  and tg.team_id = c.team_id
