-- One-row headline scorecard for the reconstruction. This is the number you
-- read first: how much data survived, and how tight the minutes fit is.
with totals as (
    select count(*) as total_possessions from {{ ref('fct_possessions') }}
),

games as (
    select count(*) as total_games from {{ ref('dim_games') }}
),

quarantined as (
    select count(distinct game_id) as quarantined_games from {{ ref('val_quarantine') }}
),

minutes as (
    select
        quantile_cont(abs(residual_seconds), 0.50) as residual_p50,
        quantile_cont(abs(residual_seconds), 0.95) as residual_p95,
        max(abs(residual_seconds))                 as residual_max
    from {{ ref('val_minutes') }}
),

reconciliation as (
    select
        avg(case when points_reconciliation_ok then 1.0 else 0.0 end)
            as points_reconciliation_pass_rate
    from {{ ref('val_possessions') }}
)

select
    t.total_possessions,
    g.total_games,
    q.quarantined_games,
    q.quarantined_games::double / nullif(g.total_games, 0) as quarantine_rate,
    m.residual_p50   as minutes_residual_p50_seconds,
    m.residual_p95   as minutes_residual_p95_seconds,
    m.residual_max   as minutes_residual_max_seconds,
    r.points_reconciliation_pass_rate
from totals t
cross join games g
cross join quarantined q
cross join minutes m
cross join reconciliation r
