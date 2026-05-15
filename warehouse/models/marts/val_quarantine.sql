-- The quarantine list: every game that fails any validation gate, one row per
-- (game, reason). Downstream modeling joins against this to exclude tainted
-- games. The same three conditions the singular tests hard-fail on are surfaced
-- here as data, so a game can be inspected instead of just blocking the build.
with build_quarantine as (
    -- The authoritative list: games recon.build could not reconstruct exactly-five
    -- (OT lineups invisible to play-by-play) and excluded from the fact tables.
    select game_id, reason from {{ source('recon', 'recon_quarantine') }}
),

bad_lineups as (
    select distinct game_id, 'lineup_not_five' as reason
    from {{ source('recon', 'recon_stints') }}
    where len(five) <> 5
),

points_mismatch as (
    select distinct game_id, 'possession_points_mismatch' as reason
    from {{ ref('val_possessions') }}
    where not points_reconciliation_ok
),

minutes_off as (
    select distinct game_id, 'minutes_residual_exceeds_tolerance' as reason
    from {{ ref('val_minutes') }}
    where abs(residual_seconds) > {{ var('minutes_tolerance_seconds') }}
)

select game_id, reason from build_quarantine
union all
select game_id, reason from bad_lineups
union all
select game_id, reason from points_mismatch
union all
select game_id, reason from minutes_off
