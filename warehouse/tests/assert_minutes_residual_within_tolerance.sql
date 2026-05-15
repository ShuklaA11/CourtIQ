-- GATE: reconstructed minutes must sit within tolerance of box minutes for all
-- but a small fraction of player-games. Unlike the other two gates this one is
-- aggregate — a few noisy player-games are tolerable, a systemic drift is not —
-- so the query returns a single row *only when* the share of player-games whose
-- residual exceeds minutes_tolerance_seconds is above minutes_residual_max_fraction
-- (default 1%). No row => within tolerance => test passes.
with flagged as (
    select
        case
            when abs(residual_seconds) > {{ var('minutes_tolerance_seconds') }}
            then 1 else 0
        end as over_tolerance
    from {{ ref('val_minutes') }}
)

select
    count(*)                                             as player_games,
    sum(over_tolerance)                                  as over_tolerance_count,
    sum(over_tolerance)::double / nullif(count(*), 0)    as over_tolerance_fraction
from flagged
having sum(over_tolerance)::double / nullif(count(*), 0)
     > {{ var('minutes_residual_max_fraction') }}
