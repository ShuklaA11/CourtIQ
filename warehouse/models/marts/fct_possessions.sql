-- The analysis-ready possession grain: one row per possession, everything a
-- downstream RAPM / win-probability model needs about who was on the floor and
-- what happened. A near-passthrough of recon_possessions, typed and ordered,
-- kept as its own model so the reconstruction table stays an implementation
-- detail behind a stable mart.
select
    game_id,
    season,
    period,
    possession_number,
    offense_team_id,
    defense_team_id,
    offense_five,          -- five person_ids on offense
    defense_five,          -- five person_ids on defense
    points,                -- points scored by the offense on this possession
    duration_seconds
from {{ source('recon', 'recon_possessions') }}
order by game_id, period, possession_number
