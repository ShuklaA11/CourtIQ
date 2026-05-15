-- GATE: for every team-game, the points summed across that team's possessions
-- must equal its official final score. Any row returned is a game where the
-- event walk miscredited or lost points — data we must not model on.
select
    game_id,
    team_id,
    possession_points,
    final_score
from {{ ref('val_possessions') }}
where possession_points <> final_score
