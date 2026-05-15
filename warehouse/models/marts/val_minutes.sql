-- Validation: reconstructed minutes vs. official box minutes, per player-game.
--
-- Reconstructed seconds = the sum of every stint a player appears in (explode
-- the five, sum the stint seconds). The residual against box_seconds is the
-- headline signal that lineup tracking is correct: if subs were parsed cleanly
-- and no one is stranded on the floor, residuals sit near zero.
with stint_seconds as (
    select
        game_id,
        team_id,
        unnest(five) as person_id,
        seconds
    from {{ source('recon', 'recon_stints') }}
),

reconstructed as (
    select
        game_id,
        team_id,
        person_id,
        sum(seconds) as reconstructed_seconds
    from stint_seconds
    group by 1, 2, 3
)

select
    b.game_id,
    b.team_id,
    b.person_id,
    b.player_name,
    coalesce(r.reconstructed_seconds, 0)          as reconstructed_seconds,
    b.box_seconds,
    coalesce(r.reconstructed_seconds, 0) - b.box_seconds as residual_seconds
from {{ ref('stg_box_player') }} b
left join reconstructed r
    on  b.game_id   = r.game_id
    and b.team_id   = r.team_id
    and b.person_id = r.person_id
-- Players who logged minutes (DNPs have nothing to reconstruct), reconstructed
-- games only (quarantined games have no stints, so every player would read as a
-- full-minutes miss and swamp the aggregate residual gate).
where b.box_seconds > 0
  and b.game_id not in (select game_id from {{ source('recon', 'recon_quarantine') }})
