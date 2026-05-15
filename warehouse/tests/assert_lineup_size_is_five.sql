-- GATE: every on-floor lineup must be exactly five players.
--
-- A singular test fails when it returns rows. It checks two grains so nothing
-- slips through: the stint segments (every team-period lineup) and the fives
-- snapshotted onto each possession. Referencing fct_possessions also keeps this
-- test inside `--select marts` indirect selection, so it runs in the same
-- `dbt build` that gates the pipeline.
select game_id, period, 'stint' as grain, len(five) as lineup_size
from {{ source('recon', 'recon_stints') }}
where len(five) <> 5

union all

select game_id, period, 'possession_offense' as grain, len(offense_five) as lineup_size
from {{ ref('fct_possessions') }}
where len(offense_five) <> 5

union all

select game_id, period, 'possession_defense' as grain, len(defense_five) as lineup_size
from {{ ref('fct_possessions') }}
where len(defense_five) <> 5
