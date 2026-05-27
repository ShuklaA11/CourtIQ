-- Sprint 3 Phase 1 hard gates. Any returned row fails dbt build.
{{ config(enabled=var('enable_winprob', false)) }}

with duplicate_keys as (
    select game_id, period, possession_number, 'duplicate_key' as reason
    from {{ ref('fct_game_states') }}
    group by 1, 2, 3
    having count(*) <> 1
),

bad_rows as (
    select * from (
        select game_id, period, possession_number,
            case
                when len(home_five) <> 5 or len(away_five) <> 5 then 'lineup_not_five'
                when home_has_possession <> (possession_team_id = home_team_id)
                    then 'possession_flag_mismatch'
            when game_clock_seconds < 0 then 'negative_game_clock'
            when period <= 4 and game_clock_seconds > 720 then 'invalid_game_clock'
            when period > 4 and game_clock_seconds > 300 then 'invalid_ot_clock'
            when elapsed_game_seconds < 0 then 'negative_elapsed_time'
            when regulation_seconds_remaining
                 <> greatest(0.0, 2880.0 - elapsed_game_seconds)
                then 'regulation_clock_mismatch'
                when rapm_coverage_status is null then 'missing_coverage_status'
            when rapm_coverage_status <> 'cold_start'
                 and rapm_source_season <> season - 1 then 'rapm_source_not_prior_season'
                else null
            end as reason
        from {{ ref('fct_game_states') }}
    )
    where reason is not null
),

nonmonotonic as (
    select game_id, period, possession_number, 'state_not_monotonic' as reason
    from (
        select
            *,
            lag(home_score) over (
                partition by game_id order by period, possession_number
            ) as prev_home_score,
            lag(away_score) over (
                partition by game_id order by period, possession_number
            ) as prev_away_score,
            lag(elapsed_game_seconds) over (
                partition by game_id order by period, possession_number
            ) as prev_elapsed
        from {{ ref('fct_game_states') }}
    )
    where home_score < prev_home_score
       or away_score < prev_away_score
       or elapsed_game_seconds < prev_elapsed
),

split_leakage as (
    select game_id, min(period) as period, min(possession_number) as possession_number,
           'game_spans_splits' as reason
    from {{ ref('fct_game_states') }}
    group by game_id
    having count(distinct split) <> 1
),

missing_possessions as (
    select coalesce(p.game_id, s.game_id) as game_id,
           coalesce(p.period, s.period) as period,
           coalesce(p.possession_number, s.possession_number) as possession_number,
           'possession_coverage_mismatch' as reason
    from {{ ref('fct_possessions') }} p
    full outer join {{ ref('fct_game_states') }} s
      using (game_id, period, possession_number)
    where p.game_id is null or s.game_id is null
),

final_state as (
    select
        s.game_id,
        sum(case when p.offense_team_id = g.home_team_id then p.points else 0 end)
            as reconstructed_home_final,
        sum(case when p.offense_team_id = g.away_team_id then p.points else 0 end)
            as reconstructed_away_final,
        max(g.home_final_score) as official_home_final,
        max(g.away_final_score) as official_away_final,
        max(s.home_win::integer) as reconstructed_home_win
    from {{ ref('fct_game_states') }} s
    join {{ ref('fct_possessions') }} p
      using (game_id, period, possession_number)
    join {{ ref('dim_games') }} g using (game_id)
    group by s.game_id
),

bad_final as (
    select game_id, 0 as period, 0 as possession_number,
           'final_score_or_winner_mismatch' as reason
    from final_state
    where reconstructed_home_final <> official_home_final
       or reconstructed_away_final <> official_away_final
       or reconstructed_home_win <> (official_home_final > official_away_final)::integer
)

select * from duplicate_keys
union all select * from bad_rows
union all select * from nonmonotonic
union all select * from split_leakage
union all select * from missing_possessions
union all select * from bad_final
