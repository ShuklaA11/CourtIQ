-- Leakage-safe state immediately BEFORE each reconstructed possession begins.
--
-- The score uses only preceding possession points (ROWS ... 1 PRECEDING), never
-- the current possession outcome. Player strength comes only from season S-1.
-- Missing prior ratings are the Gaussian-prior mean (0 points / 100) and are
-- surfaced explicitly in rapm_coverage_status rather than silently imputed.
{{ config(enabled=var('enable_winprob', false)) }}

with possessions as (
    select
        p.*,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        g.home_final_score,
        g.away_final_score,
        case
            when p.offense_team_id = g.home_team_id then p.offense_five
            else p.defense_five
        end as home_five,
        case
            when p.offense_team_id = g.away_team_id then p.offense_five
            else p.defense_five
        end as away_five
    from {{ ref('fct_possessions') }} p
    join {{ ref('dim_games') }} g using (game_id)
),

pre_possession as (
    select
        *,
        -- Canonical model score: possession-attributed points strictly before
        -- this row. It is monotone and exactly reconciles to the box score.
        -- The raw feed scoreboard is retained separately for anomaly auditing;
        -- 25 feed corrections decrease it, so it cannot satisfy the model gate.
        coalesce(sum(
            case when offense_team_id = home_team_id then points else 0 end
        ) over (
            partition by game_id
            order by period, possession_number
            rows between unbounded preceding and 1 preceding
        ), 0)::integer as home_score,
        coalesce(sum(
            case when offense_team_id = away_team_id then points else 0 end
        ) over (
            partition by game_id
            order by period, possession_number
            rows between unbounded preceding and 1 preceding
        ), 0)::integer as away_score,
        home_score_before as feed_home_score_before,
        away_score_before as feed_away_score_before
    from possessions
),

state as (
    select
        *,
        case
            when period <= 4 then 720.0 - start_seconds
            else 300.0 - start_seconds
        end as game_clock_seconds,
        case
            when period <= 4 then (period - 1) * 720.0 + start_seconds
            else 2880.0 + (period - 5) * 300.0 + start_seconds
        end as elapsed_game_seconds,
        case
            when season in (2022, 2023) then 'train'
            when season = 2024 then 'validation'
            when season = 2025 then 'test'
            else 'audit_only'
        end as split
    from pre_possession
),

ratings as (
    select
        cast(player_id as bigint) as player_id,
        cast(season as integer) as season,
        cast(off_rating as double) as off_rating,
        cast(def_rating as double) as def_rating,
        cast(net_rating as double) as net_rating
    from read_parquet('{{ var("rapm_ratings_path") }}')
),

home_expanded as (
    select
        game_id,
        period,
        possession_number,
        season,
        unnest(home_five) as player_id
    from state
),

home_players as (
    select
        p.game_id,
        p.period,
        p.possession_number,
        count(r.player_id) as rated_players,
        sum(coalesce(r.off_rating, 0.0)) as lineup_off_rapm,
        sum(coalesce(r.def_rating, 0.0)) as lineup_def_rapm,
        sum(coalesce(r.net_rating, 0.0)) as lineup_net_rapm
    from home_expanded p
    left join ratings r
      on r.player_id = p.player_id
     and r.season = p.season - 1
    group by 1, 2, 3
),

away_expanded as (
    select
        game_id,
        period,
        possession_number,
        season,
        unnest(away_five) as player_id
    from state
),

away_players as (
    select
        p.game_id,
        p.period,
        p.possession_number,
        count(r.player_id) as rated_players,
        sum(coalesce(r.off_rating, 0.0)) as lineup_off_rapm,
        sum(coalesce(r.def_rating, 0.0)) as lineup_def_rapm,
        sum(coalesce(r.net_rating, 0.0)) as lineup_net_rapm
    from away_expanded p
    left join ratings r
      on r.player_id = p.player_id
     and r.season = p.season - 1
    group by 1, 2, 3
)

select
    s.game_id,
    s.season,
    s.game_date,
    s.period,
    s.possession_number,
    s.game_clock_seconds,
    s.elapsed_game_seconds,
    greatest(0.0, 2880.0 - s.elapsed_game_seconds) as regulation_seconds_remaining,
    s.home_team_id,
    s.away_team_id,
    s.offense_team_id as possession_team_id,
    s.offense_team_id = s.home_team_id as home_has_possession,
    s.home_score,
    s.away_score,
    s.home_score - s.away_score as home_score_differential,
    s.feed_home_score_before,
    s.feed_away_score_before,
    s.home_five,
    s.away_five,
    s.home_final_score > s.away_final_score as home_win,
    case when s.season = 2021 then null else s.season - 1 end as rapm_source_season,
    round(hp.lineup_off_rapm, 10) as home_lineup_off_rapm,
    round(hp.lineup_def_rapm, 10) as home_lineup_def_rapm,
    round(hp.lineup_net_rapm, 10) as home_lineup_net_rapm,
    round(ap.lineup_off_rapm, 10) as away_lineup_off_rapm,
    round(ap.lineup_def_rapm, 10) as away_lineup_def_rapm,
    round(ap.lineup_net_rapm, 10) as away_lineup_net_rapm,
    round(hp.lineup_net_rapm - ap.lineup_net_rapm, 10)
        as lineup_net_rapm_differential,
    case
        when s.season = 2021 then 'cold_start'
        when hp.rated_players = 5 and ap.rated_players = 5 then 'full'
        when hp.rated_players = 0 and ap.rated_players = 0 then 'replacement_only'
        else 'partial'
    end as rapm_coverage_status,
    hp.rated_players as home_rated_players,
    ap.rated_players as away_rated_players,
    10 - hp.rated_players - ap.rated_players as replacement_player_appearances,
    s.split
from state s
join home_players hp using (game_id, period, possession_number)
join away_players ap using (game_id, period, possession_number)
order by s.game_id, s.period, s.possession_number
