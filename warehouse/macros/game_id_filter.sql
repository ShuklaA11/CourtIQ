{#
  Renders a SQL predicate that keeps only the game_ids in scope, driven by the
  `game_id_patterns` var. Centralizing it here keeps every staging model scoped
  identically and lets the season fan-out happen in one place (the var).

  Usage:  where {{ game_id_filter('game_id') }}
#}
{% macro game_id_filter(column) %}
  {%- set patterns = var('game_id_patterns') -%}
  (
  {%- for pattern in patterns %}
    {{ column }} like '{{ pattern }}'{% if not loop.last %} or{% endif %}
  {%- endfor %}
  )
{% endmacro %}
