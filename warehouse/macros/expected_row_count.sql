{#
  Generic schema test: fails unless the model has exactly `expected` rows.
  Used to pin dim_games to the game count implied by the scope var
  (game_id_patterns). dbt tests pass when the query returns zero rows, so we
  emit the actual count only when it differs from the expectation.

  Usage (model-level in schema.yml):
    data_tests:
      - expected_row_count:
          arguments:
            expected: "{{ var('expected_game_count', 1312) }}"
#}
{% test expected_row_count(model, expected) %}
select count(*) as actual_row_count
from {{ model }}
having count(*) != {{ expected }}
{% endtest %}
