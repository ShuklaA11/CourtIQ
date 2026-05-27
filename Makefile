# CourtIQ reconstruction + validation pipeline.
#
# Thin wrapper over run.sh (the single source of truth for the 3-step pipeline:
# dbt staging -> recon.build -> dbt marts + gates). Scope defaults to 2023-24;
# `make all-seasons` fans out to every ingested season.

ALL_SEASONS := ["00221%","00421%","00222%","00422%","00223%","00423%","00224%","00424%","00225%","00425%"]

.PHONY: pipeline all-seasons game-states clean

## Full pipeline on the default scope (2023-24).
pipeline:
	./run.sh

## Full pipeline over every ingested season.
all-seasons:
	./run.sh '$(ALL_SEASONS)'

## Build, gate, export, and audit the Sprint 3 possession-boundary feature mart.
## Requires the all-season warehouse and data/rapm/bayes_ratings.parquet.
game-states:
	./game_states.sh

## Remove the DuckDB warehouse and dbt build artifacts.
clean:
	rm -f warehouse/courtiq.duckdb warehouse/courtiq.duckdb.wal
	rm -rf warehouse/target warehouse/logs warehouse/dbt_packages
