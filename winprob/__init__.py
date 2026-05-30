"""Win-probability modeling on the leakage-safe possession game-state mart.

Sprint 3. `features` turns `fct_game_states` rows into a numeric logistic design
matrix; `model` fits the regularized logistic win-probability model against it.
The transform is pure (arrays in, arrays out) and deliberately excludes the
RAPM/lineup columns, the raw feed_* columns, and the `home_win` target so nothing
leaks into the fit. `design` exports and audits the underlying game-state mart.
"""
