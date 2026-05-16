"""
data_cleaning/validators.py
===========================

PURPOSE:
    Checks the master DataFrame for data quality problems BEFORE any
    cleaning or feature engineering happens.

    Think of this as a "health check" that runs at the start of the
    pipeline.  If something is seriously wrong with the data, it raises
    a ValueError immediately and stops everything — so you never
    accidentally train a model on garbage data.

WHY RAISE AN ERROR INSTEAD OF JUST LOGGING?
    Logging a warning and continuing might seem friendlier, but it means
    broken data silently flows through the whole pipeline and corrupts
    your model without you realising it.  An early, loud error is much
    easier to debug than mysterious bad predictions later.

    Think of it like a safety check before surgery — you want to catch
    problems BEFORE you start, not notice them at the end.

WHAT GETS CHECKED:
    1. Required columns exist              (crash early if data_intake changed)
    2. No fully empty DataFrame            (nothing to work with)
    3. No duplicate player-season rows     (would inflate stats)
    4. Impossible stat values              (negative yards, >17 games, etc.)
    5. Position values are all known       (typos / new positions from API)
    6. Season years are in a sane range    (not 1995, not 2099)

HOW TO USE:
    from src.data_cleaning import DataValidator

    validator = DataValidator()
    validator.validate(master_df)   # raises ValueError if anything is wrong
                                    # returns None silently if all checks pass
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# These are the columns that MUST exist in the master DataFrame.
# If data_intake changes and drops one of these, the validator will catch it
# immediately rather than letting the error surface as a confusing KeyError
# somewhere deep in the pipeline.
REQUIRED_COLUMNS = [
    "player_id",
    "full_name",
    "position",
    "season",
    "nfl_team",
    "college",
    "games_played",
    "years_exp",
    "draft_year",
    "pass_yards",
    "rush_yards",
    "rec_yards",
    "fantasy_pts_half_ppr",
    "fantasy_pts_ppr",
]

# The only position values we expect to see.
# If a new value appears (e.g. "FB" for fullback), we want to know about it.
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# NFL regular seasons have been 16 or 17 games since 1978.
# A player cannot have played more games than the season length.
# We use 18 as the upper limit (current schedule length) + 1 if traded.
MAX_GAMES_IN_SEASON = 18

# Sanity range for season years.
# Sleeper data starts around 2015; we add a buffer for future seasons.
MIN_VALID_SEASON = 2015
MAX_VALID_SEASON = 2035

# Stat columns that must never be negative.
# IMPORTANT — yards and fantasy points are intentionally excluded here because
# they CAN legitimately be negative in the NFL:
#   pass_yards        → negative on a sack behind the line of scrimmage
#   rush_yards        → negative on a tackle for loss
#   rec_yards         → negative on a catch behind the LOS tackled for loss
#   fantasy_pts_*     → negative when turnovers outweigh positive plays
# We only enforce non-negativity on counts (attempts, completions, TDs, games).
NON_NEGATIVE_STAT_COLUMNS = [
    "games_played",
    "pass_attempts", "pass_completions", "pass_touchdowns",
    "rush_attempts",  "rush_touchdowns",
    "targets", "receptions", "rec_touchdowns",
]


# ---------------------------------------------------------------------------
# VALIDATOR CLASS
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Runs a series of data quality checks on the master DataFrame.

    All checks raise a ValueError with a clear message if they fail.
    If all checks pass, validate() returns None silently.

    EXAMPLE:
        validator = DataValidator()
        validator.validate(master_df)
        # If this line is reached, the data passed all checks.
        print("Data is clean — proceeding to cleaning step.")
    """

    def validate(self, df: pd.DataFrame) -> None:
        """
        Runs all validation checks in order.

        PARAMETERS:
            df (pd.DataFrame) : The master DataFrame from DataCombiner.

        RETURNS:
            None  — silently if all checks pass.

        RAISES:
            ValueError — with a descriptive message on the FIRST check that fails.

        WHY STOP ON THE FIRST FAILURE?
            Multiple checks might fail for the same root cause.
            Showing every failure at once can be overwhelming.
            Fix the first error, re-run, fix the next — one at a time.
        """
        logger.info("Starting data validation…")

        self._check_not_empty(df)
        self._check_required_columns(df)
        self._check_no_duplicates(df)
        self._check_valid_positions(df)
        self._check_season_range(df)
        self._check_non_negative_stats(df)
        self._check_games_played_cap(df)

        logger.info("All validation checks passed ✓")

    # ── Private check methods ────────────────────────────────────────────────
    # Each method checks one specific thing and raises ValueError if it fails.
    # Keeping them separate makes it easy to add, remove, or adjust checks
    # without touching the others.

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        """
        Fails if the DataFrame has zero rows.

        WHY THIS MATTERS:
            An empty DataFrame usually means the API calls in data_intake
            all failed silently.  Everything downstream would break in
            confusing ways, so we catch it here with a clear message.
        """
        if df.empty:
            raise ValueError(
                "Validation failed: The master DataFrame is completely empty.\n"
                "This usually means all API calls in data_intake failed.\n"
                "Check your internet connection and re-run data_intake."
            )
        logger.info("  ✓ DataFrame is not empty (%d rows).", len(df))

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """
        Fails if any required column is missing from the DataFrame.

        WHY THIS MATTERS:
            If data_intake is updated and accidentally drops a column,
            every downstream step that uses that column will crash with a
            confusing KeyError.  This check makes the problem obvious
            immediately, with a clear list of what's missing.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Validation failed: The following required columns are missing:\n"
                f"  {missing}\n"
                f"This may mean data_intake was updated and dropped these fields.\n"
                f"Check player_data.py, team_data.py, combine.py, or college_data.py."
            )
        logger.info("  ✓ All %d required columns present.", len(REQUIRED_COLUMNS))

    def _check_no_duplicates(self, df: pd.DataFrame) -> None:
        """
        Fails if any (player_id, season) pair appears more than once.

        WHY THIS MATTERS:
            Duplicate rows would double-count stats and make a player look
            twice as productive as they really were.  This is a common
            side-effect of a bad merge (join) in data_intake.

        WHAT IS duplicated()?
            df.duplicated(subset=[...]) returns a True/False Series where
            True means "this row has the same values in those columns as
            a previous row."  We count how many Trues there are.
        """
        dupe_count = df.duplicated(subset=["player_id", "season"]).sum()
        if dupe_count > 0:
            # Show the user WHICH players are duplicated so they can investigate.
            dupes = df[df.duplicated(subset=["player_id", "season"], keep=False)]
            sample = dupes[["player_id", "full_name", "season"]].drop_duplicates().head(10)
            raise ValueError(
                f"Validation failed: {dupe_count} duplicate (player_id, season) rows found.\n"
                f"Sample duplicates:\n{sample.to_string(index=False)}\n"
                f"Check the merge logic in data_intake/combine.py."
            )
        logger.info("  ✓ No duplicate player-season rows.")

    def _check_valid_positions(self, df: pd.DataFrame) -> None:
        """
        Fails if any position value is not in our known valid set.

        WHY THIS MATTERS:
            An unexpected position (e.g. "FB", "K", or a typo like "QBB")
            means either the API returned unexpected data or our filter in
            data_intake stopped working.  Better to know now.

        WHAT IS dropna()?
            Some players may have NaN for position (missing data).
            We skip those here — NaN positions are caught as a separate
            issue in data_cleaning.
        """
        actual_positions = set(df["position"].dropna().unique())
        unknown = actual_positions - VALID_POSITIONS
        if unknown:
            raise ValueError(
                f"Validation failed: Unknown position values found: {unknown}\n"
                f"Expected only: {VALID_POSITIONS}\n"
                f"If these are valid new positions, add them to VALID_POSITIONS "
                f"in validators.py."
            )
        logger.info("  ✓ All positions are valid: %s.", actual_positions)

    def _check_season_range(self, df: pd.DataFrame) -> None:
        """
        Fails if any season year is outside the expected range.

        WHY THIS MATTERS:
            A season value of 1995 or 2099 means something went badly wrong
            in the data — likely a parsing error in data_intake.
        """
        out_of_range = df[
            (df["season"] < MIN_VALID_SEASON) | (df["season"] > MAX_VALID_SEASON)
        ]
        if not out_of_range.empty:
            bad_seasons = out_of_range["season"].unique()
            raise ValueError(
                f"Validation failed: {len(out_of_range)} rows have season values "
                f"outside the valid range ({MIN_VALID_SEASON}–{MAX_VALID_SEASON}).\n"
                f"Bad season values found: {bad_seasons}\n"
                f"Check CURRENT_NFL_SEASON in data_intake/links.py."
            )
        season_range = f"{df['season'].min()}–{df['season'].max()}"
        logger.info("  ✓ All season years are in range (%s).", season_range)

    def _check_non_negative_stats(self, df: pd.DataFrame) -> None:
        """
        Warns if any count stat column contains negative values.

        WHY WARN INSTEAD OF RAISE?
            Yardage columns (pass_yards, rush_yards, rec_yards) and fantasy
            points CAN legitimately be negative in the NFL — sacks behind
            the line of scrimmage, tackles for loss, and turnover-heavy
            games all produce negative values.  We only enforce strict
            non-negativity on count columns (attempts, completions, TDs,
            games played) where a negative value is truly impossible.

        NOTE:
            We only check columns that actually exist in the DataFrame.
            Some stat columns may not be present for all data pulls —
            we skip missing ones gracefully.
        """
        cols_to_check = [c for c in NON_NEGATIVE_STAT_COLUMNS if c in df.columns]

        for col in cols_to_check:
            negative_mask  = df[col].lt(0)
            negative_count = negative_mask.sum()

            if negative_count > 0:
                sample = df[negative_mask][["full_name", "season", col]].head(5)
                raise ValueError(
                    f"Validation failed: {negative_count} negative values found "
                    f"in column '{col}'.\n"
                    f"Sample rows:\n{sample.to_string(index=False)}\n"
                    f"This column should never be negative — check data_intake."
                )

        logger.info("  ✓ No negative values in count stat columns.")

    def _check_games_played_cap(self, df: pd.DataFrame) -> None:
        """
        Fails if any player is recorded as having played more games than
        the maximum possible in an NFL season (17).

        WHY THIS MATTERS:
            games_played > 17 is physically impossible and would corrupt
            any per-game average we calculate in feature_engineer.py.
        """
        over_cap = df[df["games_played"] > MAX_GAMES_IN_SEASON]
        if not over_cap.empty:
            sample = over_cap[["full_name", "season", "games_played"]].head(5)
            raise ValueError(
                f"Validation failed: {len(over_cap)} rows have games_played > "
                f"{MAX_GAMES_IN_SEASON} (the maximum possible in an NFL season).\n"
                f"Sample rows:\n{sample.to_string(index=False)}\n"
                f"Check the 'gp' stat field handling in data_intake/player_data.py."
            )
        logger.info(
            "  ✓ All games_played values are within the season cap (%d).",
            MAX_GAMES_IN_SEASON,
        )