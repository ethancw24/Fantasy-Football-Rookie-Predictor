"""
data_cleaning/feature_engineer.py
==================================

PURPOSE:
    Creates NEW columns derived from existing clean data.
    This step runs AFTER cleaner.py — it assumes the data is already
    trustworthy (no bad values, no missing positions, names normalised, etc.)

THE RULE:
    cleaner.py          →  fix or remove bad/missing data
    feature_engineer.py →  create new columns from clean data

    If you're FIXING something → it belongs in cleaner.py.
    If you're CREATING something new from good data → it belongs here.

WHAT THIS FILE CREATES:

    PER-GAME STATS (alongside raw totals):
        Raw season totals are misleading for players who missed games due
        to injury.  A player with 800 yards in 8 games is more productive
        than one with 900 yards in 17 games.  Per-game stats make fair
        comparisons possible.
        Columns added: pass_yards_per_game, rush_yards_per_game,
                       rec_yards_per_game, fantasy_pts_half_ppr_per_game,
                       fantasy_pts_ppr_per_game, etc.

    ROOKIE FLAG:
        A boolean column (is_rookie) that is True when years_exp == 0.
        This lets the model learn different patterns for rookies vs.
        veterans without manual filtering.

    COLLEGE-TO-NFL STAT RATIOS (for players with college data):
        How did a player's college production compare to their NFL output?
        Columns added: college_to_nfl_pass_yards_ratio,
                       college_to_nfl_rush_yards_ratio,
                       college_to_nfl_rec_yards_ratio.
        NaN where college data is missing — the model handles that.

    POSITION-AVERAGE COMPARISONS:
        How does a player perform relative to the average player at their
        position in the same season?
        Columns added: pass_yards_vs_pos_avg, rush_yards_vs_pos_avg,
                       rec_yards_vs_pos_avg, fantasy_pts_half_ppr_vs_pos_avg,
                       fantasy_pts_ppr_vs_pos_avg.
        A value of +200 means 200 yards above the position average.
        A value of -50 means 50 yards below.

HOW TO USE:
    from src.data_cleaning import FeatureEngineer

    engineer = FeatureEngineer()
    final_df = engineer.engineer(clean_df)
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# A season is considered "injury-shortened" if the player played fewer than
# this many games.  Used in per-game calculations and logging.
# (Half a 17-game season, rounded down.)
SHORT_SEASON_THRESHOLD = 8

# Stat columns to create per-game versions of.
# Each entry is (raw_total_column, per_game_column_name).
PER_GAME_STATS = [
    ("pass_attempts",           "pass_attempts_per_game"),
    ("pass_yards",              "pass_yards_per_game"),
    ("pass_touchdowns",         "pass_tds_per_game"),
    ("rush_attempts",           "rush_attempts_per_game"),
    ("rush_yards",              "rush_yards_per_game"),
    ("rush_touchdowns",         "rush_tds_per_game"),
    ("targets",                 "targets_per_game"),
    ("receptions",              "receptions_per_game"),
    ("rec_yards",               "rec_yards_per_game"),
    ("rec_touchdowns",          "rec_tds_per_game"),
    ("fantasy_pts_half_ppr",    "fantasy_pts_half_ppr_per_game"),
    ("fantasy_pts_ppr",         "fantasy_pts_ppr_per_game"),
]

# Stat columns to compare against position averages.
POSITION_AVG_STATS = [
    "pass_yards",
    "rush_yards",
    "rec_yards",
    "fantasy_pts_half_ppr",
    "fantasy_pts_ppr",
]

# College stat → NFL stat pairs for ratio calculations.
# (college_column, nfl_column, ratio_output_column)
# These only produce values for players who have college data.
COLLEGE_TO_NFL_RATIOS = [
    ("college_pass_yards",  "pass_yards",   "college_to_nfl_pass_yards_ratio"),
    ("college_rush_yards",  "rush_yards",   "college_to_nfl_rush_yards_ratio"),
    ("college_rec_yards",   "rec_yards",    "college_to_nfl_rec_yards_ratio"),
]


# ---------------------------------------------------------------------------
# FEATURE ENGINEER CLASS
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Adds derived feature columns to the cleaned master DataFrame.

    EXAMPLE:
        engineer = FeatureEngineer()
        final_df = engineer.engineer(clean_df)

        # See all new columns added:
        original_cols = set(clean_df.columns)
        new_cols = [c for c in final_df.columns if c not in original_cols]
        print(new_cols)
    """

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all feature engineering steps and returns an enriched copy.

        PARAMETERS:
            df (pd.DataFrame) : The cleaned DataFrame from DataCleaner.

        RETURNS:
            pd.DataFrame : Same rows, with new feature columns added.
                           Original columns are preserved.
        """
        logger.info("Starting feature engineering…")

        # Always work on a copy — never mutate the original
        df = df.copy()

        df = self._add_per_game_stats(df)
        df = self._add_rookie_flag(df)
        df = self._add_college_to_nfl_ratios(df)
        df = self._add_position_average_comparisons(df)

        logger.info(
            "Feature engineering complete → %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df.reset_index(drop=True)

    # ── Private feature steps ────────────────────────────────────────────────

    def _add_per_game_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds per-game versions of key stat columns alongside the raw totals.

        FORMULA:
            per_game_stat = raw_total / games_played

        EXAMPLE:
            pass_yards = 4000, games_played = 17
            pass_yards_per_game = 4000 / 17 = 235.3

        WHY KEEP THE RAW TOTALS?
            Totals still carry information — a player with 1000 yards in
            8 games is extraordinary, but 1000 yards in 17 games is solid.
            Both the rate (per game) and the volume (total) tell a story.

        DIVISION SAFETY:
            We never divide by zero.  If games_played is 0, the result is
            NaN instead of crashing.  We use .replace(0, pd.NA) to convert
            any zero games_played to NaN before dividing.

        ROUNDING:
            Results are rounded to 2 decimal places for readability.
        """
        logger.info("  Adding per-game stats…")

        # Replace 0 with NaN so division doesn't produce Inf or crash.
        # We make a temporary series — we don't change the original games_played.
        safe_games = df["games_played"].replace(0, pd.NA)

        added = 0
        for raw_col, per_game_col in PER_GAME_STATS:
            if raw_col not in df.columns:
                logger.warning("    Skipping '%s' — column not found.", raw_col)
                continue

            df[per_game_col] = (df[raw_col] / safe_games).round(2)
            added += 1

        # Log how many players had injury-shortened seasons (for awareness)
        short_season_count = (df["games_played"] < SHORT_SEASON_THRESHOLD).sum()
        if short_season_count > 0:
            logger.info(
                "    Note: %d player-seasons had fewer than %d games played "
                "(injury-shortened seasons — per-game stats help account for this).",
                short_season_count, SHORT_SEASON_THRESHOLD,
            )

        logger.info("    Added %d per-game columns.", added)
        return df

    def _add_rookie_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a boolean column 'is_rookie' that is True when years_exp == 0.

        WHY:
            Rookies and veterans follow very different production patterns.
            A 1st-year QB typically underperforms relative to their college
            stats; a 5-year veteran QB's college stats are ancient history.
            Flagging rookies lets the model learn these different patterns.

        WHAT IS years_exp == 0?
            Sleeper sets years_exp = 0 for players in their first NFL season.
            This includes both drafted rookies AND undrafted free agents
            (UDFAs) in their debut year — both are "rookies" for our purposes.

        WHAT IS .fillna(False)?
            If years_exp is NaN for some players, the == comparison returns
            NaN rather than True/False.  .fillna(False) converts those NaNs
            to False — we only flag someone as a rookie if we're sure.
        """
        df["is_rookie"] = (df["years_exp"] == 0).fillna(False)
        rookie_count = df["is_rookie"].sum()
        logger.info("  Added 'is_rookie' flag — %d rookie player-seasons found.", rookie_count)
        return df

    def _add_college_to_nfl_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds ratio columns comparing a player's college production to their
        NFL production in the same statistical category.

        FORMULA:
            ratio = college_stat / nfl_stat

        EXAMPLES:
            college_pass_yards = 4000, pass_yards (NFL) = 4500
            college_to_nfl_pass_yards_ratio = 4000 / 4500 = 0.89
            → Player produced 89% as many yards in college as in the NFL

            college_rush_yards = 1800, rush_yards (NFL) = 900
            college_to_nfl_rush_yards_ratio = 1800 / 900 = 2.0
            → Player produced TWICE as many yards in college than in NFL year 1

        WHY THIS IS USEFUL:
            A player who dominated in college and immediately dominated in
            the NFL has a ratio near 1.0.  A player who struggled in the
            NFL relative to college might have a ratio > 2.0.
            Over time, the model can learn which college production levels
            tend to translate well to the NFL.

        NaN HANDLING:
            If either the college stat or the NFL stat is NaN or 0,
            the ratio is NaN.  This is expected and correct — we don't
            invent data.  The gridsearch will test whether to impute or
            ignore these missing values.
        """
        logger.info("  Adding college-to-NFL ratio columns…")
        added = 0

        for college_col, nfl_col, ratio_col in COLLEGE_TO_NFL_RATIOS:
            if college_col not in df.columns or nfl_col not in df.columns:
                logger.warning(
                    "    Skipping ratio '%s' — missing column(s) '%s' or '%s'.",
                    ratio_col, college_col, nfl_col,
                )
                continue

            # Replace 0 with NaN in the denominator to avoid division by zero.
            # A player with 0 NFL yards shouldn't produce Inf or a crash.
            safe_nfl = df[nfl_col].replace(0, pd.NA)
            df[ratio_col] = (df[college_col] / safe_nfl).fillna(float("nan")).round(4)
            added += 1

        logger.info("    Added %d college-to-NFL ratio columns.", added)
        return df

    def _add_position_average_comparisons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns showing how each player compares to the average player
        at their position in the same season.

        FORMULA:
            stat_vs_pos_avg = player_stat - mean(stat for all players at
                                                  same position in same season)

        EXAMPLE:
            In 2023, the average QB had 3200 pass_yards.
            Patrick Mahomes had 4183 pass_yards.
            pass_yards_vs_pos_avg = 4183 - 3200 = +983

            A backup QB had 800 pass_yards.
            pass_yards_vs_pos_avg = 800 - 3200 = -2400

        WHY THIS IS USEFUL:
            Raw stats are position-dependent — comparing a QB's yards to a
            RB's yards is meaningless.  This normalisation puts everyone on
            a level playing field within their position group.

        HOW groupby + transform WORKS:
            df.groupby(["position", "season"])["pass_yards"].transform("mean")

            groupby()   → splits the DataFrame into groups (e.g. QB 2023, RB 2023)
            transform() → computes the mean for each group, then broadcasts it
                          back so every row gets its group's mean value.
            This is faster and cleaner than a manual for loop over positions.
        """
        logger.info("  Adding position-average comparison columns…")
        added = 0

        for stat_col in POSITION_AVG_STATS:
            if stat_col not in df.columns:
                logger.warning("    Skipping '%s' — column not found.", stat_col)
                continue

            vs_col = f"{stat_col}_vs_pos_avg"

            # Calculate the mean for each (position, season) group and
            # assign it back to every row in that group.
            position_season_mean = (
                df.groupby(["position", "season"])[stat_col]
                .transform("mean")
            )

            # Subtract the group mean from each player's individual value.
            df[vs_col] = (df[stat_col] - position_season_mean).round(2)
            added += 1

        logger.info("    Added %d position-average comparison columns.", added)
        return df