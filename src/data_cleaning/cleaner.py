"""
data_cleaning/cleaner.py
========================

PURPOSE:
    Fixes known data quality issues in the master DataFrame so that
    feature_engineer.py and the model training step receive trustworthy data.

    This file does NOT create new analytical columns — that's feature_engineer.py.
    The rule is:
        cleaner.py      →  fix or remove bad/missing data
        feature_engineer.py →  create new columns from clean data

WHAT THIS FILE FIXES:
    1. College season alignment
           Uses draft_year - 1 as each player's last college season.
           For undrafted free agents (UDFAs), uses first_nfl_season - 1.
           Rows where draft_year is still NaN after UDFA logic get flagged
           with a constant (DRAFT_YEAR_UNKNOWN) for manual review.

    2. Player name normalisation
           Strips extra whitespace and converts to Title Case so that
           "patrick mahomes", "Patrick  Mahomes", and "PATRICK MAHOMES"
           all match correctly when joining tables.

    3. Removes the standard scoring column
           We are targeting half-PPR and full-PPR only.
           Dropping fantasy_pts_standard keeps the DataFrame tidy.

    4. Drops rows with missing position
           A player with no position value can't be assigned to any
           position group — they're useless to the model.

    5. Flags rows with alignment fallbacks
           Any row where we couldn't determine last_college_season
           reliably gets a boolean flag (college_season_needs_review = True)
           so you can manually inspect and fix those players.

HOW TO USE:
    from src.data_cleaning import DataCleaner

    cleaner  = DataCleaner()
    clean_df = cleaner.clean(master_df)
"""

import logging
import pandas as pd

from data_intake.links import CURRENT_NFL_SEASON

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Sentinel value written into draft_year when we truly cannot determine it.
# Using -1 (an impossible real year) makes these rows easy to filter/spot.
# When you find these rows manually, you can update the value to the real year.
DRAFT_YEAR_UNKNOWN = -1

# Column name for the flag that marks rows needing manual review.
REVIEW_FLAG_COL = "college_season_needs_review"


# ---------------------------------------------------------------------------
# CLEANER CLASS
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Applies all cleaning steps to the master DataFrame in a fixed order.

    EXAMPLE:
        cleaner  = DataCleaner()
        clean_df = cleaner.clean(master_df)

        # See which rows need manual review for college season alignment:
        needs_review = clean_df[clean_df["college_season_needs_review"] == True]
        print(needs_review[["full_name", "draft_year", "last_college_season"]])
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all cleaning steps in order and returns a cleaned copy.

        We always work on a COPY of the input DataFrame (df.copy()) so
        the original is never modified.  This is a best practice in pandas —
        it prevents hard-to-track bugs where a variable you thought was
        unchanged has actually been modified in place.

        PARAMETERS:
            df (pd.DataFrame) : The master DataFrame from DataCombiner,
                                already validated by DataValidator.

        RETURNS:
            pd.DataFrame : A cleaned copy ready for feature engineering.
        """
        logger.info("Starting data cleaning pipeline…")

        # Always work on a copy — never mutate the original
        df = df.copy()

        df = self._normalise_player_names(df)
        df = self._drop_missing_positions(df)
        df = self._resolve_college_season(df)
        df = self._drop_standard_scoring(df)

        logger.info(
            "Cleaning complete → %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df.reset_index(drop=True)

    # ── Private cleaning steps ───────────────────────────────────────────────

    def _normalise_player_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardises player name formatting across all name columns.

        PROBLEMS THIS FIXES:
            - Leading/trailing spaces:  "  Patrick Mahomes " → "Patrick Mahomes"
            - Extra internal spaces:    "Patrick  Mahomes"  → "Patrick Mahomes"
            - Inconsistent case:        "patrick mahomes"   → "Patrick Mahomes"

        WHY IT MATTERS:
            Player names are used as JOIN keys (matching NFL stats to college
            stats, combine data, etc.).  A single space or capitalisation
            difference means two rows for the same player won't match.

        WHAT IS str.strip() / str.title()?
            These are pandas string methods — they apply a text operation
            to every value in a column at once, without needing a for loop.
                .str.strip()    removes leading/trailing whitespace
                .str.replace()  replaces patterns inside the string
                .str.title()    capitalises the first letter of each word
        """
        name_columns = [col for col in df.columns if "name" in col.lower()]

        for col in name_columns:
            if df[col].dtype == object:   # "object" dtype means text in pandas
                df[col] = (
                    df[col]
                    .str.strip()                        # remove edge whitespace
                    .str.replace(r"\s+", " ", regex=True)  # collapse internal spaces
                    .str.title()                        # Title Case every word
                )
                logger.info("  Normalised name column: '%s'.", col)

        return df

    def _drop_missing_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows where the position column is NaN.

        WHY:
            Every downstream step — per-game averages, position averages,
            model training — groups players by position.  A row with no
            position can't be assigned to any group and would cause errors.

        WHAT IS notna()?
            pd.Series.notna() returns True where the value is NOT NaN.
            We keep only those rows.
        """
        before = len(df)
        df = df[df["position"].notna()].copy()
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(
                "  Dropped %d rows with missing position values.", dropped
            )
        else:
            logger.info("  ✓ No rows with missing position values.")

        return df

    def _resolve_college_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determines each player's last college season and stores it in a
        new column: last_college_season.

        Also adds a boolean flag column (college_season_needs_review)
        marking rows where we could not determine the college season
        reliably — so you can manually inspect and fill them in later.

        ─────────────────────────────────────────────────────────────────
        LOGIC (applied in this order):

        CASE 1 — Normal drafted player:
            draft_year is known (not NaN and not DRAFT_YEAR_UNKNOWN)
            last_college_season = draft_year - 1
            college_season_needs_review = False

            EXAMPLE:
                Patrick Mahomes, draft_year = 2017
                last_college_season = 2016  ✓

        CASE 2 — Undrafted free agent (UDFA):
            draft_year is NaN BUT years_exp is known.
            We calculate their first NFL season:
                first_nfl_season = CURRENT_NFL_SEASON - years_exp
            Then:
                last_college_season = first_nfl_season - 1
            college_season_needs_review = False

            WHY THIS WORKS:
                UDFAs sign with teams right after the draft in the same
                year as draftees.  So their first NFL season is the same
                as their "draft year" would have been.

            EXAMPLE:
                A UDFA with years_exp = 3 in 2024:
                first_nfl_season = 2024 - 3 = 2021
                last_college_season = 2020  ✓

        CASE 3 — Neither draft_year NOR years_exp available:
            Both are NaN — we truly don't know.
            draft_year is set to DRAFT_YEAR_UNKNOWN (-1) as a sentinel.
            last_college_season is set to NaN.
            college_season_needs_review = True  ← manual review needed

        ─────────────────────────────────────────────────────────────────

        WHAT IS pd.notna()?
            Returns True if the value is NOT NaN.  We use it to check
            whether draft_year or years_exp has a real value.
        """
        logger.info("  Resolving college season alignment…")

        # Initialise the new columns with NaN / False defaults
        df["last_college_season"]     = pd.NA
        df[REVIEW_FLAG_COL]           = False

        # ── CASE 1 — Known draft year ──────────────────────────────────────
        has_draft_year = pd.notna(df["draft_year"]) & (df["draft_year"] != DRAFT_YEAR_UNKNOWN)
        df.loc[has_draft_year, "last_college_season"] = df.loc[has_draft_year, "draft_year"] - 1
        df.loc[has_draft_year, REVIEW_FLAG_COL]       = False

        case1_count = has_draft_year.sum()
        logger.info("    Case 1 (drafted): %d players resolved via draft_year.", case1_count)

        # ── CASE 2 — UDFA: no draft_year but years_exp is known ───────────
        # We only run this for rows that Case 1 didn't resolve.
        is_udfa = ~has_draft_year & pd.notna(df["years_exp"])
        first_nfl_season = CURRENT_NFL_SEASON - df.loc[is_udfa, "years_exp"]
        df.loc[is_udfa, "last_college_season"] = first_nfl_season - 1
        df.loc[is_udfa, REVIEW_FLAG_COL]       = False

        # Mark their draft_year as the sentinel so we know they're UDFAs,
        # while still distinguishing them from the "totally unknown" Case 3.
        # We use 0 as a UDFA marker (0 is not a valid draft year).
        df.loc[is_udfa, "draft_year"] = 0

        case2_count = is_udfa.sum()
        logger.info("    Case 2 (UDFA): %d players resolved via years_exp.", case2_count)

        # ── CASE 3 — Unknown: flag for manual review ───────────────────────
        is_unknown = ~has_draft_year & ~is_udfa
        df.loc[is_unknown, "draft_year"]            = DRAFT_YEAR_UNKNOWN
        df.loc[is_unknown, "last_college_season"]   = pd.NA
        df.loc[is_unknown, REVIEW_FLAG_COL]         = True

        case3_count = is_unknown.sum()
        if case3_count > 0:
            unknown_players = df.loc[is_unknown, ["full_name", "season", "years_exp"]].drop_duplicates()
            logger.warning(
                "    Case 3 (unknown): %d rows flagged for manual review.\n"
                "    These players could not be matched to a college season:\n%s",
                case3_count,
                unknown_players.to_string(index=False),
            )
        else:
            logger.info("    Case 3 (unknown): 0 rows — all players resolved ✓")

        # Convert last_college_season to float64.
        # We use float64 rather than int because NaN values require a
        # floating-point type — plain int cannot represent NaN in pandas.
        df["last_college_season"] = df["last_college_season"].astype("float64")

        return df

    def _drop_standard_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the fantasy_pts_standard column.

        WHY:
            Our model targets half-PPR and full-PPR scoring only.
            Keeping standard scoring adds noise and an extra column the
            model doesn't need.  Cleaner input = easier model interpretation.

        WHAT IS errors="ignore"?
            If fantasy_pts_standard doesn't exist in the DataFrame
            (e.g. data_intake was run without it), drop() won't crash —
            it silently skips the missing column.
        """
        df = df.drop(columns=["fantasy_pts_standard"], errors="ignore")
        logger.info("  Dropped 'fantasy_pts_standard' column (not a target format).")
        return df