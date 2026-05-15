"""
data_intake/combine.py
======================

This file has TWO jobs — both relate to the word "combine":

    PART A — ESPNCombineData  (despite the name, now uses nfl_data_py)
        Fetches NFL Draft Combine *measurements*: 40-yard dash, bench press,
        vertical jump, etc.

        WHY WE SWITCHED FROM ESPN:
            The ESPN hidden API started returning 404 errors for combine data.
            nfl_data_py (which wraps the open nflverse project) is free,
            documented, and actively maintained.  It's the better long-term choice.

    PART B — DataCombiner
        *Combines* (merges/joins) all four DataFrames into one master table
        that the rest of the pipeline will use.

─────────────────────────────────────────────────────────────────────────────

WHAT IS THE NFL COMBINE? (Part A)
    Every February, top college prospects travel to Indianapolis and perform
    standardised athletic tests so NFL teams can compare everyone fairly.

    Key tests:
        40-Yard Dash        → straight-line speed          (lower time = faster)
        Vertical Jump       → explosiveness off the ground (higher = better)
        Broad Jump          → horizontal lower-body power  (higher = better)
        3-Cone Drill        → change-of-direction agility  (lower = better)
        20-Yard Shuttle     → lateral quickness            (lower = better)
        Bench Press (225lb) → upper-body strength          (more reps = stronger)

    ⚠  NOT EVERYONE ATTENDS.  Some players skip due to injury or agent advice.
       Missing measurements are stored as NaN.  This is expected — we keep
       those players using outer/left joins later.

WHAT IS A JOIN / MERGE? (Part B)
    Imagine two spreadsheets that share a column (like "player_name").
    A JOIN lines them up by that shared column so each row gets columns
    from BOTH sheets side-by-side.

    Types used in this file:
        LEFT  → keep every row in the LEFT sheet; fill NaN where no right match

HOW TO USE:
    from src.data_intake.combine import ESPNCombineData, DataCombiner

    combine_df = ESPNCombineData().get_combine_dataframe(years=[2023, 2024, 2025])
    master_df  = DataCombiner().merge_all(nfl_df, team_df, combine_df, college_df)
"""

import logging

import pandas as pd
import nfl_data_py as nfl

from .links import (
    CURRENT_NFL_SEASON,
    FANTASY_POSITIONS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PART A — ESPNCombineData
#           Now powered by nfl_data_py (nflverse) instead of ESPN.
#           The class name is kept the same so no other files need to change.
# =============================================================================

# ---------------------------------------------------------------------------
# COLUMN NAME MAPPING
# nfl_data_py uses its own column names — we rename them to match
# the rest of our project's naming convention.
# ---------------------------------------------------------------------------
# Format:  nfl_data_py column name  →  our column name
_COMBINE_COLUMN_MAP = {
    "player_name"  : "full_name",
    "pos"          : "position",
    "school"       : "college",
    "season"       : "draft_year",       # in combine data, "season" = draft year
    "ht"           : "height_inches",
    "wt"           : "weight_lbs",
    "forty"        : "forty_yard_dash",
    "vertical"     : "vertical_jump",
    "bench_reps"   : "bench_press_reps",
    "broad_jump"   : "broad_jump",
    "cone"         : "three_cone_drill",
    "shuttle"      : "twenty_yard_shuttle",
}

# These are the columns we keep in our final output.
# Any column not in this list gets dropped.
_KEEP_COLS = [
    "draft_year",
    "full_name",
    "position",
    "college",
    "height_inches",
    "weight_lbs",
    "forty_yard_dash",
    "vertical_jump",
    "broad_jump",
    "three_cone_drill",
    "twenty_yard_shuttle",
    "bench_press_reps",
]


class ESPNCombineData:
    """
    Downloads NFL Draft Combine measurements using nfl_data_py (nflverse).

    Despite the name (kept for backward compatibility), this class no longer
    uses the ESPN API — it uses the nfl_data_py package instead, which
    downloads cleaned combine data from the nflverse open data project.

    One row in the resulting DataFrame = one prospect's measurements
    for one draft year.

    ⚠  NOT EVERYONE ATTENDS THE COMBINE:
        Some players skip due to injury, agent advice, or not being invited.
        Missing measurements are stored as NaN — not dropped.
        We never remove a player just because they lack combine data.
    """

    def __init__(self):
        logger.info("ESPNCombineData initialized (using nfl_data_py / nflverse).")

    def get_combine_dataframe(self, years: list[int] | None = None) -> pd.DataFrame:
        """
        Fetches combine measurements for the given draft years using nfl_data_py.

        PARAMETERS:
            years (list[int] | None) :
                Draft years to fetch (e.g. [2023, 2024, 2025]).
                Defaults to [CURRENT_NFL_SEASON] if not supplied.

        RETURNS:
            pd.DataFrame with one row per prospect who attended the combine.
            Measurement columns may contain NaN — that is expected and normal.
        """
        if years is None:
            years = [CURRENT_NFL_SEASON]

        logger.info("Fetching combine data for draft years: %s", years)

        # ── Fetch from nfl_data_py ────────────────────────────────────────
        # nfl_data_py downloads parquet files from GitHub (nflverse project).
        # This is a single function call that handles all the downloading.
        try:
            df = nfl.import_combine_data(years=years)
        except Exception as e:
            logger.error("Failed to fetch combine data via nfl_data_py: %s", e)
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("nfl_data_py returned no combine data for years: %s", years)
            return pd.DataFrame()

        logger.info("Raw combine data: %d rows × %d columns.", len(df), len(df.columns))

        # ── Rename columns to our naming convention ───────────────────────
        # Only rename columns that actually exist — avoids errors if
        # nfl_data_py changes a column name in a future version.
        rename_map = {
            old: new
            for old, new in _COMBINE_COLUMN_MAP.items()
            if old in df.columns
        }
        df = df.rename(columns=rename_map)

        # ── Filter to fantasy-relevant positions only ──────────────────────
        # We only care about QB, RB, WR, TE for our prediction model.
        if "position" in df.columns:
            before = len(df)
            df = df[df["position"].isin(FANTASY_POSITIONS)].copy()
            logger.info(
                "Position filter: kept %d of %d rows (QB/RB/WR/TE only).",
                len(df), before,
            )
        else:
            logger.warning("'position' column not found — skipping position filter.")

        # ── Keep only the columns we need ─────────────────────────────────
        cols_to_keep = [c for c in _KEEP_COLS if c in df.columns]
        df = df[cols_to_keep].reset_index(drop=True)

        logger.info(
            "Combine DataFrame complete: %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df


# =============================================================================
# PART B — DataCombiner
#           Merges all four separate DataFrames into one master table.
#           This class is UNCHANGED — it doesn't care where combine data
#           came from (ESPN or nfl_data_py), only what columns it has.
# =============================================================================

class DataCombiner:
    """
    Merges the four separate DataFrames into a single master table.

    WHY MERGE SEPARATELY INSTEAD OF ALL AT ONCE?
        Each dataset has different "natural keys" (the columns used to match
        rows between tables):

            nfl_df     ──→  team_df     via  nfl_team + season
            │
            └──────────→  combine_df  via  full_name + college + draft_year
            │
            └──────────→  college_df  via  full_name + college + last_college_season

    Doing it in steps makes errors easier to find — if something goes wrong
    you can see exactly which merge caused the problem.

    DIAGRAM:
        nfl_df  ←──────── the "spine" (every player-season row is kept)
    └────────┬────────┘                    └──────────────┘
             │  nfl_team + season
             ├──────────────────────────→  team_df     (team context)
             │  full_name + college + draft_year
             ├──────────────────────────→  combine_df  (combine measurements)
             │  full_name + college + (draft_year - 1)
             └──────────────────────────→  college_df  (college stats)

    All joins are LEFT joins: every NFL player row is kept even if there
    is no matching row in combine_df or college_df.  Missing data = NaN.

    COLLEGE ALIGNMENT IMPROVEMENT:
        Previously we used:  last_college_season = nfl_season - 1
        Now we use:          last_college_season = draft_year - 1

        draft_year comes directly from player_data.py (via Sleeper's
        years_exp field).  This is more accurate because it correctly
        handles players who redshirted, took gap years, or came back
        for a 5th college season.

        If draft_year is NaN for a player, we fall back to season - 1
        as a safety net.  data_cleaning will flag and review those cases.
    """

    def merge_all(
        self,
        nfl_df     : pd.DataFrame,
        team_df    : pd.DataFrame,
        combine_df : pd.DataFrame,
        college_df : pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Runs the complete merge pipeline.

        PARAMETERS:
            nfl_df     : from SleeperPlayerData().get_player_dataframe()
            team_df    : from SleeperTeamData().get_team_dataframe()
            combine_df : from ESPNCombineData().get_combine_dataframe()
            college_df : from ESPNCollegeData().get_college_dataframe()

        RETURNS:
            pd.DataFrame — master table, one row per player-season,
            with columns from all four sources.
        """
        logger.info("Starting master merge pipeline…")

        for name, df in [("nfl", nfl_df), ("team", team_df),
                         ("combine", combine_df), ("college", college_df)]:
            if df.empty:
                logger.warning("'%s' DataFrame is empty — merged output will be sparse.", name)

        # Step 1 — Player stats + team context (same team, same season)
        master = self._merge_nfl_team(nfl_df, team_df)
        logger.info("Step 1 done: %d rows × %d cols", len(master), len(master.columns))

        # Step 2 — Add combine measurements (joined on name + college + draft_year)
        master = self._merge_combine(master, combine_df)
        logger.info("Step 2 done: %d rows × %d cols", len(master), len(master.columns))

        # Step 3 — Add college stats (season before the player was drafted)
        master = self._merge_college(master, college_df)
        logger.info("Step 3 done: %d rows × %d cols", len(master), len(master.columns))

        logger.info(
            "Master merge complete → %d rows × %d columns.",
            len(master), len(master.columns),
        )
        return master.reset_index(drop=True)

    # ── Private merge steps ──────────────────────────────────────────────────

    def _merge_nfl_team(
        self, nfl_df: pd.DataFrame, team_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        LEFT joins player stats with team stats on (nfl_team, season).

        WHY LEFT?
            Every player row is kept.  Players between teams or on practice
            squads may not match a team row — they get NaN for team columns.
        """
        if team_df.empty:
            logger.warning("team_df empty — skipping team merge.")
            return nfl_df.copy()

        # Add a "team_" prefix to team columns to avoid naming collisions
        team_prefixed = team_df.rename(columns={
            col: f"team_{col}"
            for col in team_df.columns
            if col not in ("nfl_team", "season")
        })

        merged = pd.merge(
            nfl_df,
            team_prefixed,
            on  = ["nfl_team", "season"],
            how = "left",
        )
        return merged

    def _merge_combine(
        self, master: pd.DataFrame, combine_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        LEFT joins combine measurements into the master table.

        JOIN KEY: full_name + college + draft_year

        WHY draft_year?
            The combine happens before the player's first NFL season.
            Matching on draft_year ensures we link each player to the
            combine they actually attended — not a random year's data.

        COLUMN PREFIX: combine_*
            All combine columns get a "combine_" prefix to make it obvious
            where each column came from when looking at the master table.
        """
        if combine_df.empty:
            logger.warning("combine_df empty — skipping combine merge.")
            return master.copy()

        # Add "combine_" prefix to measurement columns
        combine_prefixed = combine_df.rename(columns={
            col: f"combine_{col}"
            for col in combine_df.columns
            if col not in ("full_name", "college", "position", "draft_year")
        })

        merged = pd.merge(
            master,
            combine_prefixed,
            on  = ["full_name", "college", "draft_year"],
            how = "left",
            suffixes = ("", "_comb"),
        )
        return merged

    def _merge_college(
        self, master: pd.DataFrame, college_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        LEFT joins college stats into the master table.

        JOIN KEY: full_name + college + last_college_season

        COLLEGE SEASON ALIGNMENT (IMPROVED):
            A player drafted in 2022 played their last college season in 2021.
            We calculate: last_college_season = draft_year - 1

            This is more accurate than the old season - 1 approach because
            draft_year is the actual year they entered the league, whereas
            season - 1 assumed they always played exactly one season before
            being compared — which isn't true for multi-year NFL veterans.

            FALLBACK:
                If draft_year is NaN (missing from Sleeper), we fall back to
                season - 1.  data_cleaning will flag these rows for review.

        COLUMN PREFIX: college_*
            All college columns get a "college_" prefix.
        """
        if college_df.empty:
            logger.warning("college_df empty — skipping college merge.")
            return master.copy()

        # Add "college_" prefix to stat columns
        college_prefixed = college_df.rename(columns={
            col: f"college_{col}"
            for col in college_df.columns
            if col not in ("full_name", "college", "position")
        }).rename(columns={"season": "college_season"})

        master = master.copy()

        # Build the lookup key for last college season.
        # If draft_year exists, use draft_year - 1.
        # If draft_year is NaN, fall back to season - 1 (less accurate).
        #
        # pd.notna() returns True if the value is NOT NaN.
        # np.where(condition, value_if_true, value_if_false) is like an
        # if/else applied to an entire column at once.
        import numpy as np
        master["_college_season_key"] = np.where(
            pd.notna(master["draft_year"]),
            master["draft_year"] - 1,    # preferred: accurate
            master["season"] - 1,        # fallback: approximate
        ).astype("Int64")                # Int64 supports NaN; plain int does not

        merged = pd.merge(
            master,
            college_prefixed,
            left_on  = ["full_name", "college", "_college_season_key"],
            right_on = ["full_name", "college", "college_season"],
            how      = "left",
            suffixes = ("", "_coll"),
        )

        # Drop the temporary helper column — its job is done
        merged = merged.drop(columns=["_college_season_key"], errors="ignore")
        return merged