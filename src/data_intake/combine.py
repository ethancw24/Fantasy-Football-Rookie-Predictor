"""
data_intake/combine.py
======================

This file has TWO jobs — both relate to the word "combine":

    PART A — ESPNCombineData
        Fetches NFL Draft Combine *measurements* from ESPN
        (40-yard dash, bench press, vertical jump, etc.)

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
       Missing combine data shows up as NaN.  This is expected — we keep
       those players using an outer/left join.

WHAT IS A JOIN / MERGE? (Part B)
    Imagine two spreadsheets that share a column (like "player_name").
    A JOIN lines them up by that shared column so each row gets columns
    from BOTH sheets side-by-side.

    Types used in this file:
        LEFT  → keep every row in the LEFT sheet; fill NaN where no right match

HOW TO USE:
    from src.data_intake import (
        SleeperPlayerData, SleeperTeamData, ESPNCombineData, ESPNCollegeData
    )
    from src.data_intake.combine import DataCombiner

    nfl_df     = SleeperPlayerData().get_player_dataframe()
    team_df    = SleeperTeamData().get_team_dataframe()
    combine_df = ESPNCombineData().get_combine_dataframe(years=[2021, 2022, 2023])
    college_df = ESPNCollegeData().get_college_dataframe(years=[2020, 2021, 2022])

    master_df  = DataCombiner().merge_all(nfl_df, team_df, combine_df, college_df)
"""

import logging
import time

import requests
import pandas as pd

from .links import (
    ESPN_COMBINE_URL,
    CURRENT_NFL_SEASON,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PART A — ESPNCombineData
#           Fetches NFL Draft Combine athletic measurements from ESPN.
# =============================================================================

class ESPNCombineData:
    """
    Downloads NFL Draft Combine measurements from the (unofficial) ESPN API.

    One row in the resulting DataFrame = one prospect's measurements
    for one draft year.

    ⚠  NOT EVERYONE ATTENDS THE COMBINE:
        Some players skip due to injury, agent advice, or not being invited.
        Missing measurements are stored as NaN — not dropped.
        We never remove a player just because they lack combine data.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FantasyFootballRookiePredictor/1.0"})
        logger.info("ESPNCombineData initialized.")

    def _get(self, url: str) -> dict | list | None:
        """Makes a GET request and returns parsed JSON, or None on error."""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed: %s — %s", url, e)
            return None

    def _safe_float(self, value) -> float | None:
        """
        Converts a value to float safely.
        Returns None (→ NaN in DataFrame) instead of crashing on bad data.

        WHY None INSTEAD OF 0?
            0 would imply the player ran the 40-yard dash in 0 seconds,
            which is impossible and would corrupt our model.
            NaN (from None) correctly signals "we don't have this data."
        """
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _fetch_year(self, year: int) -> list[dict]:
        """Fetches raw combine prospect data for one draft year."""
        url = ESPN_COMBINE_URL.format(year=year)
        logger.info("  Fetching combine data for draft year %d…", year)
        data = self._get(url)

        if data is None:
            logger.warning("  No combine data returned for %d.", year)
            return []

        # ESPN wraps prospect data in a nested structure
        return data.get("athletes", data) if isinstance(data, dict) else data

    def _get_position(self, prospect: dict) -> str | None:
        """Extracts position abbreviation from a prospect dict."""
        return prospect.get("athlete", {}).get("position", {}).get("abbreviation")

    def _build_row(self, year: int, prospect: dict) -> dict:
        """
        Flattens one prospect's ESPN JSON into a plain dictionary (one row).

        ESPN structure (simplified):
            {
                "athlete": {
                    "id": "...",
                    "displayName": "Bijan Robinson",
                    "position": {"abbreviation": "RB"},
                    "college": {"shortDisplayName": "Texas"}
                },
                "results": [
                    {"name": "40YardDash",  "displayValue": "4.46"},
                    {"name": "verticalJump","displayValue": "37.5"},
                    ...
                ]
            }
        """
        athlete = prospect.get("athlete", {})

        # Convert the results list to a dict for easy lookup
        results = {
            r["name"]: r.get("displayValue")
            for r in prospect.get("results", [])
            if "name" in r
        }

        return {
            # Identity
            "draft_year"          : year,
            "espn_athlete_id"     : athlete.get("id"),
            "full_name"           : athlete.get("displayName", "Unknown"),
            "position"            : athlete.get("position", {}).get("abbreviation"),
            "college"             : athlete.get("college", {}).get("shortDisplayName"),

            # Size
            "height_inches"       : self._safe_float(results.get("height")),
            "weight_lbs"          : self._safe_float(results.get("weight")),

            # Athletic measurements — None → NaN if player didn't perform test
            "forty_yard_dash"     : self._safe_float(results.get("40YardDash")),
            "vertical_jump"       : self._safe_float(results.get("verticalJump")),
            "broad_jump"          : self._safe_float(results.get("broadJump")),
            "three_cone_drill"    : self._safe_float(results.get("3ConeDrill")),
            "twenty_yard_shuttle" : self._safe_float(results.get("20YardShuttle")),
            "bench_press_reps"    : self._safe_float(results.get("benchPress")),
        }

    def get_combine_dataframe(self, years: list[int] | None = None) -> pd.DataFrame:
        """
        Fetches combine measurements for the given draft years.

        PARAMETERS:
            years : List of draft years to fetch (e.g. [2021, 2022, 2023]).
                    Defaults to [CURRENT_NFL_SEASON] if not supplied.

        RETURNS:
            pd.DataFrame with one row per prospect.
            Measurement columns may be NaN — that is expected and normal.
        """
        if years is None:
            years = [CURRENT_NFL_SEASON]

        rows = [
            self._build_row(year, p)
            for year in years
            for p in self._fetch_year(year)
        ]

        if not rows:
            logger.warning("No combine data — returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Drop rows where EVERY measurement column is NaN (placeholder entries).
        # Rows with only PARTIAL data are kept — partial data is still useful.
        measurement_cols = [
            "forty_yard_dash", "vertical_jump", "broad_jump",
            "three_cone_drill", "twenty_yard_shuttle", "bench_press_reps",
        ]
        before = len(df)
        df = df.dropna(subset=measurement_cols, how="all").reset_index(drop=True)
        logger.info(
            "Removed %d fully empty rows. Combine DataFrame: %d rows × %d cols.",
            before - len(df), len(df), len(df.columns),
        )
        return df


# =============================================================================
# PART B — DataCombiner
#           Merges (combines) all four DataFrames into one master table.
# =============================================================================

class DataCombiner:
    """
    Joins all four data sources into one master DataFrame.

    ┌─────────────────┐   season + team   ┌──────────────┐
    │  nfl_df         │ ←────────────────→ │  team_df     │
    │  (player stats) │                    │  (team stats)│
    └────────┬────────┘                    └──────────────┘
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