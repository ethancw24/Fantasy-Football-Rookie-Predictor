"""
data_intake/college_data.py
===========================

PURPOSE:
    Fetches college football statistics from the College Football Data API (CFBD).

WHY WE SWITCHED FROM ESPN:
    The ESPN hidden API for college football started returning empty results.
    CFBD (collegefootballdata.com) is a free, documented, community-maintained
    API specifically built for college football statistics.  It's far more
    reliable for this use case.

API KEY REQUIRED:
    CFBD requires a free API key.  Sign up at https://collegefootballdata.com
    Then set your key as an environment variable before running the project:

        Windows PowerShell:
            $env:CFBD_API_KEY = "your_key_here"

        Mac / Linux terminal:
            export CFBD_API_KEY="your_key_here"

    The code will raise a clear error if the key is missing.

THE BIG IDEA:
    If we can find patterns like:
        "College WRs with >1000 receiving yards AND >10 TDs in their final
         season tend to become WR1s in the NFL within 3 years"
    ...then we can apply those patterns to current college players to predict
    who will be the next fantasy stars.

HOW CFBD RETURNS STATS:
    Unlike a spreadsheet where each player has one row with all their stats,
    CFBD returns stats in "long format":
        One row = one player + one stat type + one value

    Example:
        player="Caleb Williams", team="USC", category="passing", statType="YDS", stat=3633
        player="Caleb Williams", team="USC", category="passing", statType="TD",  stat=30
        player="Caleb Williams", team="USC", category="passing", statType="INT", stat=5

    We fetch stats by category (passing, rushing, receiving) and then
    "pivot" each category into wide format (one row per player, one column
    per stat type), then merge all three together.

POSITION ASSIGNMENT:
    CFBD's stats endpoint doesn't include position.  We infer it:
        → Most passing yards = QB
        → Most rushing yards (and not QB) = RB
        → Most receiving yards (and not QB or RB) = WR
    Note: TEs are labeled as WR for now since we can't distinguish them
    from WRs without roster data.  This can be improved in a future update.

TOP-N FILTERING — WHY WE LIMIT THE DATASET:
    The CFBD API covers hundreds of players per position.  Most are backups
    or depth players who will never be drafted.  We only keep:

        Top 300 WRs  — ranked by total receiving yards per season
        Top 200 RBs  — ranked by total rushing yards per season
        Top 150 QBs  — ranked by total passing yards per season

CONSISTENCY FLAGS:
    We add two columns per player so the ML model can see multi-year trends:

        productive_seasons  : how many seasons had any non-zero stats
        total_seasons       : how many seasons we have data for at all

HOW TO USE:
    fetcher = ESPNCollegeData()
    df = fetcher.get_college_dataframe(years=[2022, 2023, 2024])
    print(df.head())
"""

import os
import logging

import requests
import pandas as pd

from .links import (
    CFBD_BASE_URL,
    CFBD_REQUEST_TIMEOUT,
    FANTASY_POSITIONS,
    CURRENT_NFL_SEASON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TOP-N LIMITS PER POSITION
# ---------------------------------------------------------------------------
TOP_N_PER_POSITION = {
    "WR": 300,
    "RB": 200,
    "QB": 150,
}

# ---------------------------------------------------------------------------
# RANKING STAT PER POSITION
# Which column we sort by to determine the top-N players.
# ---------------------------------------------------------------------------
RANKING_STAT = {
    "WR": "rec_yards",
    "RB": "rush_yards",
    "QB": "pass_yards",
}

# ---------------------------------------------------------------------------
# STAT TYPE NAME MAPPINGS
# CFBD uses specific names for each statType within a category.
# We map them to our own column names.
# ---------------------------------------------------------------------------
PASSING_STAT_MAP = {
    "YDS"         : "pass_yards",
    "TD"          : "pass_touchdowns",
    "INT"         : "interceptions",
    "ATT"         : "pass_attempts",
    "COMPLETIONS" : "pass_completions",
    "PCT"         : "completion_pct",
    "AVG"         : "pass_yards_per_attempt",
}

RUSHING_STAT_MAP = {
    "YDS" : "rush_yards",
    "TD"  : "rush_touchdowns",
    "CAR" : "rush_attempts",
    "AVG" : "yards_per_carry",
}

RECEIVING_STAT_MAP = {
    "YDS" : "rec_yards",
    "TD"  : "rec_touchdowns",
    "REC" : "receptions",
    "AVG" : "yards_per_rec",
}


class ESPNCollegeData:
    """
    Fetches college football player statistics from the CFBD API.

    Despite the name (kept for backward compatibility with the rest of
    the project), this class uses the College Football Data API — not ESPN.

    Pipeline:
        1. For each season year, fetch passing, rushing, and receiving stats.
        2. Pivot each from "long format" to "wide format" (one row per player).
        3. Merge all three stat groups together per player.
        4. Infer each player's position from their dominant stats.
        5. Keep only the top-N players per position (by total yards).
        6. Add consistency flags (productive_seasons, total_seasons).
    """

    def __init__(self):
        # Load the CFBD API key from the environment variable.
        # The program will stop here with a clear error if it's not set.
        self.api_key = os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "\n\n"
                "CFBD_API_KEY environment variable is not set.\n"
                "Sign up for a free key at: https://collegefootballdata.com\n"
                "Then run this in your terminal before starting the project:\n\n"
                "  Windows:  $env:CFBD_API_KEY = 'your_key_here'\n"
                "  Mac/Linux: export CFBD_API_KEY='your_key_here'\n"
            )

        # Set up a shared HTTP session with auth headers pre-loaded.
        # Every CFBD request needs 'Authorization: Bearer <key>' in the header.
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization" : f"Bearer {self.api_key}",
            "User-Agent"    : "FantasyFootballRookiePredictor/1.0",
        })
        logger.info("ESPNCollegeData initialized (using CFBD API).")

    # -----------------------------------------------------------------------
    # PRIVATE: HTTP HELPER
    # -----------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> list | None:
        """
        Makes a GET request to the CFBD API and returns parsed JSON.

        PARAMETERS:
            endpoint (str)         : The API path (e.g. '/stats/player/season').
            params   (dict | None) : Query parameters (e.g. {'year': 2023}).

        RETURNS:
            list : The JSON response (CFBD always returns a list), or None on error.
        """
        url = CFBD_BASE_URL + endpoint
        try:
            response = self.session.get(url, params=params, timeout=CFBD_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching: %s", url)
        except requests.exceptions.HTTPError as err:
            logger.error("HTTP error: %s — %s", url, err)
        except requests.exceptions.RequestException as err:
            logger.error("Network error: %s — %s", url, err)
        return None

    # -----------------------------------------------------------------------
    # PRIVATE: FETCH ONE STAT CATEGORY
    # -----------------------------------------------------------------------

    def _fetch_category(self, year: int, category: str) -> list[dict]:
        """
        Fetches all player stats for one category (passing/rushing/receiving)
        for one season from CFBD.

        WHAT IS "LONG FORMAT"?
            CFBD doesn't return one row per player with all their stats.
            Instead it returns many rows per player — one row for each
            individual stat type.

            Example for Caleb Williams:
                {"player": "Caleb Williams", "statType": "YDS",  "stat": 3633}
                {"player": "Caleb Williams", "statType": "TD",   "stat": 30}
                {"player": "Caleb Williams", "statType": "INT",  "stat": 5}

            We use _pivot_category() to convert this into wide format
            (one row per player with separate columns for YDS, TD, INT, etc.)

        PARAMETERS:
            year     (int) : The college season year (e.g. 2023).
            category (str) : 'passing', 'rushing', or 'receiving'.

        RETURNS:
            list of raw record dicts from CFBD, or [] on failure.
        """
        logger.info("  Fetching %s stats for %d…", category, year)

        records = self._get(
            "/stats/player/season",
            params={
                "year"       : year,
                "category"   : category,
                "seasonType" : "regular",
            },
        )

        if records is None:
            logger.warning("  No %s data returned for %d.", category, year)
            return []

        logger.info("  Got %d raw %s stat records for %d.", len(records), category, year)
        return records

    # -----------------------------------------------------------------------
    # PRIVATE: PIVOT LONG → WIDE
    # -----------------------------------------------------------------------

    def _pivot_category(
        self,
        records  : list[dict],
        stat_map : dict[str, str],
    ) -> pd.DataFrame:
        """
        Converts CFBD's "long format" stat records into a wide DataFrame
        where each row is one player and each column is one stat type.

        WHAT IS PIVOTING?
            Long format:  one row per player per stat type
                player  | statType | stat
                --------|----------|------
                Player A | YDS     | 3000
                Player A | TD      | 25
                Player B | YDS     | 2500

            Wide format:  one row per player, one column per stat type
                player   | YDS  | TD
                ---------|------|----
                Player A | 3000 | 25
                Player B | 2500 | 0

        PARAMETERS:
            records  : Raw list of CFBD stat records.
            stat_map : Maps CFBD statType names to our column names.

        RETURNS:
            pd.DataFrame in wide format, or empty DataFrame on failure.
        """
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Check that the required columns exist in the response
        required = {"player", "team", "statType", "stat"}
        if not required.issubset(df.columns):
            logger.warning("Unexpected CFBD response format — missing columns.")
            return pd.DataFrame()

        # Keep only the stat types we care about (in our stat_map)
        df = df[df["statType"].isin(stat_map.keys())].copy()

        if df.empty:
            return pd.DataFrame()

        # PIVOT: one row per player + team, one column per statType.
        # aggfunc="first" handles the rare case where a player transferred
        # mid-season and has duplicate entries.
        pivot = df.pivot_table(
            index   = ["player", "team"],
            columns = "statType",
            values  = "stat",
            aggfunc = "first",
        ).reset_index()

        # Flatten the column names (pivot_table can create multi-level columns)
        pivot.columns.name = None

        # Rename statType columns to our naming convention
        pivot = pivot.rename(columns=stat_map)

        # Rename CFBD player/team columns to our naming convention
        pivot = pivot.rename(columns={
            "player" : "full_name",
            "team"   : "college",
        })

        return pivot

    # -----------------------------------------------------------------------
    # PRIVATE: ASSIGN POSITION
    # -----------------------------------------------------------------------

    def _assign_position(self, row: pd.Series) -> str | None:
        """
        Infers a player's position from their dominant stat category.

        LOGIC:
            1. If passing yards > rushing yards AND passing yards > receiving yards
               AND passing yards is substantial (>100) → QB
            2. If rushing yards is the dominant category AND substantial (>50) → RB
            3. If receiving yards is non-zero → WR  (includes TEs)
            4. Otherwise → None (player will be filtered out)

        NOTE ON TEs:
            We can't easily distinguish WRs from TEs without roster data.
            TEs are labeled as WR for now.  This is a known limitation.
            A future improvement would be to cross-reference with CFBD rosters.

        PARAMETERS:
            row (pd.Series) : One row of the merged stats DataFrame.

        RETURNS:
            str | None : 'QB', 'RB', 'WR', or None.
        """
        pass_yds = row.get("pass_yards", 0) or 0
        rush_yds = row.get("rush_yards", 0) or 0
        rec_yds  = row.get("rec_yards",  0) or 0

        if pass_yds > 100 and pass_yds >= rush_yds and pass_yds >= rec_yds:
            return "QB"
        elif rush_yds > 50 and rush_yds > pass_yds:
            return "RB"
        elif rec_yds > 0:
            return "WR"
        else:
            return None

    # -----------------------------------------------------------------------
    # PRIVATE: TOP-N FILTER
    # -----------------------------------------------------------------------

    def _apply_top_n_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps only the top-N players per position per season, ranked by total yards.

        WHY?
            We don't want to train our model on backup players who will never
            be drafted.  Keeping only the top producers keeps the data clean
            and focused on NFL-relevant talent.
        """
        filtered_groups = []

        for (season, position), group in df.groupby(["season", "position"]):
            limit       = TOP_N_PER_POSITION.get(position)
            ranking_col = RANKING_STAT.get(position)

            # Skip positions we don't track (e.g., TE is subsumed into WR)
            if limit is None or ranking_col is None:
                continue

            # Skip if the ranking column doesn't exist in the data
            if ranking_col not in group.columns:
                filtered_groups.append(group)
                continue

            top_n = (
                group
                .sort_values(ranking_col, ascending=False, na_position="last")
                .head(limit)
            )

            logger.info(
                "  %d %s: kept top %d of %d players (ranked by %s).",
                season, position, len(top_n), len(group), ranking_col,
            )
            filtered_groups.append(top_n)

        if not filtered_groups:
            return pd.DataFrame()

        return pd.concat(filtered_groups, ignore_index=True)

    # -----------------------------------------------------------------------
    # PRIVATE: CONSISTENCY FLAGS
    # -----------------------------------------------------------------------

    def _add_consistency_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds two columns describing each player's multi-year production pattern.

        COLUMNS ADDED:
            productive_seasons : how many seasons had any non-zero yard stats
            total_seasons      : total seasons we have any data row for at all

        WHY?
            A player who produces 1000 yards for 3 straight years is more
            reliable than one who had a single 1000-yard breakout.
            These flags let the ML model learn whether consistency matters.
        """
        yard_cols          = ["pass_yards", "rush_yards", "rec_yards"]
        existing_yard_cols = [c for c in yard_cols if c in df.columns]

        df = df.copy()
        df["_has_stats"] = df[existing_yard_cols].gt(0).any(axis=1)

        df["productive_seasons"] = (
            df.groupby("full_name")["_has_stats"]
            .transform("sum")
            .astype(int)
        )
        df["total_seasons"] = (
            df.groupby("full_name")["_has_stats"]
            .transform("count")
            .astype(int)
        )

        df = df.drop(columns=["_has_stats"])
        return df

    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------

    def get_college_dataframe(self, years: list[int] | None = None) -> pd.DataFrame:
        """
        Full pipeline: fetch stats → pivot → merge → assign positions →
        top-N filter → consistency flags → return.

        PARAMETERS:
            years (list[int] | None) :
                College football season years to collect (e.g. [2022, 2023, 2024]).
                Defaults to the last completed college season.

        RETURNS:
            pd.DataFrame : One row per player-season in the top-N cut,
                           with consistency flag columns appended.
        """
        if years is None:
            years = [CURRENT_NFL_SEASON - 1]

        all_rows = []

        for year in years:
            logger.info("=== Fetching college football data for %d ===", year)

            # ── Step 1: Fetch stats by category ───────────────────────────
            passing_records   = self._fetch_category(year, "passing")
            rushing_records   = self._fetch_category(year, "rushing")
            receiving_records = self._fetch_category(year, "receiving")

            # ── Step 2: Pivot each category to wide format ─────────────────
            passing_df   = self._pivot_category(passing_records,   PASSING_STAT_MAP)
            rushing_df   = self._pivot_category(rushing_records,   RUSHING_STAT_MAP)
            receiving_df = self._pivot_category(receiving_records, RECEIVING_STAT_MAP)

            # ── Step 3: Merge all three categories per player ──────────────
            # Start with passing, then outer-join rushing, then receiving.
            # outer join = keep ALL players from ALL categories, fill missing with NaN.
            merge_keys = ["full_name", "college"]

            if passing_df.empty and rushing_df.empty and receiving_df.empty:
                logger.warning("No data for any category in %d — skipping.", year)
                continue

            # Build the merged frame starting from whichever is non-empty
            frames = [f for f in [passing_df, rushing_df, receiving_df] if not f.empty]
            season_df = frames[0]
            for frame in frames[1:]:
                season_df = pd.merge(season_df, frame, on=merge_keys, how="outer")

            # Fill NaN stats with 0 (a player who doesn't rush has 0 rush yards)
            stat_cols = [
                c for c in season_df.columns
                if c not in merge_keys
            ]
            season_df[stat_cols] = season_df[stat_cols].fillna(0)

            # ── Step 4: Add the season year ────────────────────────────────
            season_df["season"] = year

            # ── Step 5: Assign positions based on dominant stat category ───
            season_df["position"] = season_df.apply(self._assign_position, axis=1)

            # Drop players we couldn't assign a position to
            before = len(season_df)
            season_df = season_df[season_df["position"].notna()].copy()
            logger.info(
                "  Dropped %d players with no assignable position. %d remaining.",
                before - len(season_df), len(season_df),
            )

            all_rows.append(season_df)

        if not all_rows:
            logger.warning("No college data collected — returning empty DataFrame.")
            return pd.DataFrame()

        # Combine all seasons into one DataFrame
        df = pd.concat(all_rows, ignore_index=True)

        # ── Step 6: Apply top-N filter ─────────────────────────────────────
        logger.info("Applying top-N position filter…")
        df = self._apply_top_n_filter(df)

        if df.empty:
            logger.warning("Top-N filter removed all rows — returning empty DataFrame.")
            return pd.DataFrame()

        # ── Step 7: Add consistency flags ──────────────────────────────────
        logger.info("Adding consistency flags…")
        df = self._add_consistency_flags(df)

        df = df.reset_index(drop=True)
        logger.info(
            "College DataFrame complete: %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df