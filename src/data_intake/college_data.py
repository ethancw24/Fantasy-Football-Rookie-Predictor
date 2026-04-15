"""
PURPOSE:
    Fetches college football statistics from the ESPN API.
 
THE BIG IDEA:
    If we can find patterns like:
        "College WRs with >1000 receiving yards AND >10 TDs in their final
         season tend to become WR1s in the NFL within 3 years"
    ...then we can apply those patterns to current college players to predict
    who will be the next fantasy stars.
 
TOP-N FILTERING — WHY WE LIMIT THE DATASET:
    The ESPN college API covers hundreds of players per position.  Most of
    them are backups or depth players who will never be drafted.  We only
    want the players who were truly impactful contributors, so we keep:
 
        Top 300 WRs  — ranked by total receiving yards per season
        Top 200 RBs  — ranked by total rushing yards per season
        Top 150 QBs  — ranked by total passing yards per season
        Top 100 TEs  — ranked by total receiving yards per season
 
    Ranking by total yards (rather than TDs or per-game averages) captures
    volume of contribution — a key indicator of NFL draft interest.
 
CONSISTENCY FLAGS — HOW WE HANDLE MULTI-YEAR PLAYERS:
    We add two columns to every player row so the ML model can learn
    whether sustained production matters more than a single breakout year:
 
        productive_seasons  : how many seasons this player recorded ANY stats
        total_seasons       : how many seasons we have data for them at all
 
    Example interpretations:
        productive_seasons=3, total_seasons=3 → produced every year (great sign)
        productive_seasons=1, total_seasons=3 → one breakout, two quiet years
        productive_seasons=1, total_seasons=1 → only one year of data (transfer,
                                                  injury, or true freshman)
 
    We let the ML model decide how much weight to put on these — we just
    make sure the information is available.
 
WHAT DATA DO WE COLLECT?
    For each player who makes the top-N cut:
        - Bio info   : name, position, college, class year
        - Passing    : attempts, completions, yards, TDs, INTs, passer rating,
                       completion %
        - Rushing    : attempts, yards, TDs, yards per carry
        - Receiving  : receptions, yards, TDs, yards per reception
        - Consistency: productive_seasons, total_seasons (explained above)
 
HOW TO USE:
    fetcher = ESPNCollegeData()
    df = fetcher.get_college_dataframe(years=[2021, 2022, 2023])
    print(df.head())
"""
 
import time
import logging
import requests
import pandas as pd
 
from .links import (
    ESPN_COLLEGE_TEAMS_URL,
    ESPN_COLLEGE_STATS_URL,
    FANTASY_POSITIONS,
    REQUEST_TIMEOUT_SECONDS,
    CURRENT_NFL_SEASON,
)
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# POSITION MAPPING
# ESPN uses different position abbreviations for college vs NFL.
# This dict translates ESPN college positions to our standard NFL ones.
# ---------------------------------------------------------------------------
COLLEGE_TO_NFL_POSITION = {
    "QB" : "QB",   # Quarterback
    "RB" : "RB",   # Running Back
    "WR" : "WR",   # Wide Receiver
    "TE" : "TE",   # Tight End
    "HB" : "RB",   # Halfback → treat as RB
    "FB" : "RB",   # Fullback → treat as RB (rare in modern NFL)
    "SB" : "WR",   # Slot Back → treat as WR
    "FL" : "WR",   # Flanker → old term for WR
    "SE" : "WR",   # Split End → old term for WR
}
 
# ---------------------------------------------------------------------------
# TOP-N LIMITS PER POSITION
# How many players to keep per position per season, ranked by total yards.
# Stored as a dictionary so it's easy to adjust in one place.
# ---------------------------------------------------------------------------
TOP_N_PER_POSITION = {
    "WR": 300,   # wide receivers  — ranked by receiving yards
    "RB": 200,   # running backs   — ranked by rushing yards
    "QB": 150,   # quarterbacks    — ranked by passing yards
    "TE": 100,   # tight ends      — ranked by receiving yards
}
 
# ---------------------------------------------------------------------------
# RANKING STAT PER POSITION
# The yard column used to rank players and decide who makes the top-N cut.
# ---------------------------------------------------------------------------
RANKING_STAT = {
    "WR": "rec_yards",    # receiving yards
    "RB": "rush_yards",   # rushing yards
    "QB": "pass_yards",   # passing yards
    "TE": "rec_yards",    # receiving yards (same as WR)
}
 
 
class ESPNCollegeData:
    """
    Fetches college football player statistics from ESPN.
 
    Pipeline:
        1. Fetch all FBS college football teams for the given season(s).
        2. For each team, fetch their player roster.
        3. For each fantasy-relevant player, build a stats row.
        4. Rank players by total yards, keep only the top N per position.
        5. Add consistency flags (productive_seasons, total_seasons).
    """
 
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FantasyFootballRookiePredictor/1.0"})
        logger.info("ESPNCollegeData initialized.")
 
    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------
 
    def _get(self, url: str) -> dict | list | None:
        """Standard GET helper — returns parsed JSON or None on any error."""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Timeout: %s", url)
        except requests.exceptions.HTTPError as err:
            logger.error("HTTP %s error: %s", err.response.status_code, url)
        except requests.exceptions.RequestException as err:
            logger.error("Network error: %s — %s", url, err)
        return None
 
    def _fetch_all_teams(self, year: int) -> list[dict]:
        """
        Gets the full list of college football teams ESPN tracks for a season.
 
        RETURNS:
            list of dicts, each with "id" and "displayName".
            Returns [] on failure.
        """
        url  = ESPN_COLLEGE_TEAMS_URL.format(year=year)
        data = self._get(url)
        if data is None:
            return []
        teams = data.get("items", [])
        logger.info("  Found %d college teams for %d.", len(teams), year)
        return teams
 
    def _fetch_team_players(self, team_id: str, year: int) -> list[dict]:
        """
        Gets the roster of players for one team in one season.
 
        RETURNS:
            list of raw player dicts from ESPN, or [] on failure.
        """
        url  = ESPN_COLLEGE_STATS_URL.format(team_id=team_id, year=year)
        data = self._get(url)
        if data is None:
            return []
        return data.get("items", [])
 
    def _extract_stat(self, stats_list: list, stat_name: str) -> float:
        """
        Searches a list of ESPN stat objects for one named stat and returns
        its numeric value.
 
        PARAMETERS:
            stats_list (list) : List of {"name": ..., "value": ...} dicts.
            stat_name  (str)  : The name of the stat to find.
 
        RETURNS:
            float : The value if found, or 0.0 if not.
        """
        for stat in stats_list:
            if stat.get("name") == stat_name:
                try:
                    return float(stat.get("value", 0.0))
                except (ValueError, TypeError):
                    return 0.0
        return 0.0
 
    def _flatten_stats(self, player: dict) -> list[dict]:
        """
        ESPN nests stats differently across endpoints.
        This helper tries the most common locations and returns a flat list.
 
        Tries:  player → splits → categories[0] → stats
        Then:   player → stats  (direct fallback)
        """
        splits = player.get("splits", {})
        if splits:
            categories = splits.get("categories", [])
            if categories:
                return categories[0].get("stats", [])
        return player.get("stats", [])
 
    def _build_player_row(
        self,
        year: int,
        team_name: str,
        player: dict,
    ) -> dict | None:
        """
        Parses one player's raw ESPN data into a flat row dictionary.
 
        RETURNS:
            dict  — one row of stats if the player is at a fantasy position.
            None  — if the player's position is not fantasy-relevant (skip them).
        """
        athlete      = player.get("athlete", player)
        position_obj = athlete.get("position", {})
        raw_position = position_obj.get("abbreviation", "")
        nfl_position = COLLEGE_TO_NFL_POSITION.get(raw_position)
 
        # Skip non-fantasy positions (OL, DL, LB, CB, etc.)
        if nfl_position not in FANTASY_POSITIONS:
            return None
 
        stats = self._flatten_stats(player)
 
        return {
            # ── Identity ──────────────────────────────────────────────────
            "season"            : year,
            "espn_athlete_id"   : athlete.get("id"),
            "full_name"         : athlete.get("displayName", "Unknown"),
            "position"          : nfl_position,
            "espn_position_raw" : raw_position,
            "college_team"      : team_name,
            # Class year: Freshman / Sophomore / Junior / Senior
            "class_year"        : athlete.get("experience", {}).get("displayValue", "Unknown"),
 
            # ── Passing stats ─────────────────────────────────────────────
            "pass_attempts"     : self._extract_stat(stats, "passingAttempts"),
            "pass_completions"  : self._extract_stat(stats, "passingCompletions"),
            "pass_yards"        : self._extract_stat(stats, "passingYards"),
            "pass_touchdowns"   : self._extract_stat(stats, "passingTouchdowns"),
            "interceptions"     : self._extract_stat(stats, "interceptions"),
            "passer_rating"     : self._extract_stat(stats, "passerRating"),
            "completion_pct"    : (
                round(
                    self._extract_stat(stats, "passingCompletions") /
                    max(self._extract_stat(stats, "passingAttempts"), 1) * 100,
                    1
                )
            ),
 
            # ── Rushing stats ─────────────────────────────────────────────
            "rush_attempts"     : self._extract_stat(stats, "rushingAttempts"),
            "rush_yards"        : self._extract_stat(stats, "rushingYards"),
            "rush_touchdowns"   : self._extract_stat(stats, "rushingTouchdowns"),
            "yards_per_carry"   : self._extract_stat(stats, "rushingYardsPerCarry"),
 
            # ── Receiving stats ───────────────────────────────────────────
            "receptions"        : self._extract_stat(stats, "receptions"),
            "rec_yards"         : self._extract_stat(stats, "receivingYards"),
            "rec_touchdowns"    : self._extract_stat(stats, "receivingTouchdowns"),
            "yards_per_rec"     : self._extract_stat(stats, "receivingYardsPerReception"),
 
            # ── Volume indicator ──────────────────────────────────────────
            "games_played"      : self._extract_stat(stats, "gamesPlayed"),
        }
 
    # -----------------------------------------------------------------------
    # TOP-N FILTERING
    # -----------------------------------------------------------------------
 
    def _apply_top_n_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps only the top N players per position per season, ranked by
        their position's relevant yardage stat.
 
        WHY PER SEASON?
            We rank within each season independently.  A player who was top-300
            in 2021 AND 2022 keeps BOTH rows — the model sees their progression.
            A player who only cracked the top-300 in their final season still
            makes the cut for that year.
 
        HOW IT WORKS:
            For each (season, position) group:
                1. Sort by the ranking stat descending (highest yards first).
                2. Keep only the top N rows using .head(N).
 
        PARAMETERS:
            df (pd.DataFrame) : The full unfiltered DataFrame.
 
        RETURNS:
            pd.DataFrame : Filtered to top N per position per season.
        """
        filtered_groups = []   # we'll collect the filtered pieces here
 
        # Loop over every unique combination of season and position.
        # e.g. (2023, "WR"), (2023, "RB"), (2022, "WR"), etc.
        for (season, position), group in df.groupby(["season", "position"]):
 
            # Look up the limit and ranking stat for this position.
            limit       = TOP_N_PER_POSITION.get(position)
            ranking_col = RANKING_STAT.get(position)
 
            # If position isn't in our mapping (shouldn't happen but just in case)
            # keep all rows for it rather than silently dropping them.
            if limit is None or ranking_col is None:
                logger.warning(
                    "No top-N config for position '%s' — keeping all %d rows.",
                    position, len(group)
                )
                filtered_groups.append(group)
                continue
 
            # Sort by the ranking stat (highest first), then take the top N.
            # na_position="last" puts players with 0/NaN yards at the bottom
            # so they're the first ones cut when the list is trimmed.
            top_n = (
                group
                .sort_values(ranking_col, ascending=False, na_position="last")
                .head(limit)
            )
 
            logger.info(
                "  %d %s: kept top %d of %d players (ranked by %s).",
                season, position, len(top_n), len(group), ranking_col
            )
 
            filtered_groups.append(top_n)
 
        if not filtered_groups:
            return pd.DataFrame()
 
        # pd.concat glues all the filtered groups back into one DataFrame.
        return pd.concat(filtered_groups, ignore_index=True)
 
    # -----------------------------------------------------------------------
    # CONSISTENCY FLAGS
    # -----------------------------------------------------------------------
 
    def _add_consistency_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds two columns that describe each player's multi-year production
        pattern so the ML model can learn how consistency relates to NFL success.
 
        COLUMNS ADDED:
            productive_seasons (int) :
                Number of seasons this player recorded ANY stats.
                "Any stats" means at least one non-zero yard column.
                A season where everything is 0 (injury/redshirt) does NOT count.
 
            total_seasons (int) :
                Total number of seasons we have ANY data row for this player,
                regardless of whether they produced stats or not.
 
        EXAMPLES:
            Player A — 3 seasons, produced in all 3:
                productive_seasons=3, total_seasons=3
 
            Player B — 3 seasons, only produced in senior year:
                productive_seasons=1, total_seasons=3
 
            Player C — 1 season of data (transfer, injury, true freshman):
                productive_seasons=1, total_seasons=1
 
        WHY NOT HARDCODE WEIGHTS HERE?
            The ML model (in training_and_evaluate) will learn whether
            3-year producers outperform 1-year breakouts.  Our job is just
            to give it the data — not to prejudge the answer.
 
        PARAMETERS:
            df (pd.DataFrame) : The top-N filtered DataFrame.
 
        RETURNS:
            pd.DataFrame : Same DataFrame with two new columns appended.
        """
        # The yard columns we check to determine if a season was "productive."
        # If ALL of these are 0 for a season, we don't count it as productive.
        yard_cols = ["pass_yards", "rush_yards", "rec_yards"]
 
        # Make sure all yard columns exist (they should, but defensive coding
        # means we don't crash if the DataFrame is missing one for some reason).
        existing_yard_cols = [c for c in yard_cols if c in df.columns]
 
        # ── Step 1: flag each individual row as productive or not ─────────
        # A row is productive if any yard column has a value > 0.
        # .any(axis=1) checks across columns for each row — returns True/False.
        df = df.copy()   # avoid modifying the original DataFrame in place
        df["_has_stats"] = df[existing_yard_cols].gt(0).any(axis=1)
 
        # ── Step 2: count per player across all their seasons ─────────────
        # groupby("full_name") groups all rows for the same player together.
        # .transform() then writes the group-level result back to every row
        # in that group — so every row for "Bijan Robinson" gets the same counts.
 
        # productive_seasons = count of rows where _has_stats is True
        df["productive_seasons"] = (
            df.groupby("full_name")["_has_stats"]
            .transform("sum")       # sum of True values = count of productive seasons
            .astype(int)
        )
 
        # total_seasons = total number of rows (seasons) for this player
        df["total_seasons"] = (
            df.groupby("full_name")["_has_stats"]
            .transform("count")     # count of all rows regardless of value
            .astype(int)
        )
 
        # ── Step 3: clean up the helper column ───────────────────────────
        # _has_stats was only needed for the calculation above.
        # We drop it so it doesn't appear in the final CSV.
        df = df.drop(columns=["_has_stats"])
 
        return df
 
    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------
 
    def get_college_dataframe(self, years: list[int] | None = None) -> pd.DataFrame:
        """
        Full pipeline: fetch → filter to top N → add consistency flags → return.
 
        PARAMETERS:
            years (list[int] | None) :
                College football season years to collect.
                e.g. [2021, 2022, 2023]
                Defaults to the last completed college season if not provided.
 
        RETURNS:
            pd.DataFrame : One row per player-season that made the top-N cut,
                           with consistency flag columns appended.
 
        EXAMPLE:
            fetcher = ESPNCollegeData()
            df = fetcher.get_college_dataframe(years=[2021, 2022, 2023])
            print(df[["full_name", "position", "season",
                       "productive_seasons", "total_seasons"]].head(10))
        """
        if years is None:
            years = [CURRENT_NFL_SEASON - 1]   # last completed college season
 
        rows = []
 
        # ── STAGE 1: Fetch all raw player rows ────────────────────────────
        for year in years:
            logger.info("=== Fetching college football data for %d ===", year)
 
            teams = self._fetch_all_teams(year)
            if not teams:
                logger.warning("No teams found for %d — skipping.", year)
                continue
 
            for team_idx, team in enumerate(teams, start=1):
                team_id   = team.get("id")
                team_name = team.get("displayName", f"Team {team_id}")
 
                if not team_id:
                    continue
 
                if team_idx % 20 == 0:
                    logger.info("  Teams processed: %d / %d", team_idx, len(teams))
 
                players = self._fetch_team_players(team_id, year)
 
                for player in players:
                    row = self._build_player_row(year, team_name, player)
                    if row is not None:
                        rows.append(row)
 
                time.sleep(0.1)   # be polite to the ESPN API server
 
        if not rows:
            logger.warning("No college data collected — returning empty DataFrame.")
            return pd.DataFrame()
 
        df = pd.DataFrame(rows)
 
        # Remove rows where games_played = 0 (redshirt / never played).
        before = len(df)
        df = df[df["games_played"] > 0].copy()
        logger.info(
            "Removed %d zero-games rows. Remaining: %d rows.",
            before - len(df), len(df)
        )
 
        # ── STAGE 2: Apply top-N filter ───────────────────────────────────
        # Keep only the top 300 WRs, 200 RBs, 150 QBs, 100 TEs per season.
        logger.info("Applying top-N position filter…")
        df = self._apply_top_n_filter(df)
 
        # ── STAGE 3: Add consistency flags ────────────────────────────────
        # Add productive_seasons and total_seasons columns.
        logger.info("Adding consistency flags…")
        df = self._add_consistency_flags(df)
 
        df = df.reset_index(drop=True)
        logger.info(
            "College DataFrame complete: %d rows × %d columns.",
            len(df), len(df.columns)
        )
        return df