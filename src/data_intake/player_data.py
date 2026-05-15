"""
data_intake/player_data.py
==========================

PURPOSE:
    Fetches NFL player stats from the Sleeper API and returns them as a
    clean pandas DataFrame.  One row = one player's stats for one season.

WHAT IS THE SLEEPER API?
    Sleeper is a free fantasy football platform.  Their API is completely
    free and does not require an API key.  It provides:
        - A full list of all NFL players (bio info, team, college, etc.)
        - Season-by-season stats for every player

API DOCS:
    https://docs.sleeper.com/

HOW TO USE THIS FILE:
    from src.data_intake.player_data import SleeperPlayerData

    fetcher = SleeperPlayerData()
    df = fetcher.get_player_dataframe()

    print(df.head())           # peek at the first 5 rows
    print(df.columns.tolist()) # see every column name

COLUMNS PRODUCED (key ones):
    player_id         — Sleeper's unique ID for each player
    full_name         — e.g. "Patrick Mahomes"
    position          — QB / RB / WR / TE
    nfl_team          — e.g. "KC"
    college           — e.g. "Texas Tech"
    age               — player's age at time of data pull
    years_exp         — how many NFL seasons the player has completed
                        (0 = current rookie, 1 = second year, etc.)
    draft_year        — the year they were drafted / entered the NFL
                        calculated as: CURRENT_NFL_SEASON - years_exp
    season            — the NFL season year (e.g. 2025)
    games_played      — how many games played that season
    pass_yards, rush_yards, rec_yards, etc. — season stats
    fantasy_pts_*     — fantasy points in standard / half-PPR / PPR formats

WHY BATCH FETCHING?
    The original approach made one API call per player per season.
    With 2,896 players × 3 seasons = ~8,700 requests — that took 50+ minutes.

    The batch endpoint returns ALL players' stats for an entire season
    in a single API call.  3 seasons = just 3 requests.
    Runtime drops from 50+ minutes to under 1 minute.

    OLD URL (one player at a time):
        /stats/nfl/player/{player_id}?season_type=regular&season=2025

    NEW URL (all players at once):
        /stats/nfl/regular/2025?season_type=regular

WHY years_exp AND draft_year MATTER:
    When we later match NFL players to their college stats, we need to know
    which college season to look up.  A player drafted in 2022 played their
    last college season in 2021.
    Having draft_year stored here lets data_cleaning use:
        last_college_season = draft_year - 1
    ...instead of the rougher approximation:
        last_college_season = nfl_season - 1
    This is much more accurate, especially for players who redshirted,
    transferred, or came back for an extra college season.
"""

import logging

import requests
import pandas as pd

from .links import (
    SLEEPER_PLAYERS_URL,
    SLEEPER_BATCH_STATS_URL,
    CURRENT_NFL_SEASON,
    NFL_SEASONS_TO_COLLECT,
    FANTASY_POSITIONS,
    REQUEST_TIMEOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# logging lets us print progress messages without using print().
# It's preferred in real projects because you can control the level of detail
# (DEBUG, INFO, WARNING, ERROR) and redirect output to a file if needed.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class SleeperPlayerData:
    """
    Collects NFL player stats from the Sleeper API using batch requests.

    Instead of fetching stats one player at a time (slow), this class
    fetches ALL players' stats for a full season in a single API call (fast).

    HOW TO USE THIS CLASS:
        fetcher = SleeperPlayerData()
        df = fetcher.get_player_dataframe()
    """

    def __init__(self):
        """
        Sets up a persistent HTTP session for efficient API calls.

        A requests.Session reuses the same connection for multiple calls.
        This is faster and more polite than creating a new connection each time.
        """
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FantasyFootballRookiePredictor/1.0"})
        logger.info("SleeperPlayerData initialized — ready to fetch.")

    # -----------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -----------------------------------------------------------------------

    def _get(self, url: str) -> dict | list | None:
        """
        Makes a single GET request and returns parsed JSON, or None on error.

        WHAT IS JSON?
            JSON (JavaScript Object Notation) looks like a Python dictionary.
            requests automatically converts JSON text into a Python dict/list
            via response.json().
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error for %s — %s", url, e)
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed for %s — %s", url, e)

        return None

    def _fetch_all_players(self) -> dict:
        """
        Downloads the full Sleeper player database — bio info for every
        NFL player (name, position, team, college, age, years_exp, etc.)

        RETURNS:
            dict : Keys are player_id strings, values are player info dicts.
        """
        logger.info("Downloading full Sleeper player list…")
        data = self._get(SLEEPER_PLAYERS_URL)

        if data is None:
            logger.error("Failed to fetch player list — returning empty dict.")
            return {}

        logger.info("Downloaded %d total players.", len(data))
        return data

    def _fetch_season_stats_batch(self, season: int) -> dict:
        """
        Downloads stats for ALL NFL players for one full season in a
        single API call.

        This is the key performance improvement over the old approach.
        Instead of one request per player, we get everyone at once.

        PARAMETERS:
            season (int) : The NFL season year (e.g. 2025).

        RETURNS:
            dict : Keys are player_id strings, values are stats dicts.
                   Example:
                       {
                         "4046": {"pass_yd": 4183, "pass_td": 27, "gp": 17, ...},
                         "2749": {"rec_yd": 1074, "rec_td": 5,   "gp": 17, ...},
                         ...
                       }

        WHAT IS THE BATCH URL?
            /stats/nfl/regular/{season}?season_type=regular
            Returns every player's season totals in one response.
            Sleeper returns this as a list of objects, each with a
            "player_id" key and a "stats" key containing the numbers.
        """
        url = SLEEPER_BATCH_STATS_URL.format(season=season)
        logger.info("  Fetching batch stats for season %d…", season)

        data = self._get(url)

        if data is None:
            logger.warning("  No stats returned for season %d.", season)
            return {}

        # The batch endpoint returns a LIST of objects like:
        # [{"player_id": "4046", "stats": {"pass_yd": 4183, ...}}, ...]
        # We convert it to a dict keyed by player_id for fast lookups.
        #
        # WHAT IS A DICT COMPREHENSION?
        #   {key: value for item in list} builds a dictionary in one line.
        #   Here: for each entry in the list, use player_id as the key
        #   and the stats sub-dict as the value.
        stats_by_player = {
            entry["player_id"]: entry.get("stats", {})
            for entry in data
            if "player_id" in entry
        }

        logger.info(
            "  Season %d: received stats for %d players.",
            season, len(stats_by_player)
        )
        return stats_by_player

    def _build_player_row(
        self,
        player_id : str,
        info      : dict,
        season    : int,
        stats     : dict,
    ) -> dict:
        """
        Combines one player's bio info + one season's stats into a single
        flat dictionary — one row in our final DataFrame.

        PARAMETERS:
            player_id (str)  : Sleeper player ID.
            info      (dict) : Player bio info (name, team, college, etc.).
            season    (int)  : The season year.
            stats     (dict) : The stats dictionary from the batch response.

        RETURNS:
            dict : One row ready to be added to our DataFrame.

        WHAT IS .get(key, default)?
            dict.get("key", default_value) returns the value if the key exists,
            otherwise returns default_value instead of crashing.
            We use 0 as the default for numeric stats (missing = not played).

        ABOUT years_exp AND draft_year:
            Sleeper stores "years_exp" directly on every player record.
                0 = current rookie
                1 = second-year player
                7 = seven years of experience
            We calculate: draft_year = CURRENT_NFL_SEASON - years_exp

            EXAMPLE:
                Patrick Mahomes, years_exp = 8 in 2025
                draft_year = 2025 - 8 = 2017  ✓
        """
        # ── Calculate draft year from years_exp ───────────────────────────
        # years_exp may be None for some players (missing data).
        # We store None → NaN rather than guessing.
        years_exp  = info.get("years_exp")
        draft_year = (
            CURRENT_NFL_SEASON - years_exp
            if years_exp is not None
            else None
        )

        return {
            # ── Identity ──────────────────────────────────────────────────
            "player_id"            : player_id,
            "season"               : season,
            "full_name"            : info.get("full_name", "Unknown"),
            "position"             : info.get("position"),
            "nfl_team"             : info.get("team"),
            "college"              : info.get("college"),
            "age"                  : info.get("age"),

            # ── Experience & draft info ───────────────────────────────────
            "years_exp"            : years_exp,
            "draft_year"           : draft_year,

            # ── Passing stats (QBs) ───────────────────────────────────────
            "pass_attempts"        : stats.get("pass_att", 0),
            "pass_completions"     : stats.get("pass_cmp", 0),
            "pass_yards"           : stats.get("pass_yd",  0),
            "pass_touchdowns"      : stats.get("pass_td",  0),
            "interceptions"        : stats.get("pass_int", 0),
            "completion_pct"       : (
                round(stats.get("pass_cmp", 0) / stats.get("pass_att", 1) * 100, 1)
                if stats.get("pass_att", 0) > 0 else 0.0
            ),

            # ── Rushing stats (RBs + mobile QBs) ─────────────────────────
            "rush_attempts"        : stats.get("rush_att", 0),
            "rush_yards"           : stats.get("rush_yd",  0),
            "rush_touchdowns"      : stats.get("rush_td",  0),
            "yards_per_carry"      : (
                round(stats.get("rush_yd", 0) / stats.get("rush_att", 1), 2)
                if stats.get("rush_att", 0) > 0 else 0.0
            ),

            # ── Receiving stats (WRs, TEs, pass-catching RBs) ────────────
            "targets"              : stats.get("rec_tgt", 0),
            "receptions"           : stats.get("rec",     0),
            "rec_yards"            : stats.get("rec_yd",  0),
            "rec_touchdowns"       : stats.get("rec_td",  0),
            "yards_per_rec"        : (
                round(stats.get("rec_yd", 0) / stats.get("rec", 1), 2)
                if stats.get("rec", 0) > 0 else 0.0
            ),

            # ── Fantasy points ────────────────────────────────────────────
            "fantasy_pts_standard" : stats.get("pts_std",      0.0),
            "fantasy_pts_half_ppr" : stats.get("pts_half_ppr", 0.0),
            "fantasy_pts_ppr"      : stats.get("pts_ppr",      0.0),

            # ── Games played ──────────────────────────────────────────────
            # Used to compute per-game averages and flag injury seasons.
            "games_played"         : stats.get("gp", 0),
        }

    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------

    def get_player_dataframe(self) -> pd.DataFrame:
        """
        Runs the full data collection pipeline and returns a DataFrame.

        STEPS:
            1. Download the full Sleeper player list (bio info).
            2. For each season, fetch ALL players' stats in one batch call.
            3. Match bio info to stats and build one row per player-season.
            4. Filter to fantasy-relevant positions only (QB/RB/WR/TE).
            5. Drop rows with 0 games played (inactive that season).
            6. Return the final DataFrame.

        RETURNS:
            pd.DataFrame : One row per player-season.
        """
        # STEP 1 — player bio info
        all_players = self._fetch_all_players()
        if not all_players:
            logger.error("No players fetched — returning empty DataFrame.")
            return pd.DataFrame()

        # STEP 2 & 3 — batch stats fetch + row building
        # Seasons to collect: 2025, 2024, 2023
        seasons = [
            CURRENT_NFL_SEASON - i
            for i in range(NFL_SEASONS_TO_COLLECT)
        ]
        logger.info("Collecting seasons: %s", seasons)

        rows = []

        for season in seasons:
            # One API call gets stats for ALL players this season
            season_stats = self._fetch_season_stats_batch(season)

            # Match each player's bio info to their stats for this season
            for player_id, info in all_players.items():
                stats = season_stats.get(player_id, {})   # empty dict if no stats
                row   = self._build_player_row(player_id, info, season, stats)
                rows.append(row)

        # STEP 4 — convert to DataFrame and filter to fantasy positions
        df = pd.DataFrame(rows)

        # Keep only QB/RB/WR/TE — isin() checks if a value is in a list
        df = df[df["position"].isin(FANTASY_POSITIONS)].copy()
        logger.info("Filtered to %d fantasy-position player-season rows.", len(df))

        # STEP 5 — drop rows where the player had 0 games played that season
        # (on a roster but never took the field — not useful for the model)
        before = len(df)
        df = df[df["games_played"] > 0].copy()
        logger.info(
            "Removed %d rows with 0 games played. %d rows remaining.",
            before - len(df), len(df),
        )

        # STEP 6 — reset row numbers (0, 1, 2, …) after filtering
        df = df.reset_index(drop=True)

        logger.info(
            "Done! Player DataFrame: %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df