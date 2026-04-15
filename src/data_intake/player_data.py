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

    print(df.head())          # peek at the first 5 rows
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
    season            — the NFL season year (e.g. 2023)
    games_played      — how many games played that season
    pass_yards, rush_yards, rec_yards, etc. — season stats
    fantasy_pts_*     — fantasy points in standard / half-PPR / PPR formats

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
import time

import requests
import pandas as pd

from .links import (
    SLEEPER_ALL_PLAYERS_URL,
    SLEEPER_PLAYER_STATS_URL,
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
logger = logging.getLogger(__name__)   # __name__ = "data_intake.player_data"


class SleeperPlayerData:
    """
    Collects NFL player stats from the Sleeper API.

    HOW TO USE THIS CLASS:
        fetcher = SleeperPlayerData()
        df = fetcher.get_player_dataframe()

    WHAT IS A CLASS?
        A class is a blueprint for an object.  It groups related data and
        functions (called "methods") together.  The methods below all start
        with "self" which refers to this specific object.
    """

    def __init__(self):
        """
        __init__ is the constructor — it runs automatically when you create
        a new SleeperPlayerData() object.  We use it to set up a "session"
        which is a persistent connection to the internet that's more efficient
        than opening a new connection for every single API call.
        """
        # A requests.Session reuses the same connection for multiple calls.
        # This is faster and more polite than creating a new connection each time.
        self.session = requests.Session()

        # We set a header so the API knows our request is coming from Python.
        # Some servers reject requests that don't identify themselves.
        self.session.headers.update({"User-Agent": "FantasyFootballRookiePredictor/1.0"})

        logger.info("SleeperPlayerData initialized — ready to fetch.")

    # -----------------------------------------------------------------------
    # PRIVATE HELPER METHODS  (name starts with _ by convention)
    # These are internal tools used by the public methods below.
    # -----------------------------------------------------------------------

    def _get(self, url: str) -> dict | list | None:
        """
        Makes a single GET request to the given URL and returns the JSON data.

        PARAMETERS:
            url (str) : The web address to call.

        RETURNS:
            The parsed JSON response (usually a dict or list), or None on error.

        WHAT IS JSON?
            JSON (JavaScript Object Notation) looks like a Python dictionary.
            requests automatically converts JSON text into a Python dict/list
            via response.json().
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            # raise_for_status() raises an error if the server returned a
            # "bad" status code like 404 (not found) or 500 (server error).
            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error for %s — %s", url, e)
        except requests.exceptions.RequestException as e:
            logger.warning("Request failed for %s — %s", url, e)

        return None   # something went wrong; caller handles this gracefully

    def _fetch_all_players(self) -> dict:
        """
        Downloads the full Sleeper player database.

        RETURNS:
            dict : Keys are player_id strings, values are player info dicts.
                   Example:
                       {
                         "4046": {
                           "full_name": "Patrick Mahomes",
                           "position": "QB",
                           "team": "KC",
                           "college": "Texas Tech",
                           "age": 28,
                           "years_exp": 7,
                           ...
                         },
                         ...
                       }
        """
        logger.info("Downloading full Sleeper player list from: %s", SLEEPER_ALL_PLAYERS_URL)
        data = self._get(SLEEPER_ALL_PLAYERS_URL)

        if data is None:
            logger.error("Failed to fetch player list — returning empty dict.")
            return {}

        logger.info("Downloaded %d total players.", len(data))
        return data

    def _filter_fantasy_players(self, all_players: dict) -> dict:
        """
        Keeps only active players at fantasy-relevant positions (QB/RB/WR/TE).

        WHY FILTER?
            Sleeper's player list includes every NFL player — linemen,
            kickers, practice squad players, etc.  We only care about
            the skill positions relevant to fantasy football.

        PARAMETERS:
            all_players (dict) : The full Sleeper player dictionary.

        RETURNS:
            dict : Subset of all_players, same structure, fantasy positions only.
        """
        filtered = {
            player_id: info
            for player_id, info in all_players.items()
            if info.get("position") in FANTASY_POSITIONS
            and info.get("active", False)        # skip retired/cut players
        }
        logger.info("Filtered to %d active fantasy-relevant players.", len(filtered))
        return filtered

    def _fetch_player_stats(self, player_id: str, season: int) -> dict:
        """
        Downloads one player's stats for one season.

        PARAMETERS:
            player_id (str) : Sleeper's internal player ID string.
            season    (int) : The NFL season year (e.g. 2023).

        RETURNS:
            A dictionary of stat names → values, or an empty dict if missing.
        """
        url = SLEEPER_PLAYER_STATS_URL.format(player_id=player_id, season=season)
        data = self._get(url)

        if data is None:
            return {}   # no stats found — caller will handle this gracefully

        # Sleeper wraps the actual numbers inside a "stats" key.
        return data.get("stats", {})

    def _build_player_row(
        self,
        player_id: str,
        info: dict,
        season: int,
        stats: dict,
    ) -> dict:
        """
        Takes raw API data for one player+season and organises it into a
        flat dictionary (one key per column in our final DataFrame).

        PARAMETERS:
            player_id (str)  : Sleeper player ID.
            info      (dict) : Player bio info (name, team, college, etc.).
            season    (int)  : The season year.
            stats     (dict) : The stats dictionary from Sleeper.

        RETURNS:
            A flat dictionary ready to become one row in a DataFrame.

        WHAT IS .get(key, default)?
            dict.get("key", default_value) returns the value if the key exists,
            otherwise returns default_value instead of crashing.
            We use 0 as the default for numeric stats (missing = 0 played).

        ABOUT years_exp AND draft_year:
            Sleeper stores "years_exp" directly on every player record.
            It represents how many NFL seasons the player has completed:
                0 = current rookie (in their first NFL season)
                1 = second-year player
                5 = five years of experience
                etc.

            We calculate draft_year from it:
                draft_year = CURRENT_NFL_SEASON - years_exp

            EXAMPLE:
                Patrick Mahomes, years_exp = 7 in 2024
                draft_year = 2024 - 7 = 2017  ✓ (correct, he was drafted in 2017)

            WHY IS THIS BETTER THAN season - 1?
                The old approach assumed every player's last college season
                was always one year before the NFL season we're looking at.
                That breaks for players who:
                    - Redshirted a year in college
                    - Took a gap year before the draft
                    - Came back for a 5th college season

                With draft_year, we always know exactly when they entered
                the league, so data_cleaning can look up:
                    last_college_season = draft_year - 1
                ...with much higher accuracy.
        """
        # ── Calculate draft year from years_exp ───────────────────────────
        # years_exp may be None for some players (missing data).
        # We store None (which becomes NaN in a DataFrame) rather than
        # guessing, so data_cleaning can handle it explicitly.
        years_exp  = info.get("years_exp")          # int or None
        draft_year = (
            CURRENT_NFL_SEASON - years_exp
            if years_exp is not None
            else None                               # NaN in DataFrame
        )

        return {
            # ── Identity columns ──────────────────────────────────────────
            "player_id"         : player_id,
            "season"            : season,
            "full_name"         : info.get("full_name", "Unknown"),
            "position"          : info.get("position"),
            "nfl_team"          : info.get("team"),          # current NFL team abbreviation
            "college"           : info.get("college"),       # college they attended
            "age"               : info.get("age"),

            # ── Experience & draft info ───────────────────────────────────
            # years_exp = 0 means they are a current rookie this season
            # draft_year tells us which year they entered the NFL
            # last_college_season (computed in data_cleaning) = draft_year - 1
            "years_exp"         : years_exp,
            "draft_year"        : draft_year,

            # ── Passing stats (meaningful for QBs) ───────────────────────
            "pass_attempts"     : stats.get("pass_att", 0),
            "pass_completions"  : stats.get("pass_cmp", 0),
            "pass_yards"        : stats.get("pass_yd", 0),
            "pass_touchdowns"   : stats.get("pass_td", 0),
            "interceptions"     : stats.get("pass_int", 0),
            # Completion % = completions / attempts  (avoid dividing by 0)
            "completion_pct"    : (
                round(stats.get("pass_cmp", 0) / stats.get("pass_att", 1) * 100, 1)
                if stats.get("pass_att", 0) > 0 else 0.0
            ),

            # ── Rushing stats (RBs + mobile QBs) ─────────────────────────
            "rush_attempts"     : stats.get("rush_att", 0),
            "rush_yards"        : stats.get("rush_yd", 0),
            "rush_touchdowns"   : stats.get("rush_td", 0),
            "yards_per_carry"   : (
                round(stats.get("rush_yd", 0) / stats.get("rush_att", 1), 2)
                if stats.get("rush_att", 0) > 0 else 0.0
            ),

            # ── Receiving stats (WRs, TEs, pass-catching RBs) ────────────
            "targets"           : stats.get("rec_tgt", 0),
            "receptions"        : stats.get("rec", 0),
            "rec_yards"         : stats.get("rec_yd", 0),
            "rec_touchdowns"    : stats.get("rec_td", 0),
            "yards_per_rec"     : (
                round(stats.get("rec_yd", 0) / stats.get("rec", 1), 2)
                if stats.get("rec", 0) > 0 else 0.0
            ),

            # ── Fantasy points (three common scoring formats) ─────────────
            # Standard  = no bonus for receptions
            # Half-PPR  = 0.5 pts per reception
            # Full-PPR  = 1.0 pt per reception
            "fantasy_pts_standard" : stats.get("pts_std", 0.0),
            "fantasy_pts_half_ppr" : stats.get("pts_half_ppr", 0.0),
            "fantasy_pts_ppr"      : stats.get("pts_ppr", 0.0),

            # ── Games played ─────────────────────────────────────────────
            # Used later to compute per-game averages and flag injury seasons.
            # A player with only 4 games played had a very different season
            # than one with 17 — raw totals alone are misleading.
            "games_played"      : stats.get("gp", 0),
        }

    # -----------------------------------------------------------------------
    # PUBLIC METHOD — this is what you call from outside the class
    # -----------------------------------------------------------------------

    def get_player_dataframe(self) -> pd.DataFrame:
        """
        Orchestrates the full data collection pipeline and returns a DataFrame.

        STEPS:
            1. Download the full Sleeper player list.
            2. Filter to fantasy-relevant positions.
            3. For each player, loop through the last N seasons and fetch stats.
            4. Combine everything into one DataFrame.
            5. Basic cleanup (remove all-zero rows, reset index).

        RETURNS:
            pd.DataFrame : One row per player-season.
                           Columns = all keys from _build_player_row().
        """
        # STEP 1 & 2 — get filtered player list
        all_players      = self._fetch_all_players()
        fantasy_players  = self._filter_fantasy_players(all_players)

        if not fantasy_players:
            logger.error("No fantasy players found — returning empty DataFrame.")
            return pd.DataFrame()

        # STEP 3 — loop and collect
        rows = []   # we'll fill this list with one dict per player-season

        # Figure out which seasons to collect (e.g. [2023, 2022, 2021])
        seasons = [
            CURRENT_NFL_SEASON - i
            for i in range(NFL_SEASONS_TO_COLLECT)
        ]

        total  = len(fantasy_players)
        logger.info(
            "Fetching stats for %d players across seasons %s…",
            total, seasons
        )

        for idx, (player_id, info) in enumerate(fantasy_players.items(), start=1):
            # Progress log every 100 players so you know it's still running.
            if idx % 100 == 0:
                logger.info("  Progress: %d / %d players processed…", idx, total)

            for season in seasons:
                stats = self._fetch_player_stats(player_id, season)
                row   = self._build_player_row(player_id, info, season, stats)
                rows.append(row)

            # Sleep 0.05 seconds between players to avoid hammering the API.
            # Being polite prevents your IP from getting temporarily blocked.
            time.sleep(0.05)

        # STEP 4 — convert list of dicts → DataFrame
        # pd.DataFrame(rows) takes our list of dictionaries and turns each
        # dict into one row, with dict keys becoming column names.
        df = pd.DataFrame(rows)

        # STEP 5 — basic cleanup
        # Drop rows where the player recorded 0 games played in a season.
        # (They were on a roster but never played — not useful for our model.)
        before = len(df)
        df = df[df["games_played"] > 0].copy()
        logger.info(
            "Removed %d player-season rows with 0 games played.",
            before - len(df)
        )

        # reset_index(drop=True) renumbers rows 0, 1, 2, … after filtering.
        # drop=True means don't keep the old index as a column.
        df = df.reset_index(drop=True)

        logger.info(
            "Done! Final DataFrame: %d rows × %d columns.",
            len(df), len(df.columns)
        )
        return df