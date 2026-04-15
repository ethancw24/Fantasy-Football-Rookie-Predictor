"""
data_intake/team_data.py
========================

PURPOSE:
    Fetches NFL team-level statistics from the ESPN API.
    Team context matters a lot for fantasy football — a great player on a
    bad offense will score fewer fantasy points than an average player on an
    elite offense.

WHY TEAM DATA?
    Your project idea mentions several team-level factors to consider:
        Previous season win/loss record
        Previous season passer rating (how good the QB situation was)
        Previous season rushing rating
        Previous season defensive rating
        Previous season offensive line rating
        Head coach (coaching stability / offensive scheme)

    When we later combine this with player data, a rookie's fantasy ceiling
    will be partially predicted by the team they land on.

HOW TO USE:
    fetcher = SleeperTeamData()
    df = fetcher.get_team_dataframe()
    print(df.head())
"""

import logging
import requests
import pandas as pd

from .links import (
    ESPN_NFL_TEAM_STATS_URL,
    CURRENT_NFL_SEASON,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NFL TEAM REFERENCE TABLE
# ESPN uses numeric IDs for teams, not abbreviations.
# This dictionary maps ESPN's IDs to human-readable info.
# We use it so our final DataFrame has readable team names instead of numbers.
# ---------------------------------------------------------------------------

# Format:  espn_team_id : { "abbreviation": ..., "full_name": ..., "conference": ... }
NFL_TEAMS = {
    1:  {"abbreviation": "ATL", "full_name": "Atlanta Falcons",        "conference": "NFC"},
    2:  {"abbreviation": "BUF", "full_name": "Buffalo Bills",          "conference": "AFC"},
    3:  {"abbreviation": "CHI", "full_name": "Chicago Bears",          "conference": "NFC"},
    4:  {"abbreviation": "CIN", "full_name": "Cincinnati Bengals",     "conference": "AFC"},
    5:  {"abbreviation": "CLE", "full_name": "Cleveland Browns",       "conference": "AFC"},
    6:  {"abbreviation": "DAL", "full_name": "Dallas Cowboys",         "conference": "NFC"},
    7:  {"abbreviation": "DEN", "full_name": "Denver Broncos",         "conference": "AFC"},
    8:  {"abbreviation": "DET", "full_name": "Detroit Lions",          "conference": "NFC"},
    9:  {"abbreviation": "GB",  "full_name": "Green Bay Packers",      "conference": "NFC"},
    10: {"abbreviation": "TEN", "full_name": "Tennessee Titans",       "conference": "AFC"},
    11: {"abbreviation": "IND", "full_name": "Indianapolis Colts",     "conference": "AFC"},
    12: {"abbreviation": "KC",  "full_name": "Kansas City Chiefs",     "conference": "AFC"},
    13: {"abbreviation": "LV",  "full_name": "Las Vegas Raiders",      "conference": "AFC"},
    14: {"abbreviation": "LAR", "full_name": "Los Angeles Rams",       "conference": "NFC"},
    15: {"abbreviation": "MIA", "full_name": "Miami Dolphins",         "conference": "AFC"},
    16: {"abbreviation": "MIN", "full_name": "Minnesota Vikings",      "conference": "NFC"},
    17: {"abbreviation": "NE",  "full_name": "New England Patriots",   "conference": "AFC"},
    18: {"abbreviation": "NO",  "full_name": "New Orleans Saints",     "conference": "NFC"},
    19: {"abbreviation": "NYG", "full_name": "New York Giants",        "conference": "NFC"},
    20: {"abbreviation": "NYJ", "full_name": "New York Jets",          "conference": "AFC"},
    21: {"abbreviation": "PHI", "full_name": "Philadelphia Eagles",    "conference": "NFC"},
    22: {"abbreviation": "ARI", "full_name": "Arizona Cardinals",      "conference": "NFC"},
    23: {"abbreviation": "PIT", "full_name": "Pittsburgh Steelers",    "conference": "AFC"},
    24: {"abbreviation": "LAC", "full_name": "Los Angeles Chargers",   "conference": "AFC"},
    25: {"abbreviation": "SF",  "full_name": "San Francisco 49ers",    "conference": "NFC"},
    26: {"abbreviation": "SEA", "full_name": "Seattle Seahawks",       "conference": "NFC"},
    27: {"abbreviation": "TB",  "full_name": "Tampa Bay Buccaneers",   "conference": "NFC"},
    28: {"abbreviation": "WAS", "full_name": "Washington Commanders",  "conference": "NFC"},
    29: {"abbreviation": "CAR", "full_name": "Carolina Panthers",      "conference": "NFC"},
    30: {"abbreviation": "JAX", "full_name": "Jacksonville Jaguars",   "conference": "AFC"},
    33: {"abbreviation": "BAL", "full_name": "Baltimore Ravens",       "conference": "AFC"},
    34: {"abbreviation": "HOU", "full_name": "Houston Texans",         "conference": "AFC"},
}


class SleeperTeamData:
    """
    Collects NFL team statistics from the ESPN API.

    Each row in the resulting DataFrame represents one team for one season,
    with columns for win/loss record, offensive ratings, defensive ratings,
    and head coach information.
    """

    def __init__(self):
        """
        Set up the HTTP session and log that we're ready.
        """
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FantasyFootballRookiePredictor/1.0"})
        logger.info("SleeperTeamData initialized.")

    def _get(self, url: str) -> dict | list | None:
        """
        Makes a single HTTP GET request and returns parsed JSON.
        Handles errors gracefully so one bad call doesn't crash everything.
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching: %s", url)
        except requests.exceptions.HTTPError as err:
            logger.error("HTTP error: %s — %s", url, err)
        except requests.exceptions.RequestException as err:
            logger.error("Network error: %s — %s", url, err)
        return None

    def _extract_stat(self, stats_list: list, stat_name: str) -> float:
        """
        ESPN returns stats as a list of objects.  This helper searches that
        list for a specific stat by name and returns its numeric value.

        EXAMPLE ESPN stats_list item:
            {"name": "passerRating", "displayValue": "103.4", "value": 103.4}

        PARAMETERS:
            stats_list (list) : The list of stat objects from ESPN.
            stat_name  (str)  : The "name" field we're looking for.

        RETURNS:
            float : The numeric value, or 0.0 if not found.
        """
        for stat in stats_list:
            if stat.get("name") == stat_name:
                return float(stat.get("value", 0.0))
        return 0.0   # stat not found — treat as 0

    def _fetch_team_season(self, team_id: int, season: int) -> dict | None:
        """
        Fetches raw data for one team for one season from ESPN.

        PARAMETERS:
            team_id (int) : ESPN's numeric team ID.
            season  (int) : The season year (e.g. 2023).

        RETURNS:
            The raw JSON dict from ESPN, or None on failure.
        """
        url = ESPN_NFL_TEAM_STATS_URL.format(year=season, team_id=team_id)
        return self._get(url)

    def _build_team_row(self, team_id: int, season: int, data: dict) -> dict:
        """
        Parses raw ESPN JSON for one team-season into a flat row dictionary.

        PARAMETERS:
            team_id (int)  : ESPN team ID.
            season  (int)  : Season year.
            data    (dict) : Raw JSON from ESPN.

        RETURNS:
            dict : One row ready to go into the DataFrame.
        """
        # Look up our human-readable team info from our reference table above.
        team_ref  = NFL_TEAMS.get(team_id, {})

        # ESPN nests stats inside "team" → "record" and "team" → "stats"
        team_info = data.get("team", {})
        record    = team_info.get("record", {}).get("items", [{}])[0]
        stats     = team_info.get("stats", {}).get("splits", [{}])[0].get("stats", [])

        # Parse win/loss record from the record object.
        # ESPN gives these as strings like "11", so we convert to int.
        wins   = int(record.get("wins",   0))
        losses = int(record.get("losses", 0))
        ties   = int(record.get("ties",   0))
        games  = wins + losses + ties

        return {
            # ── Identity ──────────────────────────────────────────────────
            "team_id"           : team_id,
            "season"            : season,
            "team_abbreviation" : team_ref.get("abbreviation", f"ID_{team_id}"),
            "team_full_name"    : team_ref.get("full_name",    f"Team {team_id}"),
            "conference"        : team_ref.get("conference"),

            # ── Win / Loss record ─────────────────────────────────────────
            "wins"              : wins,
            "losses"            : losses,
            "ties"              : ties,
            # Win percentage = wins / total games played (0.0 – 1.0)
            "win_pct"           : round(wins / games, 3) if games > 0 else 0.0,

            # ── Offensive ratings ─────────────────────────────────────────
            # Passer rating measures QB efficiency (0–158.3 scale in NFL)
            "passer_rating"     : self._extract_stat(stats, "passerRating"),
            # Points scored per game — higher = better offense
            "points_per_game"   : self._extract_stat(stats, "pointsPerGame"),
            # Total offensive yards gained per game
            "yards_per_game_off": self._extract_stat(stats, "totalYardsPerGame"),
            # Rushing yards per game
            "rush_yards_per_game" : self._extract_stat(stats, "rushingYardsPerGame"),
            # Passing yards per game
            "pass_yards_per_game" : self._extract_stat(stats, "passingYardsPerGame"),

            # ── Defensive ratings ─────────────────────────────────────────
            # Points allowed per game — lower = better defense
            "points_allowed_per_game" : self._extract_stat(stats, "pointsAllowedPerGame"),
            # Total yards allowed per game — lower = better defense
            "yards_allowed_per_game"  : self._extract_stat(stats, "totalYardsAllowedPerGame"),
            # Sacks generated — higher means the D-line creates more pressure
            "sacks"                   : self._extract_stat(stats, "sacks"),

            # ── Coaching (useful for predicting rookie success) ───────────
            # Head coach's name — if they changed coaches, a new scheme
            # could mean unpredictable fantasy output.
            "head_coach"        : team_info.get("headCoach", {}).get("displayName", "Unknown"),
        }

    def get_team_dataframe(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """
        Collects team data for all 32 NFL teams across the given seasons.

        PARAMETERS:
            seasons (list[int] | None) :
                List of season years to fetch (e.g. [2023, 2022, 2021]).
                If None, defaults to [CURRENT_NFL_SEASON].

        RETURNS:
            pd.DataFrame : One row per team-season.

        EXAMPLE:
            fetcher = SleeperTeamData()
            df = fetcher.get_team_dataframe(seasons=[2023, 2022])
        """
        if seasons is None:
            seasons = [CURRENT_NFL_SEASON]

        rows = []

        for season in seasons:
            logger.info("Fetching team data for the %d season…", season)

            for team_id, team_ref in NFL_TEAMS.items():
                data = self._fetch_team_season(team_id, season)

                if data is None:
                    # Log the skip but keep going — one bad team shouldn't stop all others.
                    logger.warning(
                        "Skipping %s (%d) — no data returned.",
                        team_ref["abbreviation"], season
                    )
                    continue

                row = self._build_team_row(team_id, season, data)
                rows.append(row)

        if not rows:
            logger.error("No team data collected — returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).reset_index(drop=True)
        logger.info(
            "Team DataFrame complete: %d rows × %d columns.",
            len(df), len(df.columns)
        )
        return df