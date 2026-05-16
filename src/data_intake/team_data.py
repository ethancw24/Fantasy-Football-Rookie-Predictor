"""
data_intake/team_data.py
========================

PURPOSE:
    Fetches NFL team-level statistics and returns them as a DataFrame.
    One row = one team's stats for one season.

WHY WE SWITCHED FROM ESPN:
    The ESPN hidden API was returning data, but the nested JSON structure
    didn't match our parsing code — resulting in 96 rows of all-zeros.
    nfl_data_py (nflverse) provides schedule data that we can use to
    compute team records and scoring stats reliably.

WHAT WE GET FROM nfl_data_py SCHEDULES:
    - Win / loss / tie record per team per season
    - Win percentage
    - Points scored per game  (offensive output)
    - Points allowed per game (defensive quality)
    - Head coach name (schedules include coach per game)
    - Home vs away breakdown

WHAT WE DON'T GET (future improvement):
    - Passer rating, rushing yards per game, defensive yards per game
      These require play-by-play data (nfl.import_pbp_data) which is
      much larger and slower to download.  We'll add them in a future stage.

WHY TEAM DATA MATTERS FOR FANTASY FOOTBALL:
    Your project accounts for several team-level factors:
        Previous season win/loss record  → team quality indicator
        Head coach                       → offensive scheme and stability
        Points per game                  → how often this team scores
        Points allowed per game          → how aggressive they need to be

    When we join this with player data, a rookie joining a high-scoring
    team gets a fantasy ceiling boost even before playing a snap.

HOW TO USE:
    fetcher = SleeperTeamData()
    df = fetcher.get_team_dataframe(seasons=[2023, 2024, 2025])
    print(df.head())
"""

import logging

import pandas as pd
import nfl_data_py as nfl

from .links import (
    CURRENT_NFL_SEASON,
    NFL_SEASONS_TO_COLLECT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NFLVERSE → SLEEPER TEAM ABBREVIATION MAP
# ---------------------------------------------------------------------------
# nfl_data_py uses nflverse abbreviations.  Sleeper uses slightly different
# ones for some teams.  This map normalises them so our team merge works.
#
# Add more entries here if you notice team names not matching in the merge.
# ---------------------------------------------------------------------------
NFLVERSE_TO_SLEEPER = {
    "LA"  : "LAR",   # Los Angeles Rams (nflverse uses "LA", Sleeper uses "LAR")
    "JAC" : "JAX",   # Jacksonville Jaguars
    "OAK" : "LV",    # Oakland Raiders → now Las Vegas Raiders
    "SD"  : "LAC",   # San Diego Chargers → now Los Angeles Chargers
    "STL" : "LAR",   # St. Louis Rams → now Los Angeles Rams
}


def _normalize_team(abbr: str) -> str:
    """
    Converts an nflverse team abbreviation to its Sleeper equivalent.
    If the abbreviation isn't in the map, it's returned unchanged.
    """
    return NFLVERSE_TO_SLEEPER.get(abbr, abbr)


class SleeperTeamData:
    """
    Collects NFL team statistics using nfl_data_py (nflverse) schedule data.

    Each row in the resulting DataFrame represents one team for one season,
    with columns for win/loss record, scoring, and head coach.

    Note: The class is named SleeperTeamData for historical reasons — it
    no longer uses the Sleeper API for team stats, but the name is kept
    to avoid changing other files.
    """

    def __init__(self):
        logger.info("SleeperTeamData initialized (using nfl_data_py schedules).")

    def _compute_team_records(
        self,
        schedules : pd.DataFrame,
        seasons   : list[int],
    ) -> pd.DataFrame:
        """
        Computes win/loss/tie record, scoring stats, and head coach
        for every team in every season from the schedule DataFrame.

        HOW THIS WORKS:
            nfl_data_py schedules have one row per GAME, with both
            home_team and away_team.  For each game we need to give
            credit to BOTH teams — the winner gets a W and the loser gets an L.

            We do this by building two views of the same game:
                - home team view: home_team is "our team", away_team is opponent
                - away team view: away_team is "our team", home_team is opponent

            Then we stack both views, group by (team, season), and sum up
            wins, losses, ties, points scored, and points allowed.

        PARAMETERS:
            schedules (pd.DataFrame) : Schedule data from nfl.import_schedules().
            seasons   (list[int])    : Which seasons to include.

        RETURNS:
            pd.DataFrame : One row per team-season.
        """
        # Filter to regular season only (game_type == 'REG')
        # We don't want playoff wins inflating a team's record.
        reg_season = schedules[
            (schedules["season"].isin(seasons)) &
            (schedules["game_type"] == "REG")
        ].copy()

        if reg_season.empty:
            logger.warning("No regular season games found for seasons: %s", seasons)
            return pd.DataFrame()

        # Drop rows where either score is missing (game not yet played)
        reg_season = reg_season.dropna(subset=["home_score", "away_score"])

        rows = []

        for season in seasons:
            season_games = reg_season[reg_season["season"] == season]

            if season_games.empty:
                logger.warning("No completed games found for season %d.", season)
                continue

            # Get all unique teams that played in this season
            home_teams = set(season_games["home_team"].dropna())
            away_teams = set(season_games["away_team"].dropna())
            all_teams  = home_teams | away_teams   # union of both sets

            for raw_team in sorted(all_teams):
                # Normalize the abbreviation to match Sleeper's format
                team = _normalize_team(raw_team)

                # ── Home games ───────────────────────────────────────────
                home_games = season_games[season_games["home_team"] == raw_team]
                home_wins  = (home_games["home_score"] > home_games["away_score"]).sum()
                home_losses= (home_games["home_score"] < home_games["away_score"]).sum()
                home_ties  = (home_games["home_score"] == home_games["away_score"]).sum()
                home_scored   = home_games["home_score"].sum()
                home_allowed  = home_games["away_score"].sum()

                # ── Away games ───────────────────────────────────────────
                away_games = season_games[season_games["away_team"] == raw_team]
                away_wins  = (away_games["away_score"] > away_games["home_score"]).sum()
                away_losses= (away_games["away_score"] < away_games["home_score"]).sum()
                away_ties  = (away_games["away_score"] == away_games["home_score"]).sum()
                away_scored  = away_games["away_score"].sum()
                away_allowed = away_games["home_score"].sum()

                # ── Totals ───────────────────────────────────────────────
                wins   = int(home_wins   + away_wins)
                losses = int(home_losses + away_losses)
                ties   = int(home_ties   + away_ties)
                games  = wins + losses + ties
                total_scored  = home_scored  + away_scored
                total_allowed = home_allowed + away_allowed

                # ── Head coach ───────────────────────────────────────────
                # nfl_data_py schedules include the home and away coach per game.
                # We take the most common coach listed for this team — handles
                # mid-season coaching changes by picking the majority.
                #
                # home_coach column: coach for the home team that game
                # away_coach column: coach for the away team that game
                home_coaches = home_games["home_coach"].dropna()
                away_coaches = away_games["away_coach"].dropna()
                all_coaches  = pd.concat([home_coaches, away_coaches])

                if all_coaches.empty:
                    head_coach = "Unknown"
                else:
                    # mode() returns the most frequent value(s)
                    head_coach = all_coaches.mode().iloc[0]

                rows.append({
                    # ── Identity ─────────────────────────────────────────
                    "nfl_team" : team,
                    "season"   : season,

                    # ── Win / Loss record ─────────────────────────────────
                    "wins"     : wins,
                    "losses"   : losses,
                    "ties"     : ties,
                    "win_pct"  : round(wins / games, 3) if games > 0 else 0.0,

                    # ── Scoring ───────────────────────────────────────────
                    # Points per game — how productive the offense was
                    "points_per_game" : round(total_scored  / games, 1) if games > 0 else 0.0,
                    # Points allowed per game — lower = better defense
                    "points_allowed_per_game" : round(total_allowed / games, 1) if games > 0 else 0.0,

                    # ── Coaching ──────────────────────────────────────────
                    "head_coach" : head_coach,

                    # ── Advanced stats (not yet available) ────────────────
                    # These require play-by-play data — future improvement.
                    "passer_rating"      : None,
                    "rush_yards_per_game": None,
                    "pass_yards_per_game": None,
                })

        return pd.DataFrame(rows)

    def get_team_dataframe(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """
        Collects team records and scoring stats for the given seasons.

        PARAMETERS:
            seasons (list[int] | None) :
                List of season years (e.g. [2023, 2024, 2025]).
                Defaults to the last NFL_SEASONS_TO_COLLECT seasons.

        RETURNS:
            pd.DataFrame : One row per team-season.
        """
        if seasons is None:
            seasons = [
                CURRENT_NFL_SEASON - i
                for i in range(NFL_SEASONS_TO_COLLECT)
            ]

        logger.info("Fetching NFL team data for seasons: %s", seasons)

        # Download schedule data from nfl_data_py.
        # import_schedules() fetches a parquet file from the nflverse project.
        # It includes every game (regular season + playoffs) for the given years.
        try:
            schedules = nfl.import_schedules(years=seasons)
        except Exception as e:
            logger.error("Failed to fetch schedules via nfl_data_py: %s", e)
            return pd.DataFrame()

        if schedules is None or schedules.empty:
            logger.error("nfl_data_py returned no schedule data.")
            return pd.DataFrame()

        logger.info(
            "Raw schedule data: %d games across %d seasons.",
            len(schedules), len(seasons),
        )

        df = self._compute_team_records(schedules, seasons)

        if df.empty:
            logger.error("No team records computed — returning empty DataFrame.")
            return pd.DataFrame()

        df = df.reset_index(drop=True)
        logger.info(
            "Team DataFrame complete: %d rows × %d columns.",
            len(df), len(df.columns),
        )
        return df