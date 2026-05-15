"""
PURPOSE:
    One central place that stores every URL (web address / API endpoint) that
    this project calls.  Think of it like a contacts list — if a URL ever
    changes you only have to update it here, not hunt through every file.

HOW TO USE THIS FILE:
    In any other file in this package, write:
        from .links import SLEEPER_PLAYERS_URL
    Then use the variable wherever you need the URL.
"""

# Imports
from datetime import date

"""
SLEEPER API  (https://docs.sleeper.com/)
Free, public, no API key required.
"""

# Returns a giant JSON dictionary of every NFL player Sleeper knows about.
# Keys are player IDs (strings), values are player detail dictionaries.
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# Returns stats for one player for one season.
# {player_id} and {season} are placeholders — we fill them in with .format()
# Example filled in: https://api.sleeper.app/v1/stats/nfl/player/4046?season_type=regular&season=2023
SLEEPER_PLAYER_STATS_URL = (
    "https://api.sleeper.app/v1/stats/nfl/player/{player_id}"
    "?season_type=regular&season={season}"
)

# Returns stats for ALL NFL players for an entire season in ONE request.
# This is much faster than fetching one player at a time.
# {season} = the season year (e.g. 2025)
# Example: https://api.sleeper.app/v1/stats/nfl/regular/2025?season_type=regular
SLEEPER_BATCH_STATS_URL = (
    "https://api.sleeper.app/v1/stats/nfl/regular/{season}"
    "?season_type=regular"
)

# Returns trending players (most added/dropped in fantasy leagues recently).
# Useful as a quick check that the Sleeper API is responding.
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/add"

# Returns NFL team info from Sleeper.
# We use this for win/loss record, offensive & defensive ratings.
SLEEPER_TEAMS_URL = "https://api.sleeper.app/v1/league/{league_id}/rosters"

"""
ESPN HIDDEN / UNOFFICIAL API
ESPN does not publish official docs for these endpoints, but they are
publicly accessible.  They may change without warning — if something breaks,
check that the URL still works in your browser first.
"""

# Returns combine measurements for NFL draft prospects.
# {year} = draft year (e.g. 2023)
ESPN_COMBINE_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    "/seasons/{year}/combine?limit=1000"
)

# Returns college football player stats for a specific team and season.
# {team_id} = ESPN's internal ID for the college team
# {year}    = the season year (e.g. 2022)
ESPN_COLLEGE_STATS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/seasons/{year}/teams/{team_id}/athletes?limit=200"
)

# Returns a list of all college football teams ESPN tracks.
# We use this to get {team_id} values for the URL above.
ESPN_COLLEGE_TEAMS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/seasons/{year}/teams?limit=1000"
)

# Returns NFL team statistics (offensive/defensive ratings, scores, etc.)
# {year} = season year,  {team_id} = ESPN's NFL team ID
ESPN_NFL_TEAM_STATS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    "/seasons/{year}/teams/{team_id}?enable=stats"
)

"""
CONSTANTS — shared settings used across multiple files
"""

# The most recent completed NFL regular season.
# Update this each year after the season ends.
# Since this app will be ran after the superbowl and the superbowl has no set end date anything March to August will update to the to the previous season
# Anything from September to February will be set to the current season
today = date.today()
year = today.year
month = today.month
# Offseason (March-August): use last completed season (e.g. April 2026 → 2025)
# In-season (September-February): use current year (e.g. October 2025 → 2025)
if month >= 3 and month <= 8:
    CURRENT_NFL_SEASON = year - 1
else:
    CURRENT_NFL_SEASON = year

# How many past NFL seasons of data to collect per player.
# 3 seasons gives a good trend without making the project too slow.
NFL_SEASONS_TO_COLLECT = 3   # will collect: CURRENT_NFL_SEASON, CURRENT_NFL_SEASON-1, CURRENT_NFL_SEASON-2

# Positions we care about for fantasy football.
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

# Default timeout (in seconds) when making HTTP requests.
# If the API doesn't respond within this time, we stop waiting and log an error.
REQUEST_TIMEOUT_SECONDS = 10