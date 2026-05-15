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
SLEEPER_PLAYER_STATS_URL = (
    "https://api.sleeper.app/v1/stats/nfl/player/{player_id}"
    "?season_type=regular&season={season}"
)

# Returns stats for ALL NFL players for an entire season in ONE request.
# This is much faster than fetching one player at a time.
# {season} = the season year (e.g. 2025)
SLEEPER_BATCH_STATS_URL = (
    "https://api.sleeper.app/v1/stats/nfl/regular/{season}"
    "?season_type=regular"
)

# Returns trending players (most added/dropped in fantasy leagues recently).
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/add"

# Returns NFL team info from Sleeper.
SLEEPER_TEAMS_URL = "https://api.sleeper.app/v1/league/{league_id}/rosters"

"""
ESPN HIDDEN / UNOFFICIAL API
These are kept for reference only — we no longer use ESPN endpoints
because they became unreliable (403/404 errors).
"""

# Returns combine measurements for NFL draft prospects.
# DEPRECATED — replaced by nfl_data_py in combine.py
ESPN_COMBINE_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    "/seasons/{year}/combine?limit=1000"
)

# Returns college football player stats for a specific team and season.
# DEPRECATED — replaced by CFBD in college_data.py
ESPN_COLLEGE_STATS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/seasons/{year}/teams/{team_id}/athletes?limit=200"
)

# Returns a list of all college football teams ESPN tracks.
# DEPRECATED — replaced by CFBD in college_data.py
ESPN_COLLEGE_TEAMS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/seasons/{year}/teams?limit=1000"
)

# Returns NFL team statistics.
# DEPRECATED — replaced by nfl_data_py in team_data.py
ESPN_NFL_TEAM_STATS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
    "/seasons/{year}/teams/{team_id}?enable=stats"
)

"""
COLLEGE FOOTBALL DATA API  (https://collegefootballdata.com/)
Free tier available — requires a free API key.
Sign up at https://collegefootballdata.com to get your key.
Set it as an environment variable: CFBD_API_KEY
"""

# Base URL for all CFBD API endpoints.
# We append specific paths like "/stats/player/season" to this.
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Timeout in seconds for CFBD API requests.
# CFBD can be slower than Sleeper so we allow a bit more time.
CFBD_REQUEST_TIMEOUT = 30

"""
CONSTANTS — shared settings used across multiple files
"""

# The most recent completed NFL regular season.
# Dynamically calculated based on the current date:
#   Offseason (March–August) → use last completed season (e.g. April 2026 → 2025)
#   In-season (September–February) → use current year  (e.g. October 2025 → 2025)
today = date.today()
year  = today.year
month = today.month

if 3 <= month <= 8:
    CURRENT_NFL_SEASON = year - 1
else:
    CURRENT_NFL_SEASON = year

# How many past NFL seasons of data to collect per player.
NFL_SEASONS_TO_COLLECT = 3   # collects: CURRENT, CURRENT-1, CURRENT-2

# Positions we care about for fantasy football.
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

# Default timeout (in seconds) for Sleeper API requests.
REQUEST_TIMEOUT_SECONDS = 10