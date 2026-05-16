"""
APIs and data sources used to intake team rankings, NFL stats,
college stats, and combine stats.
 
--------------------------
Sources (current)
--------------------------
- Sleeper API       : NFL player stats (player_data.py)
- nfl_data_py       : NFL Combine measurements and team schedules
                      (combine.py, team_data.py)
- CFBD API          : College football stats — requires free API key
                      from https://collegefootballdata.com
                      Set as environment variable: CFBD_API_KEY
                      (college_data.py)
 
--------------------------
Previously used (now deprecated)
--------------------------
- ESPN hidden API   : Returned 403/404 errors — replaced by sources above
"""
 
from .player_data import SleeperPlayerData
from .team_data   import SleeperTeamData
from .combine     import ESPNCombineData, DataCombiner
from .college_data import ESPNCollegeData
from .loader      import DataLoader
 
__all__ = [
    "SleeperPlayerData",
    "SleeperTeamData",
    "ESPNCombineData",
    "DataCombiner",
    "ESPNCollegeData",
    "DataLoader",
]