"""
APIs used to intake team rankings, NFL stats, college stats, and combine stats

--------------------------
Sources
--------------------------
- Sleeper API: NFL player stats and team rankings information
- ESPN API: College stats and combine stats (note with combine stats not everyone participates so use outer joins and use this to fine tune)
"""

from .player_data import SleeperPlayerData
from .team_data import SleeperTeamData
from .combine import ESPNCombineData
from .college_data import ESPNCollegeData
from .loader       import DataLoader          # Move data to raw so the cleaning process can access it

__all__ = [
    "SleeperPlayerData",
    "SleeperTeamData",
    "ESPNCombineData",
    "ESPNCollegeData",
    "DataLoader",
]