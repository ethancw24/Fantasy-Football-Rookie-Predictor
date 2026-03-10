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
from .college_stats import ESPNCollegeData

__all__ = [
    "SleeperPlayerData",
    "SleeperTeamData",
    "ESPNCombineData",
    "ESPNCollegeData",
]