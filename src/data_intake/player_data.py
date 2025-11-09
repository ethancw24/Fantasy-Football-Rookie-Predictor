import requests
import pandas as pd
import os

"""
Set the directory retrival to source out to data/raw
"""
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def get_rookies(year: int) -> pd.DataFrame:
    """
    Retrieve rookie player information from the Sleeper API.

    This function queries Sleeper's NFL player endpoint and extracts data for players
    who are rookies (college_exp == 0) and play in one of the key fantasy positions:
    Quarterback (QB), Running Back (RB), Wide Receiver (WR), or Tight End (TE).

    The resulting dataset includes each player's unique ID, full name, position,
    NFL team (if assigned), college name, and college experience level.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered rookie player data.

    Example:
        df = get_rookies()
        df.head()
        player_id       full_name        position   team    college    years_exp
        0   123456789   John Smith       WR         LAR     Alabama            0
        1   987654321   Mike Jones       RB         KC      Clemson            0
    """
    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError if request fails
    data = response.json()

    valid_positions = {"QB", "RB", "WR", "TE"}

    rookies = [
        {
            "player_id": pid,
            "full_name": player.get("full_name"),
            "position": player.get("position"),
            "team": player.get("team"),
            "college": player.get("college"),
            "years_exp": player.get("years_exp"),
        }
        for pid, player in data.items()
        if player.get("years_exp") == 0 and player.get("position") in valid_positions
    ]

    df = pd.DataFrame(rookies)
    output_path = os.path.join(DATA_DIR, "rookies_{year}.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rookies (QB, RB, WR, TE) to rookies_{year}.csv")
    return df

def get_team_offensive_stats(year: int) -> pd.DataFrame:
    """
    Retrieve NFL team offensive statistics from the Sleeper API for a given year.

    This function queries the Sleeper API's regular season statistics endpoint and
    extracts team-level offensive performance metrics. The retrieved data includes
    overall offense rankings as well as specific passing, rushing, and receiving
    category rankings, along with total touchdowns and red zone efficiency.

    Args:
        year (int): The NFL season year to retrieve statistics for (e.g., 2024).

    Returns:
        pd.DataFrame: A DataFrame containing team offensive statistics with the following columns:
            - team (str): Team abbreviation (e.g., "KC", "BUF")
            - year (int): Year of the stats
            - offense_rank (int): Overall offense rank
            - pass_yards_rank (int): Rank by passing yards
            - pass_td_rank (int): Rank by passing touchdowns
            - rush_yards_rank (int): Rank by rushing yards
            - rush_td_rank (int): Rank by rushing touchdowns
            - rec_yards_rank (int): Rank by receiving yards
            - rec_td_rank (int): Rank by receiving touchdowns
            - team_td (int): Total team touchdowns
            - redzone_rank (int): Team's rank in red zone performance
            - redzone_pct (float): Red zone touchdown percentage

    Example:
        df = get_team_offensive_stats(2024)
        df.head()
             team  year  offense_rank  pass_yards_rank  pass_td_rank  ...
        0     KC  2024             2               3               1
        1     BUF 2024             4               5               2
    """
    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{year}"
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError if request fails
    data = response.json()

    team_stats = [
        {
            "team": team,
            "year": year,
            "offense_rank": stats.get("offense_rank"),
            "pass_yards_rank": stats.get("pass_yds_rank"),
            "pass_td_rank": stats.get("pass_td_rank"),
            "rush_yards_rank": stats.get("rush_yds_rank"),
            "rush_td_rank": stats.get("rush_td_rank"),
            "rec_yards_rank": stats.get("rec_yds_rank"),
            "rec_td_rank": stats.get("rec_td_rank"),
            "team_td": stats.get("td"),
            "redzone_rank": stats.get("redzone_rank"),
            "redzone_pct": stats.get("redzone_pct"),
        }
        for team, stats in data.items()
        if isinstance(stats, dict)
    ]

    df = pd.DataFrame(team_stats)
    df.to_csv(os.path.join(DATA_DIR, f"sleeper_team_stats_{year}.csv"), index=False)
    print(f"Saved team offensive stats for {year}")
    return df