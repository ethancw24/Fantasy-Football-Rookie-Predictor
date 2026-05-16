"""
data_intake/loader.py
=====================
 
PURPOSE:
    The single entry point for ALL raw data in this project.
    Instead of calling SleeperPlayerData, SleeperTeamData, etc. directly,
    every other part of the project calls DataLoader and lets it decide
    whether to fetch fresh data from the APIs or load from a saved CSV.
 
THE CACHING PATTERN — WHY THIS EXISTS:
    Fetching data from APIs is SLOW (hundreds of HTTP requests) and fragile
    (the API could be down, rate-limit you, or change its format).
    After the first successful fetch we save everything to CSV files on disk.
    Every run after that loads from those CSVs in seconds instead of minutes.
 
    First run  :  APIs → DataFrames → saved to data/raw/*.csv  (slow, ~minutes)
    Later runs :  data/raw/*.csv → DataFrames                   (fast, ~seconds)
 
WHAT IS A CSV?
    A CSV (Comma-Separated Values) file is the simplest way to store a table.
    Each row in the DataFrame becomes a line in the file, and each column is
    separated by a comma.  You can open CSVs in Excel, Google Sheets, or any
    text editor to inspect the data.
 
    Example (first 2 rows of nfl_players.csv):
        player_id,full_name,position,season,pass_yards,rush_yards,...
        4046,Patrick Mahomes,QB,2023,4183,358,...
        2749,Justin Jefferson,WR,2023,0,0,...
 
WHEN DOES THE CACHE GET REFRESHED?
    The cache is refreshed automatically if:
        1. Any CSV file is missing (first run, or someone deleted the files)
        2. The saved data is from a previous NFL season (outdated)
        3. You call DataLoader(force_refresh=True) to force a fresh fetch
 
HOW TO USE:
    # Simple — let the loader decide what to do
    from src.data_intake.loader import DataLoader
 
    loader = DataLoader()
    data   = loader.load()
 
    # Access individual DataFrames — each source stays separate.
    # Merging and NaN handling happens later in data_cleaning.
    print(data["nfl_players"].head())
    print(data["teams"].head())
    print(data["combine"].head())
    print(data["college"].head())
 
    # Force a fresh API fetch even if CSVs already exist
    loader = DataLoader(force_refresh=True)
    data   = loader.load()
"""
 
import logging
from pathlib import Path      # New way of handling file direction the OS, if I start messing up see if changing back to OS could fix it
import pandas as pd
from .links import (
    CURRENT_NFL_SEASON,
    NFL_SEASONS_TO_COLLECT,
)
from .player_data  import SleeperPlayerData
from .team_data    import SleeperTeamData
from .combine      import ESPNCombineData
from .college_data import ESPNCollegeData
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# FILE PATH CONFIGURATION
# ---------------------------------------------------------------------------
# Path(__file__) is the absolute path to THIS file (loader.py).
# We walk up the directory tree with .parent calls to find the project root,
# then point to the data/raw folder from there.
#
# Directory structure assumed:
#   project_root/
#       data/
#           raw/          ← CSVs are saved here
#       src/
#           data_intake/
#               loader.py ← this file
#
# .parent        → data_intake/
# .parent.parent → src/
# .parent.parent.parent → project_root/
 
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
 
# The expected CSV file for each dataset.
# These are the four files that will appear in your data/raw/ folder.
# NOTE: There is no master/merged CSV here — merging happens in data_cleaning
#       after NaN values have been handled for each individual dataset first.
CSV_PATHS = {
    "nfl_players" : _RAW_DATA_DIR / "nfl_players.csv",
    "teams"       : _RAW_DATA_DIR / "nfl_teams.csv",
    "combine"     : _RAW_DATA_DIR / "combine.csv",
    "college"     : _RAW_DATA_DIR / "college_stats.csv",
}
 
# A metadata file that records which NFL season the cached data is from.
# We use this to detect when the season has rolled over and the cache is stale.
_CACHE_META_PATH = _RAW_DATA_DIR / ".cache_meta"
 
 
class DataLoader:
    """
    Loads all project data — from cache if available, from APIs if not.
 
    PARAMETERS:
        force_refresh (bool) :
            If True, always re-fetch from APIs and overwrite the CSVs.
            Default is False (use cache when available).
        seasons (list[int] | None) :
            Which NFL seasons to collect for players and teams.
            Defaults to the last NFL_SEASONS_TO_COLLECT seasons.
        combine_years (list[int] | None) :
            Which draft years to collect combine data for.
            Defaults to the same window as seasons.
        college_years (list[int] | None) :
            Which college football seasons to collect.
            Defaults to one year before each NFL season (college → draft → NFL).
 
    USAGE:
        loader = DataLoader()
        data   = loader.load()
        nfl_df = data["nfl_players"]   # use individual DataFrames separately
    """
 
    def __init__(
        self,
        force_refresh  : bool            = False,
        seasons        : list[int] | None = None,
        combine_years  : list[int] | None = None,
        college_years  : list[int] | None = None,
    ):
        self.force_refresh = force_refresh
 
        # Build default season lists if none were provided.
        # e.g. if CURRENT_NFL_SEASON=2024 and NFL_SEASONS_TO_COLLECT=3:
        #      seasons       = [2024, 2023, 2022]
        #      combine_years = [2024, 2023, 2022]  (draft classes)
        #      college_years = [2023, 2022, 2021]  (one year before draft)
        self.seasons = seasons or [
            CURRENT_NFL_SEASON - i for i in range(NFL_SEASONS_TO_COLLECT)
        ]
        self.combine_years = combine_years or self.seasons
        self.college_years = college_years or [y - 1 for y in self.seasons]
 
        # Make sure the data/raw directory exists.
        # exist_ok=True means "don't crash if the folder already exists."
        _RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialized. RAW_DATA_DIR = %s", _RAW_DATA_DIR)
 
    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------
 
    def load(self) -> dict[str, pd.DataFrame]:
        """
        Main entry point.  Returns a dictionary of DataFrames.
 
        DECISION LOGIC:
            1. If force_refresh=True          → always fetch from APIs
            2. If any CSV is missing          → fetch from APIs
            3. If cached season is outdated   → fetch from APIs
            4. Otherwise                      → load from CSVs (fast path)
 
        RETURNS:
            dict with keys: "nfl_players", "teams", "combine", "college"
            Each value is a pandas DataFrame.
            Merging these together happens later in data_cleaning.
        """
        if self.force_refresh:
            logger.info("force_refresh=True — fetching fresh data from APIs.")
            return self._fetch_and_save()
 
        if not self._cache_is_valid():
            logger.info("Cache missing or outdated — fetching from APIs.")
            return self._fetch_and_save()
 
        logger.info("Valid cache found — loading from CSV files.")
        return self._load_from_cache()
 
    # -----------------------------------------------------------------------
    # CACHE VALIDATION
    # -----------------------------------------------------------------------
 
    def _cache_is_valid(self) -> bool:
        """
        Returns True only if ALL CSVs exist AND the cached season matches
        the current NFL season.
 
        WHAT IS THE METADATA FILE?
            _CACHE_META_PATH is a tiny text file we write alongside the CSVs.
            It contains just one number — the NFL season year the data is from.
            On every load we check if that year still matches CURRENT_NFL_SEASON.
            If it doesn't (a new season started), we treat the cache as stale.
        """
        # Check 1 — do all expected CSV files exist?
        for name, path in CSV_PATHS.items():
            if not path.exists():
                logger.info("Cache miss: '%s' not found at %s", name, path)
                return False
 
        # Check 2 — does the metadata file exist and match the current season?
        if not _CACHE_META_PATH.exists():
            logger.info("Cache miss: metadata file not found.")
            return False
 
        cached_season = self._read_cached_season()
        if cached_season != CURRENT_NFL_SEASON:
            logger.info(
                "Cache stale: cached season %s ≠ current season %s.",
                cached_season, CURRENT_NFL_SEASON,
            )
            return False
 
        logger.info("Cache valid for season %d.", CURRENT_NFL_SEASON)
        return True
 
    def _read_cached_season(self) -> int | None:
        """
        Reads the season year from the metadata file.
        Returns None if the file can't be read.
        """
        try:
            return int(_CACHE_META_PATH.read_text().strip())
        except (ValueError, OSError):
            return None
 
    def _write_cache_meta(self) -> None:
        """
        Writes the current NFL season year to the metadata file.
        Called after a successful API fetch so future runs know
        which season the cached data belongs to.
        """
        _CACHE_META_PATH.write_text(str(CURRENT_NFL_SEASON))
        logger.info("Cache metadata written: season %d.", CURRENT_NFL_SEASON)
 
    # -----------------------------------------------------------------------
    # FAST PATH — load from saved CSVs
    # -----------------------------------------------------------------------
 
    def _load_from_cache(self) -> dict[str, pd.DataFrame]:
        """
        Reads all CSV files from data/raw/ and returns them as DataFrames.
 
        WHY NOT JUST pd.read_csv() EVERYWHERE?
            Centralising the load here means if we ever change the CSV format
            (e.g. add compression, change dtypes) we only update one place.
        """
        logger.info("Loading cached DataFrames from %s …", _RAW_DATA_DIR)
        data = {}
 
        for name, path in CSV_PATHS.items():
            logger.info("  Reading %s …", path.name)
            data[name] = pd.read_csv(path)
 
        logger.info("All DataFrames loaded from cache.")
        return data
 
    # -----------------------------------------------------------------------
    # SLOW PATH — fetch from APIs and save to CSVs
    # -----------------------------------------------------------------------
 
    def _fetch_and_save(self) -> dict[str, pd.DataFrame]:
        """
        Runs the full API fetch pipeline, saves each DataFrame to its own CSV,
        and returns the dictionary of DataFrames.
 
        Each dataset is saved separately so data_cleaning can load them
        individually, handle their NaN values appropriately for each source,
        and then merge them once the data is clean.
 
        STEPS:
            1. Fetch NFL player stats     (SleeperPlayerData)  → nfl_players.csv
            2. Fetch NFL team stats       (SleeperTeamData)    → nfl_teams.csv
            3. Fetch combine measurements (ESPNCombineData)    → combine.csv
            4. Fetch college stats        (ESPNCollegeData)    → college_stats.csv
            5. Write cache metadata
        """
        logger.info("=== Starting full API data fetch ===")
 
        # ── Step 1: NFL player stats ──────────────────────────────────────
        logger.info("Step 1/4 — Fetching NFL player stats…")
        nfl_df = SleeperPlayerData().get_player_dataframe()
        self._save_csv(nfl_df, CSV_PATHS["nfl_players"], "nfl_players")
 
        # ── Step 2: NFL team stats ────────────────────────────────────────
        logger.info("Step 2/4 — Fetching NFL team stats…")
        team_df = SleeperTeamData().get_team_dataframe(seasons=self.seasons)
        self._save_csv(team_df, CSV_PATHS["teams"], "teams")
 
        # ── Step 3: Combine measurements ─────────────────────────────────
        logger.info("Step 3/4 — Fetching NFL Combine measurements…")
        combine_df = ESPNCombineData().get_combine_dataframe(years=self.combine_years)
        self._save_csv(combine_df, CSV_PATHS["combine"], "combine")
 
        # ── Step 4: College stats ─────────────────────────────────────────
        logger.info("Step 4/4 — Fetching college football stats…")
        college_df = ESPNCollegeData().get_college_dataframe(years=self.college_years)
        self._save_csv(college_df, CSV_PATHS["college"], "college")
 
        # ── Step 5: Write metadata so future runs know the cache is fresh ─
        self._write_cache_meta()
 
        logger.info(
            "=== Fetch complete. 4 CSVs saved to %s ===\n"
            "    Next step: data_cleaning will handle NaNs and merging.",
            _RAW_DATA_DIR,
        )
 
        return {
            "nfl_players" : nfl_df,
            "teams"       : team_df,
            "combine"     : combine_df,
            "college"     : college_df,
        }
 
    def _save_csv(self, df: pd.DataFrame, path: Path, name: str) -> None:
        """
        Saves one DataFrame to a CSV file.
 
        PARAMETERS:
            df   (pd.DataFrame) : The DataFrame to save.
            path (Path)         : The full file path to write to.
            name (str)          : Human-readable name for log messages.
 
        WHY index=False?
            pandas DataFrames have a row index (0, 1, 2, …) that is not
            part of our actual data.  index=False tells pandas NOT to write
            that index column into the CSV — otherwise you'd get an ugly
            unnamed first column when you reload it.
        """
        if df.empty:
            logger.warning("'%s' DataFrame is empty — saving empty CSV anyway.", name)
 
        df.to_csv(path, index=False)
        logger.info("  Saved %s → %s (%d rows)", name, path.name, len(df))