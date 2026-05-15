"""
main.py
=======

PURPOSE:
    Runs the Fantasy Football Rookie Predictor pipeline up to Stage 2.

STAGES CURRENTLY ACTIVE:
    Stage 1 — Data Intake    : fetch/load raw data       → data/raw/
    Stage 2 — Data Cleaning  : validate, clean, engineer → data/filtered/

HOW TO RUN:
    From your project root folder in the terminal:

        python main.py              # use cached CSVs if available
        python main.py --refresh    # force a fresh fetch from the APIs

WHEN TO USE --refresh:
    - A new NFL season has started and you want updated data
    - You deleted or modified a CSV and want to regenerate it
    - You changed something in data_intake and want to test fresh output

OUTPUT FILES:
    data/raw/nfl_players.csv        — NFL player stats by season (from Sleeper)
    data/raw/nfl_teams.csv          — NFL team context (from Sleeper)
    data/raw/combine.csv            — NFL Combine measurements (from ESPN)
    data/raw/college_stats.csv      — College football stats (from ESPN)
    data/raw/master.csv             — All four sources merged together

    data/filtered/clean_master.csv  — Validated, cleaned, and feature-engineered
                                      master table, ready for model training

COMMAND-LINE FLAGS:
    --refresh       Force a fresh API fetch even if cached CSVs exist
    --skip-cleaning Skip Stage 2 and just run data intake (useful for debugging)
"""

import sys
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES FROM .env
# ---------------------------------------------------------------------------
# This must happen BEFORE any other imports that need environment variables
# (specifically CFBD_API_KEY used by college_data.py).
#
# python-dotenv reads the .env file in your project root and loads each
# line as an environment variable — exactly as if you had set them in
# the terminal manually.
#
# IMPORTANT: .env is in your .gitignore and should never be committed
# to GitHub — it contains your private API key.

from dotenv import load_dotenv
load_dotenv()   # loads .env from the current working directory

# ---------------------------------------------------------------------------
# MAKE SURE PYTHON CAN FIND THE src FOLDER
# ---------------------------------------------------------------------------
# Path(__file__).parent gives us the project root (the folder main.py is in).
# We add the src/ subfolder to sys.path so Python can find our packages.
#
# WHAT IS sys.path?
#   A list of folders Python searches when you do "import something".
#   By default it doesn't include src/, so we add it manually here.

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# These imports only work after the sys.path line above.

from data_intake.loader import DataLoader
from data_intake.combine import DataCombiner   # merges the four separate DataFrames

from data_cleaning import DataValidator, DataCleaner, FeatureEngineer

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# Logging prints progress messages to your terminal as the pipeline runs.
# This is more useful than print() because:
#   - Every message is timestamped so you can see how long things take
#   - You can control the level of detail (INFO, DEBUG, WARNING, ERROR)
#   - Every message says which file it came from (e.g. "data_intake.loader")
#
# FORMAT EXPLAINED:
#   %(asctime)s   → timestamp:   "2024-04-10 14:32:01"
#   %(name)s      → source file: "data_intake.player_data"
#   %(levelname)s → severity:    "INFO" / "WARNING" / "ERROR"
#   %(message)s   → the actual message text

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WHERE TO SAVE THE CLEANED OUTPUT
# ---------------------------------------------------------------------------
# Path(__file__).parent = project root
# We save the cleaned master file to data/filtered/

_PROJECT_ROOT   = Path(__file__).parent
_FILTERED_DIR   = _PROJECT_ROOT / "data" / "filtered"


# ---------------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------------
# argparse lets us accept optional flags when running from the terminal.
#
# USAGE:
#   python main.py                  → normal run, use cached data if available
#   python main.py --refresh        → force fresh API fetch
#   python main.py --skip-cleaning  → run Stage 1 only (useful for testing)

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments.

    RETURNS:
        argparse.Namespace : an object where each argument is an attribute.
                             Access them like: args.refresh, args.skip_cleaning
    """
    parser = argparse.ArgumentParser(
        description="Fantasy Football Rookie Predictor — Stages 1 & 2.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--refresh",
        action  = "store_true",   # presence of flag sets this to True
        default = False,
        help    = (
            "Force a fresh fetch from the APIs, ignoring cached CSVs.\n"
            "Use at the start of a new NFL season or if data looks stale."
        ),
    )

    parser.add_argument(
        "--skip-cleaning",
        action  = "store_true",
        default = False,
        help    = (
            "Run Stage 1 (data intake) only, skipping Stage 2 (cleaning).\n"
            "Useful for debugging data intake without waiting for cleaning."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# STAGE 1 — DATA INTAKE
# ---------------------------------------------------------------------------

def run_data_intake(force_refresh: bool) -> "pd.DataFrame":
    """
    Loads all raw data and returns the merged master DataFrame.

    On the FIRST run (or when --refresh is passed), data is fetched
    from the Sleeper and ESPN APIs.  This takes several minutes.

    On LATER runs, cached CSVs in data/raw/ are loaded instantly.

    PARAMETERS:
        force_refresh (bool) : If True, always re-fetch from the APIs.

    RETURNS:
        pd.DataFrame : The merged master table (all four sources combined).
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — DATA INTAKE")
    logger.info("=" * 60)

    loader = DataLoader(force_refresh=force_refresh)
    data   = loader.load()

    # Print a summary table so you can see what was loaded at a glance.
    logger.info("Data intake complete. Summary:")
    for name, df in data.items():
        logger.info("  %-20s → %d rows × %d columns", name, len(df), len(df.columns))

    # The loader returns four SEPARATE DataFrames — one per data source.
    # We keep them separate through intake and cleaning to make debugging easier.
    # Now we merge them into one master table using DataCombiner.
    #
    # WHY MERGE HERE AND NOT IN THE LOADER?
    #   Merging messy pre-cleaned data is risky — join keys may be inconsistent.
    #   Keeping them separate through intake lets us validate each source first.
    #   DataCombiner uses LEFT joins so no player is ever dropped due to missing
    #   combine or college data.
    logger.info("\nMerging four DataFrames into master table…")
    combiner  = DataCombiner()
    master_df = combiner.merge_all(
        nfl_df     = data["nfl_players"],
        team_df    = data["teams"],
        combine_df = data["combine"],
        college_df = data["college"],
    )

    logger.info(
        "\nMaster DataFrame: %d rows × %d columns — passing to Stage 2.\n",
        len(master_df), len(master_df.columns),
    )
    return master_df


# ---------------------------------------------------------------------------
# STAGE 2 — DATA CLEANING
# ---------------------------------------------------------------------------

def run_data_cleaning(master_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Validates, cleans, and engineers features on the master DataFrame.

    STEP A — Validate:
        Checks for empty data, missing columns, duplicates, impossible
        stat values, and more.  Raises a ValueError immediately if
        anything is wrong — the pipeline stops so you can fix it.

    STEP B — Clean:
        Fixes college season alignment using draft_year, handles UDFAs,
        flags rows needing manual review, normalises player names,
        and drops the standard scoring column.

    STEP C — Engineer features:
        Adds per-game stats, rookie flag, college-to-NFL ratios, and
        position-average comparison columns.

    STEP D — Save:
        Writes the final DataFrame to data/filtered/clean_master.csv.

    PARAMETERS:
        master_df (pd.DataFrame) : Output of run_data_intake().

    RETURNS:
        pd.DataFrame : Fully cleaned and feature-engineered DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 — DATA CLEANING")
    logger.info("=" * 60)

    # ── Step A: Validate ──────────────────────────────────────────────────
    logger.info("Step A: Validating master DataFrame…")
    validator = DataValidator()
    validator.validate(master_df)   # raises ValueError and stops if anything's wrong
    logger.info("Validation passed.\n")

    # ── Step B: Clean ─────────────────────────────────────────────────────
    logger.info("Step B: Cleaning data…")
    cleaner  = DataCleaner()
    clean_df = cleaner.clean(master_df)

    # Show how many rows needed manual review after cleaning
    if "college_season_needs_review" in clean_df.columns:
        review_count = clean_df["college_season_needs_review"].sum()
        if review_count > 0:
            logger.warning(
                "%d rows flagged for manual review (college season unknown).\n"
                "  Run this to see them:\n"
                "  df[df['college_season_needs_review'] == True]"
                "[['full_name', 'draft_year', 'years_exp']]",
                review_count,
            )
    logger.info("Cleaning complete.\n")

    # ── Step C: Engineer features ─────────────────────────────────────────
    logger.info("Step C: Engineering features…")
    engineer = FeatureEngineer()
    final_df = engineer.engineer(clean_df)
    logger.info("Feature engineering complete.\n")

    # ── Step D: Save to data/filtered/ ────────────────────────────────────
    logger.info("Step D: Saving cleaned data…")
    _FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = _FILTERED_DIR / "clean_master.csv"
    final_df.to_csv(output_path, index=False)

    # index=False: don't write the row numbers (0, 1, 2…) as a column.
    # They're just internal pandas bookkeeping, not real data.
    logger.info("  Saved → %s  (%d rows × %d columns)", output_path, len(final_df), len(final_df.columns))

    return final_df


# ---------------------------------------------------------------------------
# MAIN — ties all stages together
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point.  Parses arguments and runs each active stage in order,
    passing output from one stage as input to the next.
    """
    args = parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║      Fantasy Football Rookie Predictor — Starting        ║")
    logger.info("╚══════════════════════════════════════════════════════════╝\n")

    if args.refresh:
        logger.info("--refresh flag detected: will force a fresh API fetch.\n")
    if args.skip_cleaning:
        logger.info("--skip-cleaning flag detected: Stage 2 will be skipped.\n")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    master_df = run_data_intake(force_refresh=args.refresh)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    if not args.skip_cleaning:
        final_df = run_data_cleaning(master_df)
    else:
        logger.info("Skipping Stage 2 as requested.")
        final_df = master_df

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                  Pipeline complete!                      ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info("║  Rows in final DataFrame  : %-28s ║", len(final_df))
    logger.info("║  Columns in final DataFrame: %-27s ║", len(final_df.columns))
    logger.info("║  Output saved to          : data/filtered/               ║")
    logger.info("║                                                          ║")
    logger.info("║  Next step: Stage 3 — Model Training                     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")


# ---------------------------------------------------------------------------
# ENTRY POINT GUARD
# ---------------------------------------------------------------------------
# This block only runs when you execute `python main.py` directly.
# It does NOT run if another file imports something from main.py.
# This is standard Python practice for any file that acts as a script.

if __name__ == "__main__":
    main()