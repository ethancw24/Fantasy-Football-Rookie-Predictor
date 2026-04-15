"""
data_cleaning/__init__.py
=========================

PURPOSE:
    This is the "front door" of the data_cleaning package.  It works exactly
    like data_intake/__init__.py — Python looks here first when another part
    of the project does:
        from src.data_cleaning import DataValidator

HOW THE FILES IN THIS FOLDER CONNECT:
    validators.py       →  Checks the master DataFrame for problems BEFORE
                           any cleaning happens.  Raises an error and stops
                           everything if something is wrong.

    cleaner.py          →  Fixes known issues in the data:
                               - Fills in last_college_season using draft_year
                               - Flags undrafted players (UDFAs) correctly
                               - Flags rows where draft_year was NaN (for
                                 manual review)
                               - Drops the non-target scoring column
                                 (standard scoring — we keep half_ppr + ppr)
                               - Normalises player name formatting

    feature_engineer.py →  Creates NEW columns derived from existing ones:
                               - Per-game stats (yards/game, TDs/game, etc.)
                               - Passer rating vs. position average
                               - College-to-NFL stat comparison ratios
                               - Rookie flag

    This is the recommended order to run them:
        raw master df  →  validate  →  clean  →  engineer features  →  model-ready df

RECOMMENDED USAGE:
    from src.data_cleaning import DataValidator, DataCleaner, FeatureEngineer

    validator = DataValidator()
    validator.validate(master_df)       # raises ValueError if something's wrong

    cleaner   = DataCleaner()
    clean_df  = cleaner.clean(master_df)

    engineer  = FeatureEngineer()
    final_df  = engineer.engineer(clean_df)
"""

from .validators        import DataValidator
from .cleaner           import DataCleaner
from .feature_engineer  import FeatureEngineer

__all__ = [
    "DataValidator",    # Step 1 — check for problems, raise if found
    "DataCleaner",      # Step 2 — fix known data issues
    "FeatureEngineer",  # Step 3 — create new model-ready columns
]