import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import fire
import pandas as pd
import scrapbook as sb
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_glue_values_from_notebooks(output_dir: Path) -> Dict[str, List[Any]]:
    """Extract all glue values from output notebooks.

    Args:
        output_dir: Directory containing output notebooks

    Returns:
        Dictionary mapping glue keys to lists of values
    """
    glue_values = defaultdict(list)

    # Find all notebook files
    notebook_files = sorted(output_dir.glob("*.ipynb"))

    for notebook_file in tqdm(notebook_files):
        # Read the notebook using scrapbook
        nb = sb.read_notebook(str(notebook_file))

        # Extract all glued data
        for key, scrap in nb.scraps.items():
            value = scrap.data
            glue_values[key].append(value)

    return dict(glue_values)


def generate_statistics(glue_values: Dict[str, List[Any]], output_dir: Path) -> None:
    """Generate statistics for each glue value and save to CSV.

    Args:
        glue_values: Dictionary mapping glue keys to lists of values
        output_dir: Directory where CSV files will be saved
    """
    stats_rows = []

    for key, values in glue_values.items():
        # Convert to pandas Series
        series = pd.Series(values, name=key)

        # Get descriptive statistics
        stats = series.describe()

        # Add the glue key as a column
        stats_dict = stats.to_dict()
        stats_dict["glue_key"] = key
        stats_rows.append(stats_dict)

    # Create DataFrame with glue_key as first column
    stats_df = pd.DataFrame(stats_rows)
    cols = ["glue_key"] + [col for col in stats_df.columns if col != "glue_key"]
    stats_df = stats_df[cols]

    # Sort by glue_key
    stats_df = stats_df.sort_values("glue_key")

    # Save to single CSV (4 significant figures for readability)
    csv_filename = output_dir / "statistics.csv"
    stats_df.to_csv(csv_filename, index=False, float_format="%.4g")

    logger.info(
        f"Generated statistics for {len(stats_rows)} glue values -> {csv_filename.name}"
    )


def main(
    output_dir: str = "output",
    log_level: str = "INFO",
) -> None:
    """
    Extract glue values from notebooks and generate statistics.

    This script reads all notebook files in the output directory, extracts
    glued values using scrapbook, and generates descriptive statistics
    saved to a CSV file.

    Args:
        output_dir: Directory containing output notebooks
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR")
    """
    setup_logging(log_level)
    logger.info("Starting statistics generation")

    # Extract glue values and generate statistics
    output_path = Path(output_dir)
    logger.info("Extracting glue values from output notebooks")
    glue_values = extract_glue_values_from_notebooks(output_path)
    logger.info(f"Found {len(glue_values)} unique glue keys")

    logger.info("Generating statistics and saving to CSV files")
    generate_statistics(glue_values, output_path)
    logger.info("Statistics generation completed")


if __name__ == "__main__":
    fire.Fire(main)
