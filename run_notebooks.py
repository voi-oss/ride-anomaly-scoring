import logging
import os
import uuid
from pathlib import Path
from typing import Any, List

import fire
import papermill as pm
from joblib import Parallel, delayed
from jupyter_client.manager import KernelManager
from jupyter_core.paths import jupyter_runtime_dir


class IPCKernelManager(KernelManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kernel_id = str(uuid.uuid4())
        os.makedirs(jupyter_runtime_dir(), exist_ok=True)
        connection_file = os.path.join(
            jupyter_runtime_dir(), f"kernel-{kernel_id}.json"
        )
        super().__init__(
            *args,
            transport="ipc",
            kernel_id=kernel_id,
            connection_file=connection_file,
            **kwargs,
        )


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


def run_notebook_with_seed(
    notebook_path: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """Run a single notebook with a specific seed using papermill.

    Args:
        notebook_path: Path to the input notebook.
        output_dir: Directory where the output notebook will be saved.
        seed: Seed value to pass as a parameter to the notebook.
    """
    output_path = output_dir / f"{notebook_path.stem}_seed_{seed}{notebook_path.suffix}"
    if output_path.exists():
        logger.info(
            f"Skipping {notebook_path.name} with seed={seed}: "
            f"output already exists at {output_path}"
        )
        return
    try:
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            parameters={"seed": seed},
            cwd=str(notebook_path.parent),
            progress_bar=False,
            kernel_manager_class="run_notebooks.IPCKernelManager",
        )
    except Exception as e:
        logger.error(f"Failed to execute {notebook_path.name} with seed={seed}: {e}")
        raise


def run_notebooks(
    notebooks: List[str],
    n_runs: int,
    output_dir: str,
    n_jobs: int = -1,
) -> None:
    """Run multiple notebooks N times each with different seed values.

    Args:
        notebooks: List of notebook paths to run.
        n_runs: Number of times to run each notebook.
        output_dir: Directory where output notebooks will be saved.
        n_jobs: Number of parallel jobs (-1 for all CPUs).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    total_runs = len(notebooks) * n_runs
    logger.info(f"Running {len(notebooks)} notebook(s) {n_runs} time(s) each")
    logger.info(f"Total executions: {total_runs}")

    # Prepare all tasks
    tasks = []
    for notebook in notebooks:
        notebook_path = Path(notebook)
        if not notebook_path.exists():
            logger.error(f"Notebook not found: {notebook_path}")
            continue
        for seed in range(1, n_runs + 1):
            tasks.append((notebook_path, output_path, seed))

    # Execute in parallel with tqdm progress bar
    Parallel(n_jobs=n_jobs)(
        delayed(run_notebook_with_seed)(nb_path, out_path, seed)
        for nb_path, out_path, seed in tasks
    )


def main(
    *notebooks: str,
    n_runs: int = 1,
    output_dir: str = "output",
    log_level: str = "INFO",
    n_jobs: int = -1,
) -> None:
    """
    Run notebooks multiple times with different seed values using papermill.

    This script takes a list of notebooks and runs each of them N times,
    setting the seed parameter to 1, 2, 3, ..., N for each run. The rendered
    notebooks are saved to the specified output directory with the seed value
    appended to the filename.

    Args:
        notebooks: List of notebook paths to run
        n_runs: Number of times to run each notebook
        output_dir: Directory where output notebooks will be saved
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR")
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
    """
    setup_logging(log_level)
    logger.info("Starting notebook execution pipeline")

    # Convert to list
    notebooks_list = list(notebooks)

    logger.info(f"Notebooks to run: {notebooks_list}")

    # Run notebooks in parallel
    run_notebooks(notebooks_list, n_runs, output_dir, n_jobs)

    logger.info("Notebook execution pipeline completed successfully")


if __name__ == "__main__":
    fire.Fire(main)
