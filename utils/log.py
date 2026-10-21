import csv
from pathlib import Path
import yaml
import logging
import sys
from ultralytics.utils.plotting import plot_results

def create_log_directory(role: str = "teacher") -> Path: 
    """
    Creates a numbered experiment directory inside either 'runs_teacher' or 'runs_student'.
    
    Args:
        role (str): 'teacher' or 'student'.
    """
    log_directory = Path("runs") / role
    log_directory.mkdir(parents=True, exist_ok=True)

    experiment_number = 1

    while (log_directory / f"exp_{experiment_number:03d}").exists():
        experiment_number += 1

    exp_directory = log_directory / f"exp_{experiment_number:03d}"
    exp_directory.mkdir(parents=True, exist_ok=True)

    return exp_directory

def log_model_config(config: dict, log_directory: Path, logger: logging.Logger):
    width = max(len(setting) for setting in config)
    
    logger.info("Model Configuration: ")
    for setting, configuration in config.items(): 
        logger.info(f"{setting:<{width}} : {configuration}")

    with open(log_directory / "config.yaml", "w") as file: 
        yaml.dump(config, file)

def load_model_config(log_directory: Path) -> dict:
    """The settings a run recorded in its config.yaml."""
    with open(Path(log_directory) / "config.yaml") as file:
        return yaml.safe_load(file)

def log_epoch_results(row: dict, log_directory: Path, logger: logging.Logger) -> None:
    """Add one epoch to results.csv and redraw the training curves as results.png.

    The plot is Ultralytics' plot_results, which draws every column with "loss" or "metric" in its
    name. A resumed run repeats the interrupted epoch, so rows from that epoch onwards are replaced.
    The plot is only a convenience, so failing to save it (Windows refuses while results.png is
    open in a viewer) is logged and training continues; the next epoch redraws it.
    """
    results_file = Path(log_directory) / "results.csv"
    rows = []
    if results_file.is_file():
        with open(results_file, newline="") as file:
            rows = [old for old in csv.DictReader(file) if int(old["epoch"]) < row["epoch"]]
    rows.append(row)
    with open(results_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    try:
        plot_results(file=str(results_file))
    except OSError as error:
        logger.warning("Could not redraw %s (%s); results.csv is up to date", results_file.with_suffix(".png"), error)

def log_quality_weights(rows: list[dict], log_directory: Path) -> None:
    """Append one epoch's per-image event-quality rows to quality_weights.csv."""
    weights_file = Path(log_directory) / "quality_weights.csv"
    new_file = not weights_file.is_file()
    with open(weights_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

def log_command(argv: list[str], log_directory: Path):
    command = " ".join(argv)

    with open(log_directory / "run_command.txt", "w") as file:
        file.write(command)

    with open(log_directory / "run_args.yaml", "w") as file:
        yaml.safe_dump([str(arg) for arg in argv], file)

def load_command(log_directory: Path) -> list[str]:
    """The command a run was started with, as an argv list (script first)."""
    with open(Path(log_directory) / "run_args.yaml") as file:
        return yaml.safe_load(file)

def create_logger(
    name: str,
    log_directory: Path,
    console_level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]: 
        logger.removeHandler(handler)
        handler.close()

    log_format = "%(asctime)s %(levelname)-5s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    file_handler = logging.FileHandler(
        log_directory / "train.log", 
        mode = "a", 
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger