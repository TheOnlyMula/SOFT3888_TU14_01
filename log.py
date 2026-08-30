from pathlib import Path
import yaml
import logging
import sys

def create_log_directory() -> Path: 
    log_directory = Path("runs")
    log_directory.mkdir(exist_ok = True)

    experiment_number = 1 

    while (log_directory / f"exp_{experiment_number:03d}").exists(): 
        experiment_number += 1

    log_directory = log_directory / f"exp_{experiment_number:03d}"
    log_directory.mkdir()

    return log_directory

def log_model_config(config: dict, log_directory: Path, logger: logging.Logger):
    width = max(len(setting) for setting in config)\
    
    logger.info("Model Configuration: ")
    for setting, configuration in config.items(): 
        logger.info(f"{setting:<{width}} : {configuration}")

    with open(log_directory / "config.yaml", "w") as file: 
        yaml.dump(config, file)

def log_command(argv: str, log_directory: Path):
    command = " ".join(argv)

    with open(log_directory / "run_command.txt", "w") as file: 
        file.write(command)

def create_logger(name: str, log_directory: Path) -> logging.Logger: 
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers: 
        logger.handlers.clear() 

    # 2026-08-30 09:05:38 INFO     Selected Device    : cuda
    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt = log_format, datefmt = date_format)

    file_handler = logging.FileHandler(
        log_directory / "train.log", 
        mode = "a"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger