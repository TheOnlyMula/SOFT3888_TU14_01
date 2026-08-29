from pathlib import Path
import yaml
import logging

def create_log_directory() -> Path: 
    log_directory = Path("runs")
    log_directory.mkdir(exist_ok = True)

    experiment_number = 1 

    while (log_directory / f"exp_{experiment_number:03d}").exists(): 
        experiment_number += 1

    log_directory = log_directory / f"exp_{experiment_number:03d}"
    log_directory.mkdir()

    return log_directory

def log_model_config(config: dict, log_directory: Path):
    width = max(len(setting) for setting in config)
    for setting, configuration in config.items(): 
        print(f"{setting:<{width}} : {configuration}")

    with open(f"{log_directory}/config.yaml", "w") as file: 
        yaml.dump(config, file)

def log_command(argv: str, log_directory: Path):
    command = " ".join(argv)
    with open(f"{log_directory}/run_command.txt", "w") as file: 
        file.write(command)