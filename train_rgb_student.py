import argparse
import log
import sys
import logging
import math
from pathlib import Path
import torch
from types import SimpleNamespace
from ultralytics import YOLO
from torch.utils.data import DataLoader
from ultralytics import settings
from evaluate import (
    _prepare_yolo_batch,
    evaluate_map_model,
    evaluate_test_model,
    validate_model,
)
from dataset_loader import create_rgb_dataloaders, create_rgb_test_loader
from utils import (
    select_device, 
    extract_trainable_model, 
    create_optimiser, 
    load_checkpoint, 
    fetch_best_dict
)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger, 
    total_epochs: int,
    checkpoint_path: Path,
    latest_path: Path
) -> Path:
    """Train the student and save the weights with the lowest validation loss."""
    if len(train_loader) == 0 or len(val_loader) == 0:
        message = "Training and validation loaders must both be non-empty."
        logger.error(message)
        raise ValueError(message)
    
    checkpoint_path = Path(checkpoint_path)
    latest_path = Path(latest_path)
    
    # Checkpoint logic from the weight-saving branch
    checkpoint_info = load_checkpoint(latest_path, device, model, optimiser)
    past_epochs = checkpoint_info["past_epochs"]
    
    if past_epochs >= total_epochs:
        logger.info(
            "Training skipped completed_epochs = %d target_epochs = %d",
            past_epochs,
            total_epochs,
        )
        return checkpoint_path
        
    # Get the historical best stats so we only overwrite if we beat the previous best
    best_stats = fetch_best_dict(checkpoint_path, device, model, optimiser, mode = 0)
    best_val_loss = best_stats["best_val_loss"]

    step_width = len(str(len(train_loader)))
    epoch_width = len(str(total_epochs))

    for epoch in range(past_epochs, total_epochs):
        
        logger.info(
            "Epoch started epoch = %d/%d",
            epoch + 1,
            total_epochs,
        )

        model.train()
        total_loss = 0.0
        total_train_box_loss = 0.0
        total_train_cls_loss = 0.0
        total_train_l1_loss = 0.0
        
        for batch_idx_step, (imgs, targets) in enumerate(train_loader):

            batch_data = _prepare_yolo_batch(imgs, targets, device)
            optimiser.zero_grad()
            
            loss, loss_items = model(batch_data)
            batch_loss = loss.sum()
            batch_loss.backward()
            
            # Prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimiser.step()

            step_loss = batch_loss.item()
            
            step_box_loss = loss_items['box_loss']
            step_cls_loss = loss_items['cls_loss']
            step_l1_loss = loss_items['l1_loss']
            
            step_box_loss = float(loss_items['box_loss'])
            step_cls_loss = float(loss_items['cls_loss'])
            step_l1_loss = float(loss_items['l1_loss'])

            total_loss += step_loss
            total_train_box_loss += step_box_loss
            total_train_cls_loss += step_cls_loss
            total_train_l1_loss += step_l1_loss

            step = batch_idx_step + 1
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                logger.debug(
                    "Batch completed epoch = %d/ %d step = %d/ %d "
                    "loss = %.4f box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
                    epoch + 1,
                    total_epochs,
                    step,
                    len(train_loader),
                    step_loss,
                    step_box_loss,
                    step_cls_loss,
                    step_l1_loss,
                )

        num_train_batches = len(train_loader)
        avg_train_batch_loss = total_loss / num_train_batches
        avg_train_image_loss = avg_train_batch_loss / train_loader.batch_size
        
        avg_train_box_loss = total_train_box_loss / num_train_batches
        avg_train_cls_loss = total_train_cls_loss / num_train_batches
        avg_train_l1_loss = total_train_l1_loss / num_train_batches
        
        (avg_val_batch_loss, 
         avg_val_box_loss, 
         avg_val_cls_loss, 
         avg_val_l1_loss) = validate_model(val_loader, model, device, logger)
        
        if not math.isfinite(avg_val_batch_loss):
            message = "Non-finite validation loss; cannot select a best checkpoint."
            logger.error(message)
            raise ValueError(message)
            
        # Weight saving per epoch (dict containing model and optimiser states)
        checkpoint_dict = {
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(), 
            'optimiser_state_dict': optimiser.state_dict(), 
            'val_loss': avg_val_batch_loss
        }
        
        # Always save the latest checkpoint to resume training if interrupted
        torch.save(checkpoint_dict, latest_path)

        # Save best model if val_loss is lower than the current best_val_loss
        if avg_val_batch_loss < best_val_loss:
            torch.save(checkpoint_dict, checkpoint_path)
            best_val_loss = avg_val_batch_loss
            logger.info(
                "Best checkpoint saved epoch = %d val_loss = %.4f path = %s",
                epoch + 1,
                best_val_loss,
                checkpoint_path,
            )
            
        avg_val_image_loss = avg_val_batch_loss / val_loader.batch_size

        logger.info(
            "Epoch completed epoch = %d/%d "
            "train_loss = %.4f val_loss = %.4f best_val_loss = %.4f",
            epoch + 1,
            total_epochs,
            avg_train_batch_loss,
            avg_val_batch_loss,
            best_val_loss,
        )

        logger.debug(
            "Training metrics epoch = %d batch_loss = %.4f image_loss = %.4f "
            "box_loss = %.4f cls_loss = %.4f l1_loss = %.4f",
            epoch + 1,
            avg_train_batch_loss,
            avg_train_image_loss,
            avg_train_box_loss,
            avg_train_cls_loss,
            avg_train_l1_loss,
        )

    if not checkpoint_path.is_file(): 
        message = "No best checkpoint was saved during this run."
        logger.error(message)
        raise RuntimeError(message)
    
    return checkpoint_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate the RGB student model."
    )
    script_dir = Path(__file__).resolve().parent
    default_yaml_path = script_dir / "configs" / "dsec_rgb.yaml"
    
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="DSEC root with labelled train and val directories.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs to train.")
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Run training and validation only; do not run final eval step.",
    )
    parser.add_argument(
        "--data-yaml", type=Path, default=default_yaml_path,
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument( 
        "--console-level", type = str, default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "Set the logging level.",
    )
    return parser.parse_args()

def main() -> None:
    log_directory = log.create_log_directory("student")
    settings.update({"runs_dir": str(log_directory)})
    args = parse_args()
    
    logger = log.create_logger(
        "train",
        log_directory,
        console_level=args.console_level,
    )
    log.log_command(sys.argv, log_directory)
    
    device = select_device()
    
    model = extract_trainable_model(
        yaml_path="configs/yolo26n_student.yaml", 
        weights_path="yolo26n.pt", 
        device=device
    )

    train_loader, val_loader = create_rgb_dataloaders(
        dataset_path = args.dataset_path,
        logger = logger, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )

    optimiser = create_optimiser(model)

    config = { 
        'Selected Device': device,
        'Epochs': args.epochs,
        'Training Samples': len(train_loader.dataset), 
        'Validation Samples': len(val_loader.dataset), 
        'Batch Size': train_loader.batch_size, 
        'Sampler': type(train_loader.sampler).__name__, 
        'Number of Workers': train_loader.num_workers, 
        'Pin Memory': train_loader.pin_memory, 
        'Persistent Workers': train_loader.persistent_workers, 
        'Console Level' : args.console_level, 
        'Skip Test Evaluation': args.skip_test,
    }
    log.log_model_config(config, log_directory, logger)
    
    best_path = log_directory / "best_student_yolo.pt"
    latest_path = log_directory / "latest_student_yolo.pt"

    checkpoint_path = train_model(
        train_loader, 
        val_loader, 
        model, 
        optimiser, 
        device, 
        logger, 
        total_epochs=args.epochs,
        checkpoint_path=best_path, 
        latest_path=latest_path
    )
    
    if args.skip_test:
        logger.info("Test evaluation skipped (--skip-test).")
        return
        
    test_loader = create_rgb_test_loader(
        dataset_path=args.dataset_path,
        logger=logger,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    evaluate_test_model(
        test_loader=test_loader,
        model=model,
        device=device,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )
    evaluate_map_model(
        checkpoint_path=checkpoint_path,
        model_yaml="configs/yolo26n_student.yaml",
        data_yaml=args.data_yaml,
        device=device,
        logger=logger,
    )
        
if __name__ == "__main__":
    main()