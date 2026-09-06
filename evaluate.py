import argparse
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataset_loader import create_test_dataloader

class ValLoaderWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
    def __iter__(self):
        for imgs, targets in self.loader:
            B = imgs.shape[0]
            H, W = imgs.shape[2], imgs.shape[3]
            yield {
                'img': imgs * 255,
                'bboxes': targets[:, 2:],
                'cls': targets[:, 1:2],
                'batch_idx': targets[:, 0],
                'im_file': [f"dummy_{i}.png" for i in range(B)],
                'ori_shape': [(H, W)] * B,
                'resized_shape': [(H, W)] * B,
                'ratio_pad': [((1.0, 1.0), (0.0, 0.0))] * B
            }
    def __len__(self):
        return len(self.loader)

def _prepare_yolo_batch(imgs, targets, device):
    """Move a collated batch to the device and expose YOLO's target fields."""
    imgs = imgs.to(device)
    targets = targets.to(device)
    return {
        "img": imgs,
        "cls": targets[:, 1:2],
        "bboxes": targets[:, 2:],
        "batch_idx": targets[:, 0],
    }

def validate_model(
        val_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        logger: logging.Logger
    ) -> tuple[float, float, float, float]:
    
    model.eval()
    total_loss = 0.0
    total_box_loss = 0.0
    total_cls_loss = 0.0
    total_l1_loss = 0.0
    logger.info("Starting Validation...")
    
    with torch.no_grad():
        for batch_idx_step, (imgs, targets) in enumerate(val_loader):
            loss, loss_items = model(_prepare_yolo_batch(imgs, targets, device))
            
            step_loss = loss.sum().item()
            
            step_box_loss = loss_items['box_loss']
            step_cls_loss = loss_items['cls_loss']
            step_l1_loss = loss_items['l1_loss']
            
            if isinstance(step_box_loss, torch.Tensor): step_box_loss = step_box_loss.item()
            if isinstance(step_cls_loss, torch.Tensor): step_cls_loss = step_cls_loss.item()
            if isinstance(step_l1_loss, torch.Tensor): step_l1_loss = step_l1_loss.item()
            
            total_loss += step_loss
            total_box_loss += step_box_loss
            total_cls_loss += step_cls_loss
            total_l1_loss += step_l1_loss
            
            if batch_idx_step % 10 == 0 and batch_idx_step > 0:
                logger.debug(
                    f"Val Step [{batch_idx_step}/{len(val_loader)}] "
                    f"| Val: {step_loss:.4f} "
                    f"| Box: {step_box_loss:.4f} "
                    f"| Cls: {step_cls_loss:.4f} "
                    f"| L1: {step_l1_loss:.4f}"
                )
                
    num_batches = len(val_loader)
    return (
        total_loss / num_batches, 
        total_box_loss / num_batches, 
        total_cls_loss / num_batches, 
        total_l1_loss / num_batches
    )


def evaluate_test_model(
    checkpoint_path: Path, 
    model_yaml: Path | str,
    data_yaml: Path | str,
    device: torch.device,
    logger: logging.Logger, 
    dataset_path: Path = None,
    rgb_only: bool = False,
    batch_size: int = 16,
    num_workers: int = 4
):
    if not checkpoint_path.is_file():
        message = f"Best checkpoint not found: {checkpoint_path}"
        logging.error(message)
        raise FileNotFoundError(message)

    logger.info(f"Loading best weights from {checkpoint_path} into Ultralytics engine...")
    
    import yaml
    with open(model_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    ch = cfg.get('ch', 3)

    yolo_model = YOLO(str(model_yaml), task="detect") 
    if ch != 3:
        from ultralytics.nn.tasks import DetectionModel
        yolo_model.model = DetectionModel(str(model_yaml), ch=ch, nc=cfg.get('nc', 80))
    
    dsec_names = {
        0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus', 
        4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
    }
    yolo_model.model.names = dsec_names
    
    custom_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    yolo_model.model.load_state_dict(custom_checkpoint["model_state_dict"])
    
    logger.info("Starting mAP evaluation on the test split...")
    
    validator = yolo_model.task_map["detect"]["validator"](args={'data': str(data_yaml), 'split': 'test', 'device': str(device)})
    
    if dataset_path is not None:
        test_loader = create_test_dataloader(dataset_path, rgb_only=rgb_only, batch_size=batch_size, num_workers=num_workers)
        validator.dataloader = ValLoaderWrapper(test_loader)
        
    import ultralytics.nn.autobackend
    ultralytics.nn.autobackend.AutoBackend.warmup = lambda *args, **kwargs: None
    validator(model=yolo_model.model)
    metrics = validator.metrics
    
    logger.info(f"Test mAP@50: {metrics.box.map50:.4f}")
    logger.info(f"Test mAP@50-95: {metrics.box.map:.4f}")
    
    return metrics


if __name__ == "__main__":
    import log
    import sys
    
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained weights (.pt)")
    parser.add_argument("--type", type=str, required=True, choices=["student", "teacher"], help="Evaluating student or teacher")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    
    data_yaml = script_dir / "configs" / "dsec_rgb.yaml" 
    model_yaml = script_dir / "configs" / f"yolo26n_{args.type}.yaml"  
    
    log_directory = log.create_log_directory(args.type)
    logger = log.create_logger("evaluate", log_directory)
    log.log_command(sys.argv, log_directory)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_test_model(
        args.weights, 
        model_yaml, 
        data_yaml, 
        device, 
        logger,
        dataset_path=args.dataset_path,
        rgb_only=(args.type == "student"),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )