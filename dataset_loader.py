import argparse
from pathlib import Path
import cv2
import hdf5plugin  # Required to prevent Blosc decompression error
import numpy as np
import torch
from torch.utils.data import Dataset

from dsec_det.dataset import DSECDet
from dsec_det.label import CLASSES, COLORS


class DSECFusionDataset(Dataset):
    def __init__(self, dataset_path, split="train", rgb_only=False):
        # sync="back" ensures one fixed event time window before the image
        self.base_dataset = DSECDet(dataset_path, split=split, sync="back")
        self.rgb_only = rgb_only  # Toggle for RGB-only evaluation mode

    def __len__(self):
        return len(self.base_dataset)

    def generate_event_histogram(self, events, height=1080, width=1440):
        # Generates a two channel event histogram representation
        hist = np.zeros((2, height, width), dtype=np.float32)

        # Check if events exist and if the internal 'x' array actually has data
        if events is not None and len(events.get("x", [])) > 0:
            x = events["x"].astype(int)
            y = events["y"].astype(int)
            p = events["p"].astype(int)

            # Channel 0 (Negative/0), Channel 1 (Positive/1)
            np.add.at(hist[0], (y[p == 0], x[p == 0]), 1)
            np.add.at(hist[1], (y[p == 1], x[p == 1]), 1)

        return torch.from_numpy(hist)

    def __getitem__(self, idx):
        if self.rgb_only:
            # RGB-only training/testing must not open event files.
            sample = {
                "image": self.base_dataset.get_image(idx),
                "tracks": self.base_dataset.get_tracks(idx),
            }
        else:
            sample = self.base_dataset[idx]

        # RGB Image
        image = sample["image"]
        h, w = image.shape[0], image.shape[1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # YOLO Bounding Box Conversion
        yolo_targets = []
        for box in sample["tracks"]:
            cls_id = int(box["class_id"])
            box_x, box_y, box_w, box_h = box["x"], box["y"], box["w"], box["h"]
            
            # Calculate center points and normalize
            x_center = (box_x + box_w / 2.0) / w
            y_center = (box_y + box_h / 2.0) / h
            norm_w = box_w / w
            norm_h = box_h / h
            
            yolo_targets.append([cls_id, x_center, y_center, norm_w, norm_h])
            
        # Convert to a PyTorch tensor
        target_tensor = torch.tensor(yolo_targets, dtype=torch.float32)

        if self.rgb_only:
            return img_tensor, target_tensor
        else:
            event_tensor = self.generate_event_histogram(
                sample.get("events"), height=h, width=w
            )
            return img_tensor, event_tensor, target_tensor

def yolo_rgb_collate_fn(batch):
    """Collate function for RGB-only YOLO training."""
    imgs, targets = zip(*batch)
    
    # Stack images into a standard batch tensor
    imgs = torch.stack(imgs, 0)
    
    # Add the batch index to the targets
    targets_with_idx = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            batch_idx = torch.full((target.shape[0], 1), i, dtype=target.dtype)
            targets_with_idx.append(torch.cat((batch_idx, target), dim=1))
            
    if len(targets_with_idx) > 0:
        targets = torch.cat(targets_with_idx, 0) # Shape: [Total_Boxes, 6]
    else:
        targets = torch.empty((0, 6))
        
    return imgs, targets

def yolo_collate_fn(batch):
    """
    Collate function for YOLO to handle variable number of bounding boxes.
    """
    imgs, events, targets = zip(*batch)
    
    # Stack images and events into standard batch tensors
    imgs = torch.stack(imgs, 0)
    events = torch.stack(events, 0)
    
    # Add the batch index to the targets
    targets_with_idx = []
    for i, target in enumerate(targets):
        if target.shape[0] > 0:
            # Create a column of batch indices (i)
            batch_idx = torch.full((target.shape[0], 1), i, dtype=target.dtype)
            # Concatenate index with the [cls, x, y, w, h] tensor
            targets_with_idx.append(torch.cat((batch_idx, target), dim=1))
            
    if len(targets_with_idx) > 0:
        targets = torch.cat(targets_with_idx, 0) # Shape: [Total_Boxes, 6]
    else:
        targets = torch.empty((0, 6))
        
    return imgs, events, targets

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DSEC RGB and Event Histogram samples with bounding boxes."
    )
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=Path,
        required=True,
        help="Path to the root directory of the DSEC dataset.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="Sample index to fetch and visualize (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading dataset from: {args.dataset_path}")
    my_dataset = DSECFusionDataset(args.dataset_path, split=args.split, rgb_only=False)

    if args.index >= len(my_dataset) or args.index < 0:
        raise IndexError(
            f"Index {args.index} out of range. Dataset contains {len(my_dataset)} samples."
        )

    print(f"Fetching sample {args.index}...")
    img_tensor, event_tensor, targets = my_dataset[args.index]

    # --- VISUALIZATION CODE ---

    # Convert RGB Tensor back to standard OpenCV image (H, W, C)
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)

    # Convert Event Tensor into a color image
    events_np = event_tensor.numpy()
    H, W = events_np.shape[1], events_np.shape[2]
    event_vis = np.zeros((H, W, 3), dtype=np.uint8)

    # Match polarity colors: p=0 -> Red (BGR index 2), p=1 -> Blue (BGR index 0)
    event_vis[:, :, 2] = np.clip(events_np[0] * 50, 0, 255).astype(np.uint8)
    event_vis[:, :, 0] = np.clip(events_np[1] * 50, 0, 255).astype(np.uint8)

    # Draw Bounding Boxes and Class Labels on both images
    # Draw Bounding Boxes and Class Labels on both images
    for row in targets:
        cls_id = int(row[0].item())
        x_c, y_c, norm_w, norm_h = row[1].item(), row[2].item(), row[3].item(), row[4].item()

        # Un-normalize YOLO coordinates back to absolute pixels
        w = int(norm_w * W)
        h = int(norm_h * H)
        x = int((x_c * W) - (w / 2))
        y = int((y_c * H) - (h / 2))

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = f"{CLASSES[cls_id]}"

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5 * txt_size[1])

        for img in (img_np, event_vis):
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y - txt_height), (x + txt_size[0] + 1, y + 1), txt_bk_color, -1)
            cv2.putText(img, text, (x, y + txt_size[1] - txt_height), font, 0.4, txt_color, 1)
    # Display the results side by side
    combined_img = np.hstack((img_np, event_vis))

    print("Opening visualizer window... (Press any key on the image window to close it)")
    cv2.imshow(f"Sample {args.index}: RGB vs Event Histogram", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()