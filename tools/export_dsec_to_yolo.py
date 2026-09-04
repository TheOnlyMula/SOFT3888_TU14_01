import argparse
from pathlib import Path
from tqdm import tqdm
from dsec_det.dataset import DSECDet
from dsec_det.io import yaml_file_to_dict


def export_split_to_yolo(dataset_path: Path, split: str, split_config_path: Path) -> None:
    """Extract tracks from DSECDet and export them as standard YOLO .txt label files."""
    split_config = yaml_file_to_dict(split_config_path)
    
    print(f"\nLoading DSEC dataset for split: '{split}'...")
    dsec_data = DSECDet(
        dataset_path, 
        split=split, 
        sync="back", 
        split_config=split_config
    )
    
    print(f"Exporting {len(dsec_data)} frames to YOLO format...")

    for idx in tqdm(range(len(dsec_data)), desc=f"Exporting {split}"):
        # Fetch image shape and corresponding bounding box tracks
        image = dsec_data.get_image(idx)
        tracks = dsec_data.get_tracks(idx)
        h, w = image.shape[0], image.shape[1]

        # Retrieve the image path to determine the corresponding label path
        # DSEC image path format: .../{split}/{sequence}/images/left/distorted/000000.png
        # Retrieve the relative index and directory to grab the exact image path
        index_rel, _, directory = dsec_data.rel_index(idx)
        img_path = Path(directory.images.image_files_distorted[index_rel])
        
        # Mirror image path to label path: replace '/images/' with '/labels/' and '.png' with '.txt'
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_lines = []
        for box in tracks:
            cls_id = int(box["class_id"])
            box_x, box_y, box_w, box_h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])

            # Filter invalid or zero-dimension boxes
            if box_w <= 0 or box_h <= 0:
                continue

            # Convert (top-left x, y, width, height) to normalized (center_x, center_y, width, height)
            x_center = (box_x + box_w / 2.0) / w
            y_center = (box_y + box_h / 2.0) / h
            norm_w = box_w / w
            norm_h = box_h / h

            # Clamp coordinates to [0, 1] range to avoid clipping warnings
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Write labels out to .txt (creates an empty file for background frames with no objects)
        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

    print(f"Finished exporting '{split}' labels.")


def main():
    parser = argparse.ArgumentParser(description="Export DSEC-Det tracks to YOLO format.")
    parser.add_argument(
        "--dataset-path", 
        type=Path, 
        required=True, 
        help="Root path to the DSEC dataset."
    )
    parser.add_argument(
        "--config-path", 
        type=Path, 
        default=Path("./DSEC-det/config/train_val_test_split.yaml"),
        help="Path to train_val_test_split.yaml."
    )
    parser.add_argument(
        "--splits", 
        nargs="+", 
        default=["train", "val", "test"], 
        help="Splits to process."
    )
    args = parser.parse_args()

    for split in args.splits:
        export_split_to_yolo(args.dataset_path, split, args.config_path)


if __name__ == "__main__":
    main()