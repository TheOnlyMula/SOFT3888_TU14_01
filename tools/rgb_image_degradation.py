import argparse
from pathlib import Path
import cv2
import random

class RGBImageDegradation:
    def __init__(self, dataset_path: Path, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_path = self.dataset_path / split

    def validate_dataset(self) -> None:
        if not self.split_path.exists():
            raise FileNotFoundError(
                f"DSEC {self.split} folder not found: {self.split_path.resolve()}"
            )

    def get_sequences(self) -> list[dict]:
        self.validate_dataset()
        sequences = []

        for sequence_path in sorted(self.split_path.iterdir()):
            if not sequence_path.is_dir():
                continue

            rgb_path = sequence_path / "images" / "left" / "distorted"
            if rgb_path.exists():
                sequences.append({
                    "name": sequence_path.name,
                    "sequence_path": sequence_path,
                    "rgb_path": rgb_path,
                })

        if not sequences:
            raise RuntimeError(f"No valid DSEC sequences found in: {self.split_path.resolve()}")
        return sequences

    def get_image_paths(self) -> list[dict]:
        sequences = self.get_sequences()
        image_paths = []

        for sequence in sequences:
            rgb_path = sequence["rgb_path"]
            images = sorted(rgb_path.glob("*.png"))
            for image_path in images:
                image_paths.append({
                    "sequence": sequence["name"],
                    "path": image_path,
                })

        if not image_paths:
            raise RuntimeError(f"No RGB images found in the DSEC {self.split} dataset.")
        return image_paths

    def load_image(self, image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        return image

    def degrade_brightness(self, image):
        if random.choice([True, False]):
            factor = random.uniform(0.3, 0.7)
        else:
            factor = random.uniform(1.3, 2.0)
            
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def save_degraded_dataset(self, output_path: Path, limit: int = None) -> int:
        """
        Load, degrade, and save RGB images one by one.
        If limit is provided, only processes that many images.
        """
        image_paths = self.get_image_paths()

        if limit is not None and limit > 0:

            image_paths = image_paths[:limit]

        processed_count = 0

        for sample in image_paths:
            original_image = self.load_image(sample["path"])
            
            degraded_image = self.degrade_brightness(original_image)
            
            save_directory = (
                output_path
                / self.split
                / sample["sequence"]
                / "images"
                / "left"
                / "distorted"
            )
            save_directory.mkdir(parents=True, exist_ok=True)
            save_path = save_directory / sample["path"].name

            success = cv2.imwrite(str(save_path), degraded_image)
            if not success:
                raise RuntimeError(f"Failed to save image: {save_path}")
                
            processed_count += 1

        return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and degrade RGB images from the DSEC dataset."
    )
    parser.add_argument(
        "--dataset-path", "-d", type=Path, required=True,
        help="Path to the root directory of the DSEC dataset.",
    )
    parser.add_argument(
        "--output-path", "-o", type=Path, default=Path("degraded_data"),
        help="Directory used to save degraded RGB images.",
    )
    parser.add_argument(
        "--split", "-s", type=str, default="train", choices=["train", "test"],
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Maximum number of images to degrade. If not set, degrades all images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.split} RGB dataset from: {args.dataset_path}")

    rgb_pipeline = RGBImageDegradation(
        dataset_path=args.dataset_path,
        split=args.split,
    )

    sequences = rgb_pipeline.get_sequences()
    print(f"Found {len(sequences)} DSEC sequences.")

    print(f"\nSaving degraded images to: {args.output_path.resolve()}")
    
    images_saved = rgb_pipeline.save_degraded_dataset(
        output_path=args.output_path,
        limit=args.limit
    )

    print(f"Successfully saved {images_saved} degraded RGB images.")


if __name__ == "__main__":
    main()