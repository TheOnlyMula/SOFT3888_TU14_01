import argparse
from pathlib import Path

import cv2


class RGBImageDegradation:
    def __init__(
        self,
        dataset_path: Path,
        split: str = "train"
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_path = self.dataset_path / split

    def validate_dataset(self) -> None:
        """
        Check that the requested DSEC dataset split exists.
        """
        if not self.split_path.exists():
            raise FileNotFoundError(
                f"DSEC {self.split} folder not found: "
                f"{self.split_path.resolve()}"
            )

    def get_sequences(self) -> list[dict]:
        """
        Find every DSEC sequence inside the selected split.
        """
        self.validate_dataset()

        sequences = []

        for sequence_path in sorted(self.split_path.iterdir()):
            if not sequence_path.is_dir():
                continue

            rgb_path = (
                sequence_path
                / "images"
                / "left"
                / "distorted"
            )

            if rgb_path.exists():
                sequences.append({
                    "name": sequence_path.name,
                    "sequence_path": sequence_path,
                    "rgb_path": rgb_path,
                })

        if not sequences:
            raise RuntimeError(
                f"No valid DSEC sequences found in: "
                f"{self.split_path.resolve()}"
            )

        return sequences

    def get_image_paths(self) -> list[dict]:
        """
        Get all RGB image paths from every sequence
        in the selected split.
        """
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
            raise RuntimeError(
                f"No RGB images found in the "
                f"DSEC {self.split} dataset."
            )

        return image_paths

    def load_image(self, image_path: Path):
        """
        Load one RGB image using OpenCV.
        """
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            )

        return image

    def load_dataset(self) -> list[dict]:
        """
        Load RGB images from every DSEC sequence
        in the selected split.
        """
        image_paths = self.get_image_paths()
        dataset = []

        for sample in image_paths:
            image_path = sample["path"]
            image = self.load_image(image_path)

            dataset.append({
                "split": self.split,
                "sequence": sample["sequence"],
                "filename": image_path.name,
                "original_path": image_path,
                "image": image,
            })

        return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load RGB images from the DSEC dataset."
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
        choices=["train", "test"],
        help="Dataset split to load (default: train).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Loading {args.split} RGB dataset from: "
        f"{args.dataset_path}"
    )

    rgb_pipeline = RGBImageDegradation(
        dataset_path=args.dataset_path,
        split=args.split,
    )

    sequences = rgb_pipeline.get_sequences()

    print(f"Found {len(sequences)} DSEC sequences:")

    for sequence in sequences:
        print(f"  {sequence['name']}")

    dataset = rgb_pipeline.load_dataset()

    print(
        f"\nLoaded {len(dataset)} total "
        f"{args.split} RGB images."
    )

    print("\nFirst RGB images:")

    for sample in dataset[:5]:
        print(
            sample["sequence"],
            sample["filename"],
            sample["image"].shape,
        )


if __name__ == "__main__":
    main()