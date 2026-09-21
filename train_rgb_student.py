"""Train the RGB-only student: YOLO26n on 3-channel RGB images.

Usage: python train_rgb_student.py --dataset-path DATASET PATH --epochs 50
"""
from training import main


if __name__ == "__main__":
    main(role="student", rgb_only=True)
