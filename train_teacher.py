"""Train the teacher: YOLO26n on RGB images plus a 2-channel event histogram (5 channels).

Usage: python train_teacher.py --dataset-path dataset --epochs 50
"""
from training import main


if __name__ == "__main__":
    main(role="teacher", rgb_only=False)
