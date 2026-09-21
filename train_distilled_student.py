"""Train the RGB student with distillation from a frozen RGB + event teacher.

The student trains on RGB only and the teacher sees the paired event channels. Validation and
test use RGB-only data.

Usage: python train_distilled_student.py --dataset-path DATASET PATH --epochs 50 --teacher-checkpoint PATH TO CHECKPOINT --lambda-distill 1.0
"""
from training import main


if __name__ == "__main__":
    main(role="distill", rgb_only=True)
