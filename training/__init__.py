"""Shared training pipeline for the RGB student, the RGB + event teacher and the distilled student.

Both models use the same YOLO26n architecture, training loop, validation, logging and test
evaluation. The only difference is the input: the student sees 3 RGB channels, the teacher
sees RGB plus a 2-channel event histogram (5 channels).

Passing a teacher checkpoint turns on distillation: the student still trains on RGB only, while
a frozen teacher sees the paired RGB + event frames and supervises the student's features.
Validation and testing always use RGB-only data, so student results never depend on events.

The modules follow the order a run happens in:

    cli        command-line arguments, including resuming an interrupted run
    run        main(): builds the data, models and optimiser from the arguments, trains, tests
    loop       train_model(): epochs, gradient accumulation, validation, best-checkpoint choice
    losses     one batch's detection loss and, with a teacher, its distillation loss
    schedule   Ultralytics' warmup and linear learning-rate decay
    reporting  each epoch's summary in train.log and its row in results.csv
"""
from .cli import parse_args
from .loop import train_model
from .run import main

__all__ = ["main", "parse_args", "train_model"]
