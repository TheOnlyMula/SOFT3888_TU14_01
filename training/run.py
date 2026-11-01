"""One training run: from command-line arguments to a validated, tested best checkpoint."""
import sys

from utils import log
from loaders.dataset_loader import create_dataloaders, create_test_loader, sequence_names
from .distillation import (
    EVENT_DENSITY_REFERENCE,
    MASK_ENLARGEMENT,
    TEACHER_CONFIDENCE_THRESHOLD,
    load_teacher,
)
from evaluation.metrics import evaluate_test_model
from ultralytics.utils.torch_utils import init_seeds
from utils.model import (
    NOMINAL_BATCH_SIZE,
    OPTIMISER_LR,
    select_device,
    extract_trainable_model,
    create_optimiser,
)

from .cli import parse_args
from .loop import train_model
from .schedule import WARMUP_EPOCHS


def main(role: str, rgb_only: bool) -> None:
    """Train, validate and test one model. `role` is "student", "teacher" or "distill"."""
    input_channels = 3 if rgb_only else 5
    description = (
        "Train and validate the RGB student model." if rgb_only
        else "Train and validate the RGB + event teacher model."
    )
    args = parse_args(description)
    resuming = args.resume is not None

    log_directory = args.resume if resuming else log.create_log_directory(role)
    latest_path = log_directory / f"latest_{role}_yolo.pt"

    logger = log.create_logger(
        "train",
        log_directory,
        console_level=args.console_level,
    )
    config_file = str(args.config) if args.config else None
    if resuming:
        if not latest_path.is_file():
            message = f"Cannot resume: no latest checkpoint at {latest_path}. Start a new run instead."
            logger.error(message)
            raise FileNotFoundError(message)
        logger.info("Resuming %s with: %s", log_directory, " ".join(sys.argv))
        config_file = log.load_model_config(log_directory).get("Config File")
    else:
        # With --config, this is the command with the file's settings spelled out, so a resume
        # repeats exactly this run even if the file changes later
        log.log_command(args.command, log_directory)

    device = select_device()
    # Seed before anything random is built: the student's fresh head, the shuffle, augmentation
    init_seeds(args.seed, deterministic=True)

    distilling = args.teacher_checkpoint is not None
    if distilling and not rgb_only:
        message = "--teacher-checkpoint only applies to the RGB student."
        logger.error(message)
        raise ValueError(message)
    if args.rgb_dropout and rgb_only:
        # A student must always see its RGB, including in distillation's event-carrying loader
        message = "--rgb-dropout only applies to the teacher, whose input includes events."
        logger.error(message)
        raise ValueError(message)
    quality_weighted = distilling and args.distill_mode == "object-quality"

    model = extract_trainable_model(
        ch=input_channels,
        weights_path="yolo26n.pt",
        device=device,
        epochs=args.epochs,
    )
    # Build the teacher after the student: its random initialisation (overwritten by the checkpoint)
    # would otherwise shift the student's fresh head away from the RGB baseline's for the same seed
    teacher = load_teacher(args.teacher_checkpoint, device, logger) if distilling else None

    loader_settings = dict(
        dataset_path = args.dataset_path,
        logger = logger,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        max_samples = args.max_samples,
    )
    train_settings = dict(
        augment=not args.no_augment,
        seed=args.seed,
        augmentation_shares=args.degradation_shares if args.degradation_augmentation else None,
        dark_strength_range=args.dark_strength_range,
        overexposed_strength_range=args.overexposed_strength_range,
        augmentation_seed=args.augmentation_seed,
    )
    train_loader, val_loader = create_dataloaders(
        rgb_only=rgb_only, rgb_dropout=args.rgb_dropout, **loader_settings, **train_settings,
    )
    if distilling:
        train_loader, _ = create_dataloaders(
            rgb_only=False, with_event_window=quality_weighted, **loader_settings, **train_settings,
        )

    optimiser = create_optimiser(model, batch_size=args.batch_size)

    config = {
        'Model': role,
        'Config File': config_file,
        'Input Channels': input_channels,
        'Selected Device': str(device),
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
        'Max Samples': args.max_samples,
        'Seed': args.seed,
        'Flip Scale Colour Augmentation': not args.no_augment,
        'Nominal Batch Size': NOMINAL_BATCH_SIZE,
        'Warmup Epochs': WARMUP_EPOCHS,
        'Optimiser': type(optimiser).__name__,
        'Learning Rate': OPTIMISER_LR,
        'Deterministic': True,
        'RGB Dropout': args.rgb_dropout,
        'Teacher Checkpoint': str(args.teacher_checkpoint) if distilling else None,
        'Lambda Distill': args.lambda_distill if distilling else None,
        'Distill Mode': args.distill_mode if distilling else None,
        'Object Mask Enlargement': MASK_ENLARGEMENT if distilling and args.distill_mode != "unweighted" else None,
        'Teacher Confidence Threshold': TEACHER_CONFIDENCE_THRESHOLD if distilling else None,
        'Event Density Reference': EVENT_DENSITY_REFERENCE if quality_weighted else None,
        'Training Augmentation': {
            'enabled': args.degradation_augmentation,
            'shares': dict(zip(('clean', 'dark', 'overexposed'), args.degradation_shares)),
            'dark_strength_range': list(args.dark_strength_range),
            'overexposed_strength_range': list(args.overexposed_strength_range),
            'seed': args.augmentation_seed,
        },
    }
    if args.max_samples is not None:
        config['Training Sequences'] = sequence_names(train_loader.dataset)
        config['Validation Sequences'] = sequence_names(val_loader.dataset)
    log.log_model_config(config, log_directory, logger)

    best_path = log_directory / f"best_{role}_yolo.pt"

    checkpoint_path = train_model(
        train_loader,
        val_loader,
        model,
        optimiser,
        device,
        logger,
        total_epochs=args.epochs,
        checkpoint_path=best_path,
        latest_path=latest_path,
        teacher=teacher,
        lambda_distill=args.lambda_distill,
        distill_mode=args.distill_mode,
    )

    if args.skip_test:
        logger.info("Test evaluation skipped (--skip-test).")
        return

    test_loader = create_test_loader(
        dataset_path=args.dataset_path,
        logger=logger,
        rgb_only=rgb_only,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    if args.max_samples is not None:
        logger.info("Test samples = %d sequences = %s", len(test_loader.dataset), sequence_names(test_loader.dataset))
    evaluate_test_model(
        test_loader=test_loader,
        model=model,
        device=device,
        logger=logger,
        checkpoint_path=checkpoint_path,
        save_dir=log_directory / "test",
    )
