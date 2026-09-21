from torch.utils.data import DataLoader
import log
from dataset_loader import DSECFusionDataset, limit_dataset, yolo_collate_fn
from training import parse_args, train_model
from utils import select_device, extract_trainable_model, create_optimiser


def tiny_loader(dataset_path, num_samples, batch_size):
    dataset = DSECFusionDataset(dataset_path=dataset_path, split="train", rgb_only=True)
    subset = limit_dataset(dataset, num_samples)
    return DataLoader(subset, batch_size=batch_size, collate_fn=yolo_collate_fn)


def main():
    args = parse_args("Overfit the RGB student on a tiny training subset.")
    log_directory = log.create_log_directory("overfit")
    logger = log.create_logger("overfit", log_directory)
    device = select_device()
    model = extract_trainable_model(ch=3, device=device, epochs=args.epochs)
    loader = tiny_loader(args.dataset_path, args.max_samples or 8, args.batch_size)

    # Validate on the training frames: the model should memorise them.
    train_model(
        loader,
        loader,
        model,
        create_optimiser(model),
        device,
        logger,
        total_epochs=args.epochs,
        checkpoint_path=log_directory / "best_overfit_yolo.pt",
        latest_path=log_directory / "latest_overfit_yolo.pt",
    )


if __name__ == "__main__":
    main()