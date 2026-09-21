from torch.utils.data import DataLoader
from dataset_loader import DSECFusionDataset, limit_dataset, yolo_collate_fn


def tiny_loader(dataset_path, num_samples, batch_size):
    dataset = DSECFusionDataset(dataset_path=dataset_path, split="train", rgb_only=True)
    subset = limit_dataset(dataset, num_samples)
    return DataLoader(subset, batch_size=batch_size, collate_fn=yolo_collate_fn)