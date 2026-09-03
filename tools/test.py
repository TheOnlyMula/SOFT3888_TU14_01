import numpy as np
from pathlib import Path

cache_path = Path(r"C:\Users\Max\Documents\CapStone Project\dataset\test\interlaken_00_a\labels\left\distorted.cache")

# Load the cache dictionary
data = np.load(cache_path, allow_pickle=True).item()

# Print sample image paths stored inside the cache
print(f"Total entries cached: {len(data['labels'])}")
print("\nSample cached sequence paths:")
for img_path in list(data["labels"].keys())[::500]:
    print(img_path)