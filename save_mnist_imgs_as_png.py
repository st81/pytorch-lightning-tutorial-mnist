from pathlib import Path

import polars
from torchvision.datasets import MNIST
from tqdm.contrib import tenumerate

data_dir = "data"
save_dir = Path("data/mnist_imgs")
save_dir.mkdir(parents=True, exist_ok=True)
test_dataset = MNIST(root=data_dir, train=False, download=False)
img_path_targets = []
for i, (img, target) in tenumerate(test_dataset):
    save_path = save_dir / f"mnist_img_{i:04d}.png"
    img.save(save_path)
    img_path_targets.append({"img_path": str(save_path), "target": target})
img_path_targets = polars.DataFrame(img_path_targets)
print(img_path_targets)
img_path_targets.write_csv("data/test.csv")
