import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, df, labels, root, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.transform = transform

        # 🔹 Build image index
        self.image_map = {}
        for shard in os.listdir(root):
            shard_path = os.path.join(root, shard, "images")
            if os.path.isdir(shard_path):
                for fname in os.listdir(shard_path):
                    self.image_map[fname] = os.path.join(shard_path, fname)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]

        if img_name not in self.image_map:
            raise FileNotFoundError(f"Image not found: {img_name}")

        img_path = self.image_map[img_name]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label