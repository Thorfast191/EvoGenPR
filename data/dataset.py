import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, df, labels, root_dir, transform=None):
        """
        df: pandas DataFrame with column 'Image Index'
        labels: numpy array [N, num_classes]
        root_dir: NIH image root directory
        """
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
