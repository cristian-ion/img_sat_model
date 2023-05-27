import numpy as np
import pandas as pd
import PIL as pil

import torch.utils.data


class ImgClsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file: str, dataset_type: str, shuffle: bool, load_folds=None, transform=None):
        """
        load_folds: [1,2,3] means 60% of data if num_folds is 5
        """
        data = pd.read_csv(dataset_file)
        self.total_samples = len(data)
        print(data.head())
        
        if shuffle:
            print("Shuffle dataset.")
            data = data.sample(frac=1).reset_index(drop=True)

        if load_folds:
            data = data[data["fold"].isin(load_folds)]

        self.num_samples = len(data)
        self.samples = data
        self.transform = transform
        self.labels = list(data['label'].unique())
        self.num_classes = len(self.labels)

        print(f"Total count samples {self.total_samples}")
        print(f"Count samples {self.num_samples} ({self.num_samples / self.total_samples})")
        print(f"Labels = {self.labels}")
        print(f"Num Classes = {self.num_classes}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = self.samples.iloc[idx, 0]
        label_index = self.samples.iloc[idx, 2]

        label = self.one_hot_labels(label_index)

        if self.transform:
            image = pil.Image.open(img_path)
            image = self.transform(image)

        return image, label
    
    def one_hot_labels(self, label_index):
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[label_index] = 1.0
        return label
