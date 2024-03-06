from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np

class CaltechDataPytorch:
    def __init__(self, data_dir='./data', batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset_class = Caltech101Filtered

        self._prepare_datasets()

    def _prepare_datasets(self):
        full_dataset = datasets.Caltech101(self.data_dir, download=False, transform=self.transform)
        num_samples = len(full_dataset)
        print("Total number of samples:", num_samples)

        num_train = 800
        num_val = 200
        assert num_train + num_val <= num_samples, "Requested train/val exceeds available samples"

        # Generate indices and split the dataset
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Create datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


class Caltech101Filtered(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.Caltech101(root=root, download=False, transform=transform)
        self.transform = transform
        self.indices = self._filter_indices()

    def _filter_indices(self):
        indices = []
        for i in range(len(self.dataset)):
            img, _ = self.dataset[i]
            if img.mode == 'RGB':  
                indices.append(i)
        return indices

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)