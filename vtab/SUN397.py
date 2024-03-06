from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class SUN397DataPytorch:
    def __init__(self, data_dir='./data', batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset_class = datasets.SUN397

        self._prepare_datasets()

    def _prepare_datasets(self):
        full_dataset = self.dataset_class(self.data_dir, download=True, transform=self.transform)
        num_samples = len(full_dataset)
        print("Total number of samples:", num_samples)

        num_train = 800
        num_val = 200
        assert num_train + num_val <= num_samples, "Requested train/val exceeds available samples"

        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader