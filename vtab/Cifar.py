from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

class CifarDataPytorch:
    def __init__(self, num_classes=10, data_dir='./data', train_split_percent=90, batch_size=64):
        self.data_dir = data_dir
        self.train_split_percent = train_split_percent
        self.batch_size = batch_size

        if num_classes not in [10, 100]:
            raise ValueError("Number of classes must be 10 or 100, got {}".format(num_classes))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if num_classes == 10:
            self.dataset_class = datasets.CIFAR10
        else:
            self.dataset_class = datasets.CIFAR100

        self._prepare_datasets()

    def _prepare_datasets(self):

        full_train_dataset = self.dataset_class(self.data_dir, train=True, download=True, transform=self.transform)
        test_dataset = self.dataset_class(self.data_dir, train=False, download=True, transform=self.transform)

        num_train = 1000 
        print(num_train, len(test_dataset))
        indices = list(range(num_train))
        split = int(np.floor(self.train_split_percent / 100.0 * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[:split], indices[split:]
        train_dataset = Subset(full_train_dataset, train_idx)
        valid_dataset = Subset(full_train_dataset, valid_idx)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
