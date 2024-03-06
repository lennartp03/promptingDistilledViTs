from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Flowers102DataPytorch:
    def __init__(self, data_dir='./data', train_split_percent=80, batch_size=64):
        self.data_dir = data_dir
        self.train_split_percent = train_split_percent
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dataset_class = datasets.Flowers102

        self._prepare_datasets()

    def _prepare_datasets(self):

        full_dataset = self.dataset_class(root=self.data_dir, split='train', download=True, transform=self.transform)
        
        num_samples = len(full_dataset)
        num_train = int(num_samples * self.train_split_percent / 100)
        num_val = num_samples - num_train
        
        train_dataset, valid_dataset = random_split(full_dataset, [num_train, num_val])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=self.batch_size)

        test_dataset = self.dataset_class(root=self.data_dir, split='test', download=True, transform=self.transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader