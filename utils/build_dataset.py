from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

class LSTDataset(Dataset):
    def __init__(self, image_path, target_path, transform=None):
        self.data = np.load(image_path)
        self.targets = np.load(target_path)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x).reshape(self.data.shape[1:])

        return x, y

    def __len__(self):
        return len(self.data)

class LSTDataLoader():
    def __init__(self, image_path, target_path, image_transformations=None, train_test_split=0.9, batch_size=32):
        if image_transformations is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.batch_size = batch_size

        dataset = LSTDataset(image_path, target_path, transform)
        train_len = int(len(dataset.data) * train_test_split)
        self.train_set, self.test_set = random_split(dataset, [train_len, len(dataset.data) - train_len])


    def build_loaders(self):
        train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, test_dataloader





if __name__ == "__main__":
    train_dataloader, test_dataloader = LSTDataLoader("../image_data.npy", "../LST_labels.npy").build_loaders()
    print(len(train_dataloader))

