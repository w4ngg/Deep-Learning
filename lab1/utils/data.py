from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import datasets, transforms

class customMNIST(Dataset):
    def __init__(self,root='./data',train=True):
        self.data = datasets.MNIST(root=root, train=train, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image,label = self.data[idx]
        if self.transform:
            image = self.transform(image)      
        return image, label
    
def get_data(batch_size=32,num_workers=2):
        '''default setting: val size = 0.1, batch_size = 32'''
        full_ds = customMNIST(train=True)
        test_ds = customMNIST(train=False)
        train_size = int(0.9 * len(full_ds))
        eval_size = len(full_ds) - train_size
        train_ds, eval_ds= random_split(full_ds, [train_size, eval_size])

        train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        val_loader = DataLoader(eval_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        return train_loader,val_loader,test_loader


### data loader return images, labels