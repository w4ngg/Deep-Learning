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
    
def get_MNIST_data(batch_size=32,num_workers=2):
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

def get_VNFood21_data(batch_size=32, num_workers=2, val_ratio=0.1):
   
    train_path = r"D:\Dowload\VinaFood21\VinaFood21\train"
    test_path = r"D:\Dowload\VinaFood21\VinaFood21\test"
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # tăng tính đa dạng
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # chuẩn ImageNet
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load full training dataset
    full_train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=val_test_transform)

    # Split training data into train and validation
    train_size = int((1 - val_ratio) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # For validation, we want to use the same transform as test (without augmentation)
    val_dataset.dataset.transform = val_test_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
### data loader return images, labels