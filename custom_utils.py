from torch.utils.data import Dataset, DataLoader
from custom_dataset import TripleFaceDataset, TestFaceDataset, TestPairFaceDataset
import config
import cv2
from torchvision import transforms

def prepare_transforms():
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, valid_transforms

def prepare_dataloader(hard_triplet = False, train_transforms = None, valid_transforms = None):

    train_ds = TripleFaceDataset(hard_triplet = hard_triplet, transform = train_transforms)
    valid_ds = TestPairFaceDataset(transform = valid_transforms)

    print('Length of training set: ', len(train_ds))
    print('Length of valid set: ', len(valid_ds))

    train_loader = DataLoader(
        train_ds, 
        batch_size = config.train_batch,
        shuffle = True, 
        num_workers = 4,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_ds, 
        batch_size = config.valid_batch,
        shuffle = False, 
        num_workers = 4,
    )

    return train_loader, valid_loader