import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import torchvision
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import config

def get_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_support_set(n_shot = config.n_shot, transform = None):
    num_classes = config.num_classes
    class_name = config.class_name
    path = config.train_path
    support_set = {}
    for name in class_name:
        support_set[name] = []
        class_path = os.path.join(path, name)
        for img_name in os.listdir(class_path)[:n_shot]:
            img_path = os.path.join(class_path, img_name)
            support_set[name].append(img_path)
    
    return support_set

def getPKsamples(p_classes = config.P_classes, k_samples = config.k_samples):
    num_classes = config.num_classes
    class_name = config.class_name
    path = config.train_path
    PK_samples = []
    P_classes = random.sample(class_name, p_classes)
    for id_class, name in enumerate(P_classes):
        class_path = os.path.join(path, name)
        list_img = random.sample(os.listdir(class_path), k_samples)
        for img_name in list_img:
            img_path = os.path.join(class_path, img_name)
            PK_samples.append((img_path, id_class))

    return PK_samples

class TripleFaceDataset(Dataset):
    def __init__(self, hard_triplet = False, transform = None):
        super().__init__()
        self.transform = transform
        self.num_classes = config.num_classes
        self.hard_triplet = hard_triplet
        self.class_name = config.class_name
        self.path = config.train_path
    
    def __len__(self):
        return config.num_per_epoch # Number of iterations per epoch
    
    def __getitem__(self, idx):
        if self.hard_triplet: # if hard_triplet is available
            PK_samples = getPKsamples()
            batch_img = []
            labels = []
            for img_path, label in PK_samples:
                img = get_img(img_path)
                if self.transform:
                    img = self.transform(img)
                batch_img.append(img.numpy())
                labels.append(label)

            batch_img = np.array(batch_img)
            
            return torch.from_numpy(batch_img), torch.from_numpy(np.array(labels, dtype=np.float32))


        anchor_image, positive_image, negative_image = None, None, None

        idx1 = random.randint(0, self.num_classes - 1)
        path_class = os.path.join(self.path, self.class_name[idx1])
        list_img = os.listdir(path_class)
        anchor_image = get_img(os.path.join(path_class, random.choice(list_img)))
        positive_image = get_img(os.path.join(path_class, random.choice(list_img)))

        idx2 = random.randint(0, self.num_classes - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, self.num_classes - 1)
        path_class = os.path.join(self.path, self.class_name[idx2])
        list_img = os.listdir(path_class)
        negative_image = get_img(os.path.join(path_class, random.choice(list_img)))

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


class TestPairFaceDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.transform = transform
        self.num_classes = config.num_classes
        self.class_name = config.class_name
        self.path = config.test_path
        self.list_path = []
        label = 0
        for name in self.class_name:
            class_path = os.path.join(self.path, name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.list_path.append((img_path, label))
            label += 1
        self.pair_path_img = []
        for i in range(len(self.list_path)):
            for j in range(i+1, len(self.list_path)):
                img1, label1 = self.list_path[i]
                img2, label2 = self.list_path[j]
                label = False
                if label1 == label2:
                    label = True
                self.pair_path_img.append((img1, img2, label))

    def __len__(self):
        return len(self.pair_path_img)

    def __getitem__(self, idx):
        path_img1, path_img2, label = self.pair_path_img[idx]
        img1 = get_img(path_img1)
        img2 = get_img(path_img2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label


class TestFaceDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.transform = transform
        self.num_classes = config.num_classes
        self.class_name = config.class_name
        self.path = config.test_path
        self.list_path = []
        label = 0
        for name in self.class_name:
            class_path = os.path.join(self.path, name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.list_path.append((img_path, label))
            label += 1
    
    def __len__(self):
        return len(self.list_path)
    
    def __getitem__(self, idx):
        #print(idx, len(self.list_path))
        img_path, label = self.list_path[idx]
        img = get_img(img_path)
        if self.transform:
            img = self.transform(img)
            #img = self.transform(image = img)['image']

        return img, label

class TripletMnistDataset(Dataset):
    def __init__(self, train, transform = None):
        self.train = train
        self.num_classes = train.shape[0]
        self.transform = transform

    def __len__(self):
        n, s, w, h = self.train.shape
        return n*s

    def __getitem__(self, idx):
        anchor = None
        positive = None
        negative = None

        idx1 = random.randint(0, self.num_classes - 1)
        anchor = random.choice(self.train[idx1])
        positive = random.choice(self.train[idx1])
        
        idx2 = random.randint(0, self.num_classes - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, self.num_classes - 1)
        negative = random.choice(self.train[idx2])
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

class TestMnistDataset(Dataset):
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(np.array([label], dtype=np.float32))