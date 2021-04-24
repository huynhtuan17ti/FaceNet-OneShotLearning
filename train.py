import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import torchvision
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2
import os
from tqdm import tqdm
import random
from custom_utils import prepare_dataloader, prepare_transforms
from models import EmbeddingMobileNetV3, EmbeddingMobileNetV2, EmbeddingFaceNet, TripletNet
import time
from torch.autograd import Variable
from losses import TripletLoss, HardTripletLoss
import numpy as np
import config
from logger import init_logger

NUM_CLASSES = config.num_classes
NUM_EPOCHS = config.num_epochs 
thresholds = config.thresholds
hard_triplet = config.hard_triplet
LOGGER = init_logger()

def get_model():
    #embeddingNet = EmbeddingFaceNet()
    #embeddingNet = InceptionResnetV1(pretrained='vggface2').eval()
    embeddingNet = EmbeddingMobileNetV3()
    if hard_triplet:
        return embeddingNet
    model = TripletNet(embeddingNet)
    return model

def train_one_epoch_with_hard_triplet(epoch, model, loss_fn, optimizer, train_loader):
    model.train()

    loss_val = 0

    t = time.time()
    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    for step, (batch_img, labels) in pbar:
        batch_img = np.squeeze(batch_img, axis = 0)
        batch_img = Variable(batch_img.float().cuda())
        labels = Variable(labels.float().cuda())

        batch_output = model(batch_img)

        loss = loss_fn(batch_output, labels)

        loss_val += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = f'epoch {epoch} loss: {loss_val/(step+1):.6f}'
        pbar.set_description(description)
    
    LOGGER.info('[TRAIN] Epoch: {}    loss: {}'.format(epoch, loss_val/(len(train_loader))))

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader):
    model.train()

    loss_val = 0

    t = time.time()
    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    for step, (anchor, pos, neg) in pbar:
        anchor = Variable(anchor.float().cuda())
        pos = Variable(pos.float().cuda())
        neg = Variable(neg.float().cuda())
        
        E1, E2, E3 = model(anchor, pos, neg)
        
        loss = loss_fn(E1, E2, E3)

        loss_val += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = f'epoch {epoch} loss: {loss_val/(step+1):.6f}'
        pbar.set_description(description)
    
    LOGGER.info('[TEST] Epoch: {}    loss: {}'.format(epoch, loss_val/(len(train_loader))))

def valid_one_epoch(epoch, model, loss_fn, valid_loader):
    model.eval()

    t = time.time()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    false_positive = [0]*len(thresholds)
    true_positive = [0]*len(thresholds)
    false_negative = [0]*len(thresholds)

    for step, (images1, images2, labels) in pbar:
        images1 = Variable(images1.float().cuda())
        images2 = Variable(images2.float().cuda())
        labels = Variable(labels.float().cuda())

        outputs1 = model(images1)
        outputs2 = model(images2)

        for output1, output2, label in zip(outputs1, outputs2, labels):
            x = output1.unsqueeze(0)
            y = output2.unsqueeze(0)
            dist = (x-y).pow(2).sum()
            for i, threshold in enumerate(thresholds):
                predict = False
                if threshold >= dist:
                    predict = True
                if predict == label and label == True:
                    true_positive[i] += 1
                if predict != label and label == False:
                    false_positive[i] += 1
                if predict != label and label == True:
                    false_negative[i] += 1
            
    esp = 1e-9
    max_f1 = 0
    LOGGER.info('[TEST] Epoch {}:'.format(epoch))
    for i, threshold in enumerate(thresholds):
        precision = true_positive[i]/(true_positive[i] + false_positive[i] + esp)
        recall = true_positive[i]/(true_positive[i] + false_negative[i] + esp)
        f1 = 2*(precision*recall)/(precision + recall + esp)
        max_f1 = max(max_f1, f1)
        LOGGER.info('* Threshold {}:'.format(threshold))
        LOGGER.info('       Precision score: {:.4f}'.format(precision))
        LOGGER.info('       Recall score: {:.4f}'.format(recall))
        LOGGER.info('       F1 score: {:.4f}'.format(f1))

    return max_f1

def test(optimizer):
    optimizer.zero_grad()
    print('fuck')

if __name__ == '__main__':
    train_transforms, valid_transforms = prepare_transforms()
    train_loader, valid_loader = prepare_dataloader(hard_triplet = hard_triplet, train_transforms = train_transforms, valid_transforms = valid_transforms)

    loss_fn = None
    if hard_triplet:
        loss_fn = HardTripletLoss()
    else:
        loss_fn = TripletLoss()
    
    model = get_model().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
    test(optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 70, 80], 
                                                     gamma=0.1, last_epoch=-1, verbose=True)

    LOGGER.info('Loss function :{}, optimizer: {}'.format(loss_fn, optimizer))
    LOGGER.info('========== Start training =========='.format(config.n_shot))
    valid_range = 1
    best_F1 = 0
    for epoch in range(NUM_EPOCHS):
        if hard_triplet:
            train_one_epoch_with_hard_triplet(epoch, model, loss_fn, optimizer, train_loader)
        else:
            train_one_epoch(epoch, model, loss_fn, optimizer, train_loader)
        if (epoch+1)%valid_range == 0 or epoch == NUM_EPOCHS-1:
            with torch.no_grad():
                valid_F1 = valid_one_epoch(epoch, model, loss_fn, valid_loader)
                if valid_F1 > best_F1:
                    best_F1 = valid_F1
                    torch.save(model.state_dict(), config.model_path + '/' + config.model_name)
                    print('Best max F1 score save at epoch {}, score: {}'.format(epoch, best_F1))
                LOGGER.info('### Epoch {}: cur_f1: {}, best_f1: {} ###'.format(epoch, valid_F1, best_F1))
        scheduler.step()





