import config 
import numpy as np
import time
from tqdm import tqdm 
from torch.autograd import Variable
import torch 
from custom_dataset import get_support_set
from custom_dataset import get_img

NUM_CLASSES = config.num_classes
thresholds = config.thresholds
hard_triplet = config.hard_triplet

def compare_img(img_output, support_output):
    output1 = img_output
    class_vote = np.zeros((len(thresholds), NUM_CLASSES))

    for class_name in range(NUM_CLASSES):
        total_output = [0]*len(thresholds)
        for output2 in support_output[class_name]:
            distance = (output1 - output2).pow(2).sum()
            for i, threshold in enumerate(thresholds):
                if distance <= threshold:
                    total_output[i] += 1  
        for i in range(len(thresholds)):
            class_vote[i][class_name] = total_output[i]
    
    labels_predict = [0]*len(thresholds)
    class_vote = np.array(class_vote)
    
    for i in range(len(thresholds)):
        labels_predict[i] = np.argmax(class_vote[i])
    
    return labels_predict

def valid_one_epoch(epoch, model, loss_fn, valid_loader, transforms = None):
    model.eval()

    t = time.time()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    num = 0
    
    class_name = config.class_name
    support_output = []
    support_set = get_support_set()

    for id_class in range(NUM_CLASSES):
        batch_image = []
        for path_img in support_set[class_name[id_class]]:
            img = get_img(path_img)
            if transforms:
                img = transforms(image = img)['image']
            batch_image.append(img.numpy())
        batch_image = torch.tensor(batch_image)
        batch_image = Variable(batch_image.float().cuda())
        class_output = None
        if hard_triplet:
            class_output = model(batch_image)
        else:
            class_output = model.embeddingNet(batch_image)
        support_output.append(class_output)
    print('Get ouput of support set done!')

    correct = [0]*len(thresholds)
    for step, (images, labels) in pbar:
        images = Variable(images.float().cuda())
        labels = Variable(labels.float().cuda())
        num += len(images)
        bacth_output = None
        if hard_triplet:
            bacth_output = model(images)
        else:
            bacth_output = model.embeddingNet(images)
        for i in range(len(bacth_output)):
            labels_predict = compare_img(bacth_output[i], support_output)
            for j, label_predict in enumerate(labels_predict):
                if label_predict == labels[i]:
                    correct[j] += 1

    max_acc = 0
    print('[TEST] Epoch {}:'.format(epoch))
    for i, threshold in enumerate(thresholds):
        max_acc = max(max_acc, correct[i]/num)
        print('   Threshold {}: accuracy {} [{}/{}]'.format(threshold, correct[i]/num, correct[i], num))

    return max_acc