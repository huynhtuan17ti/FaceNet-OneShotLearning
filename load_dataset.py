import os
import numpy as np
import gzip
import matplotlib.pyplot as plt

NUM_CLASSES = 10
SAMPLE_SIZE = 10

def load_mnist(path = './data/unzip', s = 'train', one_shot = True):
    if s != 'train':
        one_shot = False

    labels_path = os.path.join(path, "{}-labels-idx1-ubyte.gz".format(s))
    images_path = os.path.join(path, "{}-images-idx3-ubyte.gz".format(s))

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.float64)

    print("Load {} mnist data successfully!".format(s))
    print("{}: {} images".format(s, len(images)))

    if one_shot:
        print("Modify data for one-shot learning! ...", end = ' ')
        w, h = images[0].shape
        train = np.zeros((NUM_CLASSES, SAMPLE_SIZE, w, h))
        cnt = [0]*NUM_CLASSES
        for i in range(len(images)):
            label = labels[i]
            cur_id = cnt[label]
            if cnt[label] < SAMPLE_SIZE:
                train[label][cur_id] = images[i]
                cnt[label] += 1
        print('Done')
        print(train.shape)
        return train

    return images, labels

if __name__ == '__main__':
    train = load_mnist()