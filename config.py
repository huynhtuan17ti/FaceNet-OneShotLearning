log_path = '../FaceNet-OneShotLearning/log_file/log_MobileNetV3'
train_path = '../FaceNet-OneShotLearning/data/FaceData/Train'
test_path = '../FaceNet-OneShotLearning/data/FaceData/Test'
model_path = '../FaceNet-OneShotLearning/save_model'
model_name = 'MobileNetV3.pth'
num_classes = 5
class_name = ['Tuan', 'Tuong', 'Nhi', 'Lam', 'Kiet']
P_classes = 5
k_samples = 12
n_shot = 5
num_epochs = 30
thresholds = [0.5, 0.75, 1, 1.2]
hard_triplet = True
train_batch = 1 # 1 if hard_triplet = True
valid_batch = 16
num_per_epoch = 50