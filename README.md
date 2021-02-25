# FaceNet-OneShotLearning
## Overview  
An experiment of using FaceNet to recognize my friend's face  
* Model:  
  * InceptionResnetV1 (base on https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py)  
  * Resnet50  
  * MobileNetV3  
* Loss function: HardTriplet (Folow from paper https://arxiv.org/pdf/1703.07737.pdf)  
* Optimizer: Adam  
## Result  
Model | F1 score  
------- | -------
InceptionResnetV1 | 0.9803459119496856  
Resnet50 | 0.6976565501344603  
MobileNetV3 | 0.7615541917291393  
## To do  
* Experient with more MobileNet models  
* Face detection with CenterNet  
