import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import torch
import timm
from torchvision.models import resnet50

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = dict(
    acc_920='https://github.com/khrlimam/facenet/releases/download/acc-0.920/model920-6be7e3e9.pth',
    acc_921='https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'
)


def load_state(arch, progress=True):
    state = load_state_dict_from_url(model_urls.get(arch), progress=progress)
    return state


def model_920(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_920', progress)
        model.load_state_dict(state['state_dict'])
    return model


def model_921(pretrained=True, progress=True):
    model = FaceNetModel()
    if pretrained:
        state = load_state('acc_921', progress)
        model.load_state_dict(state['state_dict'])
    return model


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)

class FaceNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super(FaceNetModel, self).__init__()

        self.model = resnet50(pretrained)
        embedding_size = 128
        num_classes = 500
        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        # modify fc layer based on https://arxiv.org/abs/1703.07737
        self.model.fc = nn.Sequential(
            Flatten(),
            # nn.Linear(100352, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(100352, embedding_size))

        self.model.classifier = nn.Linear(embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    # returns face embedding(embedding_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        features = features * alpha
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

class EmbeddingFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FaceNetModel(pretrained=True)

    def forward(self, x):
        output = self.model(x)
        #output = F.normalize(x, p=2, dim=1)
        return output

class EmbeddingMobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('tf_mobilenetv3_small_100', pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 256)

    def forward(self, x):
        x = self.model(x)
        output = F.normalize(x, p=2, dim=1)
        return output


class EmbeddingLeNet(nn.Module):
    def __init__(self):
        super(EmbeddingLeNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Linear(128, 64)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = F.normalize(output, p=2, dim=1)
        return output

class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        return E1, E2, E3

if __name__ == '__main__':
    net = EmbeddingFaceNet()
    print(net)
    #print(list(net.parameters()))
