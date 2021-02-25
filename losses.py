import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class HardTripletLoss(nn.Module):
    
    def __init__(self, margin=0.5, mutual_flag = False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        dist = torch.zeros([n, n], dtype = torch.float32)
        for i in range(n):
            for j in range(n):
                x = inputs[i].unsqueeze(0)
                y = inputs[j].unsqueeze(0)
                dist[i][j] = (x-y).pow(2).sum() # Euclidean distance
                #dist[i][j] = F.pairwise_distance(x, y, 2)

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = 0
        for i in range(n):
            dist_ap = dist[i][mask[i]].max()
            dist_an = dist[i][mask[i] == 0].min()
            loss += F.relu(dist_ap - dist_an + self.margin)

        return loss
