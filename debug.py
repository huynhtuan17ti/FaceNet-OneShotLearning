import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    a = [[1.2, 2, 3],
         [3, 4.5, 5],
         [3, 2, 2.3],
         [2, 3, 6]]

    b = [[1.5, 2, 3],
         [3, 4.5, 5],
         [3, 2, 2.3],
         [2, 3, 6]]

    b = torch.tensor(b)
    a = torch.tensor(a)
    
    dist = F.pairwise_distance(a, b)

    print(dist)