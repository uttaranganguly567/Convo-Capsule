import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    """
    Angular Additive Margin Softmax (ArcFace) Loss.
    
    This loss function helps the model learn embeddings that are:
    1. Close to the center of their class (Speaker Identity)
    2. Far away from other classes
    
    It does this by manipulating the angle between the embedding vector and the class weight vector.
    """
    def __init__(self, in_features, n_classes, s=30, m=0.5):
        super(AAMSoftmax, self).__init__()
        self.in_features = in_features # Embedding Size (e.g. 192)
        self.n_classes = n_classes     # Number of Speakers
        self.s = s                     # Scale factor
        self.m = m                     # Margin
        
        # The weight matrix essentially holds the "center" for each speaker
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # 1. Normalize Features and Weights
        # x shape: (batch, in_features)
        # weight shape: (n_classes, in_features)
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cosine shape: (batch, n_classes)
        
        # 2. Add Margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Robustness checks (keep phi stable)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 3. Create One Hot encoding to only apply margin to the Correct Class
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 4. Scale
        output *= self.s
        return output
