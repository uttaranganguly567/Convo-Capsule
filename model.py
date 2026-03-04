import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(planes // scale)
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = int((kernel_size - 1) * dilation / 2)
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        # Use Conv1d with kernel_size=1 as a replacement for Linear layers on 3D tensors (batch, channels, time)
        self.match = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.activation = nn.Tanh()
        self.assign = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)
        
    def forward(self, x):
        # x is (batch, channels, time)
        # Attention score calculation
        w = self.match(x)
        w = self.activation(w)
        w = self.assign(w)  # (batch, channels, time)
        
        # Apply softmax over time dimension to get attention weights
        alpha = F.softmax(w, dim=2)
        
        # Weighted mean
        mean = torch.sum(alpha * x, dim=2)
        
        # Weighted standard deviation
        residuals = x.unsqueeze(3) - mean.unsqueeze(2).unsqueeze(3) # This expansion might be tricky, let's do simplistic
        # Simpler weighted calc for std: E[x^2] - E[x]^2
        # But for attention pooling usually:
        # \mu = \sum \alpha_t x_t
        # \sigma = \sqrt{ \sum \alpha_t (x_t - \mu)^2 }
        
        # Vectorized weighted variance
        # (batch, channels, time)
        mean_expanded = mean.unsqueeze(2)
        residuals = x - mean_expanded
        weighted_sq_residuals = alpha * (residuals ** 2)
        variance = torch.sum(weighted_sq_residuals, dim=2)
        std = torch.sqrt(torch.clamp(variance, min=1e-9))
        
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, C=512):
        super(ECAPA_TDNN, self).__init__()
        # Front-end: Conv1D
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        
        self.attention = AttentiveStatsPool(1536)
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192) # Final Embedding Dimension = 192
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        # Input x: (Batch, Mels, Time) -> e.g. (B, 80, 200)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]
        
        # Pooling
        x = self.attention(x)
        x = self.bn5(x)
        
        # Embedding
        x = self.fc6(x)
        x = self.bn6(x)

        return x
