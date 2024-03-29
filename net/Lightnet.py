from torch import nn
import torch.nn.functional as F

class IluminationLightnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, (1, 1))

    def forward(self, X):
        return X + self.conv3(F.relu(self.conv2(F.relu(self.conv1(X)))))

class ReflectionLightnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, (3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 3, (1, 1))

    def forward(self, X):
        return X + self.conv6(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(X)))))))))))
        # return X + self.conv3(F.relu(self.conv2(F.relu(self.conv1(X)))))