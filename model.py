import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 11), padding=(1,3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = ResBlock(32, 64, stride=(2,2))
        self.layer2 = ResBlock(64, 128, stride=(2,2))
        self.layer3 = ResBlock(128, 256, stride=(2,2))
        self.layer4 = ResBlock(256, 256, stride=(2,2))

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)
    
def load_model(weight_path, num_classes):
    model = ResNet(num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

GENRE_MAP = {
    0: "CCM",
    1: "EDM",
    2: "J-POP",
    3: "OST",
    4: "POP",
    5: "R&B/Soul",
    6: "UNKNOWN",
    7: "국악",
    8: "뉴에이지",
    9: "댄스",
    10: "랩/힙합",
    11: "록/메탈",
    12: "뮤직테라피",
    13: "발라드",
    14: "성인가요",
    15: "어린이/태교",
    16: "월드뮤직",
    17: "일렉트로니카",
    18: "일렉트로니카(스타일)",
    19: "재즈",
    20: "종교음악",
    21: "클래식",
    22: "포크/블루스",
}
