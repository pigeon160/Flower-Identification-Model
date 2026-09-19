"""
构建CNN
"""
import torch.nn as nn
class Mymodel(nn.Module):
  def __init__(self):
    super(Mymodel, self).__init__()

    #卷积
    self.conv1 = nn.Conv2d(3, 96,  kernel_size = 11, stride = 4, padding = 2)
    self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2)
    self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1)
    self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1)
    self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1)

    # 激活
    self.relu = nn.ReLU(inplace = True)
    self.bn1 = nn.BatchNorm2d(96)
    self.bn2 = nn.BatchNorm2d(256)
    self.bn3 = nn.BatchNorm2d(384)
    self.bn4 = nn.BatchNorm2d(384)
    self.bn5 = nn.BatchNorm2d(256)

    # 池化
    self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)

    # 全连接
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(0.5)
    self.fc6 = nn.Linear(256 * 6 * 6, 4096)
    self.fc7 = nn.Linear(4096, 4096)
    self.fc8 = nn.Linear(4096, 5)

    # 损失函数
    self.criterion = nn.CrossEntropyLoss()
  def forward(self, x) :

    # 卷积部分，含卷积+激活+可能的池化
    x = self.pool(self.relu(self.bn1(self.conv1(x))))
    x = self.pool(self.relu(self.bn2(self.conv2(x))))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.pool(self.relu(self.bn5(self.conv5(x))))

    # 全连接部分
    x = self.flatten(x)
    x = self.dropout(self.relu(self.fc6(x)))
    x = self.dropout(self.relu(self.fc7(x)))
    return self.fc8(x)