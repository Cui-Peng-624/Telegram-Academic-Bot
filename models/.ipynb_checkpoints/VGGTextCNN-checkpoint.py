import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGTextCNN(nn.Module):
    def __init__(self, num_classes):
        super(VGGTextCNN, self).__init__()
        
        # 第一层卷积块
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=(3, 3072))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1)) 
        
        # 第二层卷积块
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1)) 
        
        # 第三层卷积块
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 1))
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 1))
        # self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)) # 定义一个MaxPool2d层，kernel_size为(2, 1)，stride为(1, 1)
        
        # 全连接层
        self.fc1 = nn.Linear(128*3, 128*3*4)
        self.fc2 = nn.Linear(128*3*4, 128*3)
        self.fc3 = nn.Linear(128*3, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状：(batch_size, 1, max_len, 768)
        x = F.relu(self.conv1_1(x))
        # print(x.shape)
        x = F.relu(self.conv1_2(x))
        # print(x.shape)
        x = self.pool1(x)
        # print("1:", x.shape)

        x = F.relu(self.conv2_1(x))
        # print(x.shape)
        x = F.relu(self.conv2_2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print("2:", x.shape)

        # x = F.relu(self.conv3_1(x))
        # print(x.shape)
        # x = F.relu(self.conv3_2(x))
        # print(x.shape)
        # x = self.pool3(x)
        # print("3:", x.shape)

        x = x.view(x.size(0), -1)
        # print("Flatten:", x.shape)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x