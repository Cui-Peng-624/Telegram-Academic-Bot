import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(TextRCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.conv = nn.Conv2d(1, 100, (3, embedding_dim))  # 卷积层 - 一个输入通道，100个输出通道，卷积核的size：(3,embedding_dim)
        self.maxpool = nn.MaxPool2d((2, 1))  # 最大池化层
        self.fc = nn.Linear(hidden_dim * 2 + 100, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)

        x = x.unsqueeze(1)
        conv_out = F.relu(self.conv(x)).squeeze(3)
        conv_out = self.maxpool(conv_out).squeeze(2)

        out = torch.cat((lstm_out, conv_out), dim=1)
        dense_outputs = self.fc(out)
        outputs = self.softmax(dense_outputs)
        return outputs