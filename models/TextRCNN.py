import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self, max_len, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(TextRCNN, self).__init__()
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.maxpool = nn.MaxPool1d(max_len)
        self.fc = nn.Linear(hidden_dim * 2 + 3072, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """输入x的size就是：(batch_size, max_len, 3072)"""
        lstm_out, _ = self.bilstm(x) # torch.Size([64, 24, 256])
        total_x = torch.cat((lstm_out, x), dim=2) # torch.Size([64, 24, 256+3072])
        total_x = F.relu(total_x)
        total_x = total_x.permute(0,2,1) # torch.Size([64, 256+3072, 24])
        out = self.maxpool(total_x).squeeze() # torch.Size([64, 256+3072, 1]) -> torch.Size([64, 256+3072])
        outputs = self.fc(out) #  torch.Size([64, output_dim])
        # x = self.dropout(x)
        return outputs