import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pathlib
import numpy as np


class SQUEEZE_EXCITE_BLOCK(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SQUEEZE_EXCITE_BLOCK, self).__init__()
        self.n_channels = n_channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # batch, channel, _, _
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FCN_model(nn.Module):

    def __init__(self, n_timesteps, n_classes, batch_size, n_channels, n_lstm_out=128, n_lstm_layers=1,
                 Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP=0.8, FC_DropP=0.3, SEB=True):

        super(FCN_model, self).__init__()

        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_lstm_out = n_lstm_out
        self.n_lstm_layers = n_lstm_layers

        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF

        self.lstm = nn.LSTM(self.n_channels, self.n_lstm_out, self.n_lstm_layers)  # (seq_len, batch, input_size)  -> (time_steps, batch, input_size(channels)
        self.C1 = nn.Conv1d(self.n_channels, self.Conv1_NF, 8)  # kernel size 8
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5) # kernel size 5
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3) # kernel size 3

        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.n_lstm_out, self.n_classes)
        self.SEB = SEB

        if self.SEB:
            self.SEB1 = SQUEEZE_EXCITE_BLOCK(n_channels=Conv1_NF)
            self.SEB2 = SQUEEZE_EXCITE_BLOCK(n_channels=Conv2_NF)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        cuda = torch.device('cuda')
        h0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.n_lstm_out).to(cuda)
        c0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.n_lstm_out).to(cuda)
        return h0, c0

    def forward(self, x):
        # input x [B,T,F] , where B = Batch size T = Time sampels  F = features
        h0, c0 = self.init_hidden()
        xx = torch.transpose(x, 1, 0)
        xx = torch.transpose(xx, 0, 2)

        # RuntimeError: Given groups=1, weight of size [128, 24, 8], expected input[360, 51, 24] to have 24 channels, but got 51 channels instead
        # (seq_len, batch, input_size)  (51, 360, 24) h0 (1, 360, hidden_size) (1, 360, hidden_suize)
        x1, (ht, ct) = self.lstm(xx, (h0, c0))
        # (seq_len, batch, num_directions * hidden_size)
        x1 = x1[-1, :, :]

        if self.SEB:
            x2 = self.C1(x)
            x2 = self.BN1(x2)
            x2 = self.relu(x2)
            x2 = self.ConvDrop(x2)
            x2 = self.SEB1(x2)
            # todo : spilt this layers like what I did above
            x2 = self.SEB2(self.ConvDrop(self.relu(self.BN2(self.C2(x2)))))

        else:
            x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x))))
            x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))

        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)
        x_out = self.softmax(x_out)
        return x_out
