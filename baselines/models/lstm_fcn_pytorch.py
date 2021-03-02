import pathlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class SQUEEZE_EXCITE_BLOCK(nn.Module):
    def __init__(self, n_channels):
        super(SQUEEZE_EXCITE_BLOCK, self).__init__()
        self.filters = n_channels
        self.gap = nn.AvgPool1d(kernel_size=3)
        self.dense1 = nn.Linear(self.filters, self.filters // 16, bias=False)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        self.dense2 = nn.Linear(self.filters // 16, self.filters, bias=False)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x2 = self.gap(x)
        x2 = torch.reshape((1, self.filters))(x)
        x2 = self.dense1(x)
        x2 = self.dense2(x)
        x = torch.multiply([x, x2])
        return x

class FCN_model(nn.Module):
    def __init__(self, NumClassesOut, N_time, N_Features, N_LSTM_Out=128, N_LSTM_layers=1,
                 Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP=0.8, FC_DropP=0.3, SEB=True):
        super(FCN_model, self).__init__()
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features, self.N_LSTM_Out, self.N_LSTM_layers)
        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.NumClassesOut)
        self.SEB = SEB
        if self.SEB:
            self.SEB1 = SQUEEZE_EXCITE_BLOCK(n_channels=N_Features)
            self.SEB2 = SQUEEZE_EXCITE_BLOCK(n_channels=N_Features)

    def init_hidden(self):
        cuda = torch.device('cuda')
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(cuda)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(cuda)
        return h0, c0


    def forward(self, x):

        x, labels, idx_train, idx_val, idx_test = x  # x is N * L, where L is the time-series feature dimension

        # input x [B,T,F] , where B = Batch size T = Time sampels  F = features
        h0, c0 = self.init_hidden()
        x1, (ht, ct) = self.lstm(x, (h0, c0))
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)

        if self.SEB:
            x2 = self.SEB1(self.ConvDrop(self.relu(self.BN1(self.C1(x2)))))
            x2 = self.SEB2(self.ConvDrop(self.relu(self.BN2(self.C2(x2)))))
        else:
            x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
            x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)
        return x_out