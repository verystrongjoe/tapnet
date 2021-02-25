import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import euclidean_dist, normalize, output_conv_size, dump_embedding
import numpy as np

class SQUEEZE_EXCITE_BLOCK(nn.Module):
    def __init__(self, n_channels):
        self.filters = n_channels
        self.gap = nn.AvgPool1d()
        self.dense1 = nn.Linear(self.filters // 16, bias=False)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        self.dense2 = nn.Linear(self.filters, bias=False)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.gap(x)
        x = torch.reshape((1, self.filters))(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.multiply([input,x])
        return x




class SQUEEZE_EXCITE_BLOCK(Model):
    def __init__(self, n_channels):
        self.filters = n_channels
        self.gap = GlobalAveragePooling1D()
        self.dense1 = Dense(self.filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.dense2 = Dense(self.filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, input, training=False):
        # todo : check input keras shape == (n_batch, n_timesteps , n_channels)
        assert self.input._keras_shape[-1] == self.filters

        x = self.gap(input)
        x = Reshape((1, self.filters))(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = multiply([input,x])
        return x

class MLSTM_FCN(tf.keras.model):

    def __init__(self, n_channels, n_timesteps, n_classes):
        super(MLSTM_FCN, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        self.lstm = LSTM(8)
        self.dropout = Dropout(0.8)
        self.conv1d_1 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')
        self.conv1d_2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')
        self.conv1d_3 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.seb1 = SQUEEZE_EXCITE_BLOCK(n_channels=self.n_channels)
        self.seb2 = SQUEEZE_EXCITE_BLOCK(n_channels=self.n_channels)
        self.dense = Dense(self.n_classes)

    def call(self, inputs, training=False):

        # todo : check keras tensor shape (batch, timesteps, channels)
        assert inputs._keras_shape[1:] ==  (self.n_timesteps, self.n_channels)

        x = Masking()(inputs)
        x = self.lstm(x)
        # todo : check whether or not dropout is available only when training step 
        x = Dropout(0.8)(x)
        y = Permute((2,1))(inputs)
        y = self.conv1d_1(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.seb1(y)

        y = self.conv1d_2()(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.seb2(y)

        y = self.conv1d_3(y)
        y = self.bn(y)
        y = self.relu(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x,y])
        out = self.dense(x)

        return out









