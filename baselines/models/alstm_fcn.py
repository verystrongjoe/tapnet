import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
from tensorflow.keras import Model

class ALSTM_FCN(Model):

    def __init__(self, n_channels, n_timesteps, n_classes):
        super(ALSTM_FCN, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        # attention
        self.alstm = AttentionLSTM(8)

        self.dropout = Dropout(0.8)
        self.conv1d_1 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')
        self.conv1d_2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')
        self.conv1d_3 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')
        self.bn = BatchNormalization()
        self.relu = Activation('relu')

        self.dense = Dense(self.n_classes)

    def call(self, inputs, training=False):
        # todo : check keras tensor shape (batch, timesteps, channels)
        assert inputs._keras_shape[1:] == (self.n_timesteps, self.n_channels)

        x = Masking()(inputs)
        x = self.alstm(x)
        # todo : check whether or not dropout is available only when training step
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(inputs)
        y = self.conv1d_1(y)
        y = self.bn(y)
        y = self.relu(y)

        y = self.conv1d_2()(y)
        y = self.bn(y)
        y = self.relu(y)

        y = self.conv1d_3(y)
        y = self.bn(y)
        y = self.relu(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])
        out = self.dense(x)

        return out









