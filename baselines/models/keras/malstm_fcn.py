import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM
from tensorflow.keras import Model

# todo : remove duplicates
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


class MALSTM_FCN(tf.keras.model):

    def __init__(self, n_channels, n_timesteps, n_classes):
        super(MALSTM_FCN, self).__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
        self.stride = 3

        self.conv1d_0 = Conv1D(self.n_channels / self.stride, 8, strides=self.stride, padding='same', activation='relu',
                               use_bias=False, kernel_initializer='he_uniform')  # (None, n_channels / stride, n_timestemps)

        self.conv1d_1 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')
        self.conv1d_2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')
        self.conv1d_3 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')

        # attention
        self.alstm = AttentionLSTM(384, unroll=True)
        self.dropout = Dropout(0.8)

        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.seb1 = SQUEEZE_EXCITE_BLOCK(n_channels=self.n_channels)
        self.seb2 = SQUEEZE_EXCITE_BLOCK(n_channels=self.n_channels)
        self.dense = Dense(self.n_classes)

    def call(self, inputs, training=False):
        # todo : check keras tensor shape (batch, timesteps, channels)
        assert inputs._keras_shape[1:] == (self.n_timesteps, self.n_channels)

        x = Permute((2, 1))(inputs)
        x = self.conv1d_0(x)

        x = Masking()(x)
        x = self.alstm(x)
        x = self.dropout(x)

        y = Permute((2, 1))(inputs)
        # todo : check whether or not dropout is available only when training step
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
