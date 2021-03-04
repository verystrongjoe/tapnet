"""
https://dzlab.github.io/timeseries/2018/11/25/LSTM-FCN-pytorch-part-1/
https://github.com/dzlab/deepprojects/blob/master/timeseries/LSTM_FCN_pytorch.ipynb
"""
import pathlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=256, drop_out=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_variables)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # shape(batch_size, num_variables, time_steps) e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        x, (h_n, c_n) = self.lstm(x)
        y = self.dropout(x)
        return y


class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99,
                 epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        y = self.relu(x)
        return y


class BlockFCN(nn.Module):

    def __init__(self, time_steps, channels=[24, 128, 256, 128], kernels=[8, 5, 3],
                 mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)

        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # apply Global Average Pooling 1D
        y= self.global_pooling(x)
        return y

# https://arxiv.org/pdf/1709.01507.pdf
# https://wwiiiii.tistory.com/entry/SqueezeandExcitation-Networks
class SQUEEZE_EXCITE_BLOCK(nn.Module):
    def __init__(self, n_channels):
        super(SQUEEZE_EXCITE_BLOCK, self).__init__()
        self.filters = n_channels
        self.gap = nn.AvgPool1d()
        self.dense1 = nn.Linear(self.filters // 16, bias=False)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        self.dense2 = nn.Linear(self.filters, bias=False)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x2 = self.gap(x)
        x2 = torch.reshape((1, self.filters))(x)
        x2 = self.dense1(x)
        x2 = self.dense2(x)
        x = torch.multiply([x, x2])
        return x


class LSTMFCN(nn.Module):

    def __init__(self, n_timesteps, n_channels=1, lstm_hs=256, channels=[1, 128, 256, 128], n_classes=2):
        super().__init__()
        self.lstm_block = BlockLSTM(n_timesteps, n_channels)
        self.fcn_block = BlockFCN(n_timesteps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, n_channels)
        self.softmax = nn.LogSoftmax(dim=n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        x1 = torch.squeeze(x1)

        # pass input through FCN block
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)

        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        x = self.dense(x)
        y = self.softmax(x)

        return y


class SimpleLearner():
    def __init__(self, data, model, loss_func, wd=1e-5):
        self.data, self.model, self.loss_func = data, model, loss_func
        self.wd = wd

    def update_manualgrd(self, x, y, lr):
        y_hat = self.model(x)

        # weight decay
        w2 = 0.

        for p in model.parameters():
            w2 += (p**2).sum()
        loss = self.loss_func(y_hat,  y) + w2 * self.wd
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.sub_(lr*p.grad)
                p.grad_zero_()

        return loss.item()

    def update(self, x, y, lr):
        opt = optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def fit(self, epochs=1, lr=1e-3):
        losses = []
        for i in tqdm(range(epochs)):
            for x,y in self.data[0]:
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
        return losses

    def evaluate(self, X):
        result = None
        for x, y in X:
            y_hat = self.model(x)
            y_hat = y_hat.cpu().detach().numpy()
            result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result


def one_hot_encode(input, labels):
    m = input.shape[0]
    output = np.zeros((m, labels), dtype=int)
    row_index = np.arange(m)
    output[row_index, input] = 1
    return output


def split_xy(data, classes):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    # hot encode
    #y = one_hot_encode(y, classes)
    return X, y


def create_dataset(X, y, device):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return TensorDataset(X_tensor, y_tensor)


def load_data(path, classes):
    data = np.loadtxt(path)
    return split_xy(data, classes)


if __name__ == '__main__':

    classes = 2
    print('start lstm-fcn in pytorch')

    # load training dataset
    X_train, y_train = load_data('data/Earthquakes_TRAIN.txt', classes)
    # load testing dataset
    X_test, y_test = load_data('data/Earthquakes_TEST.txt', classes)

    print('X_train %s   y_train %s' % (X_train.shape, y_train.shape))
    print('X_test  %s   y_test  %s' % (X_test.shape, y_test.shape))

    class_0_count = (y_train == 0).sum()
    class_1_count = (y_train == 1).sum()

    print(f'class_0_count : {class_0_count}, class_1_count : {class_1_count}')

    cuda = torch.device('cuda')

    train_ds = create_dataset(X_train, y_train, cuda)
    test_ds = create_dataset(X_test, y_test, cuda)

    class_sample_count = [class_0_count, class_1_count]
    weights = 1 / torch.Tensor(class_sample_count)

    samplers = torch.utils.data.sampler.WeightedRandomSampler(weights, 64)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    time_steps = X_train.shape[1]
    num_variables = classes

    model = LSTMFCN(time_steps, num_variables).cuda()

    loss_func = nn.NLLLoss().cuda()  # weight=weights

    learner = SimpleLearner([train_dl, test_dl], model, loss_func)
    losses = learner.fit(10)

    # for m in model.children():
    #     print(m.training)
    #     for j in m.children():
    #         print(j.training, j)

    plt.plot(losses)
    plt.show()

    y_pred = learner.evaluate(test_dl)
    print(f'mse : {((y_test - y_pred.argmax(axis=1)) ** 2).mean()}')

