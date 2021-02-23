import numpy as np
from tsaug.visualization import plot
from utils import load_raw_ts
import matplotlib.pyplot as plt

path = "./data/"
dataset="NATOPS"
path = path + "raw/" + dataset + "/"

x_train = np.load(path + 'X_train.npy')  # (180, 51, 24)  (Batch, Length, Dimensions)  (N, T, C)
y_train = np.load(path + 'y_train.npy')  # (180, 1) (Batch, Class) (N, T, L)
# x_test = np.load(path + 'X_test.npy')
# y_test = np.load(path + 'y_test.npy')

# plot(x_train, y_train)

X = x_train[0:10, :, :]
plot(X)


from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse


my_augmenter = (
        # TimeWarp() * 5  # random time warping 5 times in parallel
        # + Crop(size=20)  # random crop subsequences with length 300
        # + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        + Reverse() @ 0.5  # with 50% probability, reverse the sequence
)
X_aug = my_augmenter.augment(X)

print(X.shape)
print(X_aug.shape)

plot(X_aug)
plt.show()
