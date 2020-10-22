# TapNet


This is a Pytorch implementation of Attentional Prototype Network for the paper **TapNet: Multivariate Time Series Classification with Attentional Prototype Network** published in AAAI 2020.

## Run the demo

```bash
python train.py --dataset <DATASET>
```
You can find all the parameters we used in the file `run.sh`.

## Data

**[NEWS] You can download all the preprocessed data from [Google Drive](https://drive.google.com/file/d/1xZswfMeZuWovExsXh7U8T9uamlj6cQ85/view?usp=sharing).**

We use the latest multivariate time series classification dataset from [UAE archive](http://timeseriesclassification.com) with 30 datasets in wide range of applications.

The raw data is converted into npy data files in the following format:
* Training Samples: an N by M by L tensor (N is the training size of time series, M is the multivariate dimension, L is the length of time series),
* Train labels: an N by 1 vector (N is the training size of time series)
* Testing Samples: an N by M by L tensor (N is the testing size of time series, M is the multivariate dimension, L is the length of time series),
* Testing labels: an N by 1 vector (N is the testing size of time series)


You can specify a dataset as follows:

```bash
python train.py --dataset NATOPS
```

(or by editing `train.py`)

The default data is located at './data'.


## Paper

if you use our code in this repo, please cite our paper `\cite{zhang2020tapnet}`.

```
@inproceedings{zhang2020tapnet,
  title={TapNet: Multivariate Time Series Classification with Attentional Prototypical Network.},
  author={Zhang, Xuchao and Gao, Yifeng and Lin, Jessica and Lu, Chang-Tien},
  booktitle={AAAI},
  pages={6845--6852},
  year={2020}
}
```
