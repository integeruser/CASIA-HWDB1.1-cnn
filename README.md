# CASIA-HWDB1.1-cnn
This repository contains our experiments with the deep learning library [Keras](http://keras.io/) and a subset of the [CASIA-HWDB1.1](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) data set. The code here provided implements a convolutional neural network with test set accuracy greater than 95%. The network configuration is a simplified version of the one described in [Deep Convolutional Network for Handwritten Chinese Character Recognition](http://yuhao.im/files/Zhang_CNNChar.pdf) by Yuhao Zhang. No attempt was made to improve the accuracy of the model since we just wanted to play with convolutional networks and deep learning libraries.

Alessandro and Francesco


## Requisites
We assume you have already installed and configured Keras. Install the other required dependencies with
```
$ pip2 install h5py
```
and the optional dependencies with
```
$ pip2 install Pillow scikit-image
```
The code was last tested on Keras 2.0.4 using Theano 0.9.0 as backend, h5py 2.7.0, Pillow 4.1.1 and scikit-image 0.13.0. Tests were run under a single GeForce GTX 970 using a floating-point precision of 32 bits.


## Usage
In the releases section we uploaded a (zipped) [subset](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases/download/v1.1/HWDB1.1subset.hdf5.zip) of the CASIA-HWDB1.1 data set, the [network model](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases/download/v1.1/model.json), [network weights](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases/download/v1.1/weights.hdf5) and [some classifications](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases/download/v1.1/results.html), all generated following the steps below. If you use our subset, start from step 3.

0. Download the CASIA-HWDB1.1 data set from the official locations and unzip [HWDB1.1trn_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip) (1873 MB) and [HWDB1.1tst_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip) (471 MB);
1. Convert the data set into the HDF5 binary data format (~20 min):
```
$ python2 1-gnt_to_dataset.py HWDB1.1trn_gnt/ HWDB1.1tst_gnt/
Converting 'trn'...
Converting 'tst'...
```
2. Extract from the HDF5 data set a subset of 200 classes of characters (~10 min):
```
$ python2 2-dataset_to_subset.py HWDB1.1.hdf5
Subsetting 'trn'...
Subsetting 'tst'...
```
3. Train the network on the subset (~15 min):
```
$ python2 3-train_subset.py HWDB1.1subset.hdf5
Using Theano backend.

Using gpu device 0: GeForce GTX 970 (CNMeM is disabled, cuDNN not available)
Train on 40000 samples, validate on 8020 samples
Epoch 1/15
40000/40000 [==============================] - 65s - loss: 3.5933 - acc: 0.2570 - val_loss: 1.0810 - val_acc: 0.7385
Epoch 2/15
40000/40000 [==============================] - 65s - loss: 1.0313 - acc: 0.7361 - val_loss: 0.5121 - val_acc: 0.8686
Epoch 3/15
40000/40000 [==============================] - 65s - loss: 0.5632 - acc: 0.8469 - val_loss: 0.3775 - val_acc: 0.8994
Epoch 4/15
40000/40000 [==============================] - 66s - loss: 0.3812 - acc: 0.8923 - val_loss: 0.3912 - val_acc: 0.8964
Epoch 5/15
40000/40000 [==============================] - 67s - loss: 0.2735 - acc: 0.9218 - val_loss: 0.2458 - val_acc: 0.9374
Epoch 6/15
40000/40000 [==============================] - 67s - loss: 0.2164 - acc: 0.9356 - val_loss: 0.2603 - val_acc: 0.9330
Epoch 7/15
40000/40000 [==============================] - 67s - loss: 0.1749 - acc: 0.9469 - val_loss: 0.2283 - val_acc: 0.9389
Epoch 8/15
40000/40000 [==============================] - 66s - loss: 0.1402 - acc: 0.9570 - val_loss: 0.2030 - val_acc: 0.9507
Epoch 9/15
40000/40000 [==============================] - 66s - loss: 0.1217 - acc: 0.9625 - val_loss: 0.2070 - val_acc: 0.9481
Epoch 10/15
40000/40000 [==============================] - 66s - loss: 0.0981 - acc: 0.9690 - val_loss: 0.2097 - val_acc: 0.9481
Epoch 11/15
40000/40000 [==============================] - 65s - loss: 0.0885 - acc: 0.9727 - val_loss: 0.2145 - val_acc: 0.9485
Epoch 12/15
40000/40000 [==============================] - 65s - loss: 0.0826 - acc: 0.9734 - val_loss: 0.2091 - val_acc: 0.9522
Epoch 13/15
40000/40000 [==============================] - 66s - loss: 0.0703 - acc: 0.9779 - val_loss: 0.1951 - val_acc: 0.9541
Epoch 14/15
40000/40000 [==============================] - 66s - loss: 0.0665 - acc: 0.9787 - val_loss: 0.2239 - val_acc: 0.9484
Epoch 15/15
40000/40000 [==============================] - 66s - loss: 0.0568 - acc: 0.9820 - val_loss: 0.2226 - val_acc: 0.9519
Test score: 0.189350152647
Test accuracy: 0.95839959825
```
4. (Optional) Generate a report of some classifications (~20 sec):
```
$ python2 4-draw_results.py HWDB1.1subset.hdf5 model.json weights.hdf5
Using Theano backend.

Using gpu device 0: GeForce GTX 970 (CNMeM is disabled, cuDNN not available)
Evaluating the network on the test set...
Test score: 0.189350152647
Test accuracy: 0.95839959825
Extracting some results...
```
