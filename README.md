# CASIA-HWDB1.1-cnn
This repository contains our experiments with the deep learning library [Keras](http://keras.io/) and a subset of the [CASIA-HWDB1.1](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) data set. The code here provided implements a convolutional neural network with test set accuracy sometimes greater than 95%. The network configuration is a simplified version of the one described in [_Deep Convolutional Network for Handwritten Chinese Character Recognition_](http://yuhao.im/files/Zhang_CNNChar.pdf) by Yuhao Zhang; no attempt was made to improve the model since we just wanted to play with convolutional networks and deep learning libraries.

Alessandro and Francesco

## Requisites
We assume you have already installed and configured Keras with any of its backends. Install the other required dependencies with:
```
$ pip3 install -r requirements.txt
```
The code was last tested on Keras 2.1.5 using TensorFlow 1.8.0 as backend, h5py 2.7.1, numpy 1.14.2, Pillow 5.1.0, scikit-image 0.14.0 and scipy 1.0.0. Tests were run under a single GeForce GTX 970.

## Usage
In the [releases section](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases) we uploaded a (zipped) subset of the CASIA-HWDB1.1 data set, a trained model and some classifications, all generated following the steps below. If you use our subset, start from step 3.

0. Download the CASIA-HWDB1.1 data set from the official locations ([HWDB1.1trn_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip) (1873 MB) and [HWDB1.1tst_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip) (471 MB)) and unzip it (it is required to decompress an archive in ALZ format):
```
$ unzip HWDB1.1trn_gnt.zip
Archive:  HWDB1.1trn_gnt.zip
  inflating: HWDB1.1trn_gnt.alz
$ unalz HWDB1.1trn_gnt.alz
unalz v0.65 (2009/04/01)
Copyright(C) 2004-2009 by kippler@gmail.com (http://www.kipple.pe.kr)

Extract HWDB1.1trn_gnt.alz to .

unalziiiing : 1219-c.gnt (21547829bytes) ...........
unalziiiing : 1220-c.gnt (26773966bytes) ...........
[...]
$ mkdir HWDB1.1trn_gnt
$ mv *.gnt HWDB1.1trn_gnt/

$ unzip HWDB1.1tst_gnt.zip
Archive:  HWDB1.1tst_gnt.zip
  inflating: 1289-c.gnt
  inflating: 1290-c.gnt
  [...]
$ mkdir HWDB1.1tst_gnt
$ mv *.gnt HWDB1.1tst_gnt/
```
1. Convert the data set into the HDF5 binary data format:
```
$ python3 1-gnt_to_dataset.py HWDB1.1trn_gnt/ HWDB1.1tst_gnt/
Converting 'trn'...
Converting 'tst'...
```
2. Extract from the HDF5 data set a subset of 200 classes of characters:
```
$ python3 2-dataset_to_subset.py HWDB1.1.hdf5
Subsetting 'trn'...
Subsetting 'tst'...
```
3. Train the network on the subset:
```
$ python3 3-train_subset.py HWDB1.1subset.hdf5
Using TensorFlow backend.
. . .
2018-06-10 10:18:51.390792: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3608 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, p
ci bus id: 0000:02:00.0, compute capability: 5.2)
Train on 40000 samples, validate on 8020 samples
Epoch 1/15
40000/40000 [==============================] - 41s 1ms/step - loss: 4.5114 - acc: 0.0975 - val_loss: 2.2487 - val_a
cc: 0.4542
Epoch 2/15
40000/40000 [==============================] - 30s 761us/step - loss: 1.7218 - acc: 0.5745 - val_loss: 0.7780 - val
_acc: 0.8026
Epoch 3/15
40000/40000 [==============================] - 30s 762us/step - loss: 0.8618 - acc: 0.7704 - val_loss: 0.4930 - val
_acc: 0.8675
. . .
Epoch 14/15
40000/40000 [==============================] - 31s 763us/step - loss: 0.1265 - acc: 0.9607 - val_loss: 0.3307 - val_acc: 0.9181
Epoch 15/15
40000/40000 [==============================] - 30s 762us/step - loss: 0.1179 - acc: 0.9640 - val_loss: 0.3282 - val_acc: 0.9198
Test score: 0.32826207710256655
Test accuracy: 0.9214865656650205
```
4. (Optional) Generate a report of some classifications:
```
$ python3 4-draw_results.py HWDB1.1subset.hdf5 model-1528618732.json weights-1528618732-0.921487.hdf5
Using TensorFlow backend.
. . .
2018-06-10 10:33:17.881820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3608 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:02:00.0, compute capability: 5.2)
Evaluating the network on the test set...
Test score: 0.32826207710256655
Test accuracy: 0.9214865656650205
Extracting some results...
```
