# CASIA-HWDB1.1-cnn
This repository contains our experiments with the deep learning library [Keras](http://keras.io/) and a subset of the [CASIA-HWDB1.1](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) data set. The code here provided implements a convolutional neural network with test set accuracy greater than 94%. The network configuration is a simplified version of the one described in [Deep Convolutional Network for Handwritten Chinese Character Recognition](http://cs231n.stanford.edu/reports/zyh_project.pdf) by Yuhao Zhang. No attempt was made to improve the accuracy of the network since we just wanted to play with convolutional networks and deep learning libraries.

Alessandro and Francesco


## Prerequisites
The code uses several Python libraries like Keras and [Theano](http://deeplearning.net/software/theano/). Unfortunately, it is infeasible to run the code on a CPU, and **you must configure Theano to use your GPU**. Install the required dependencies with:
```
pip2 install git+git://github.com/Theano/Theano.git
pip2 install keras h5py pillow scikit-image
```
We tested our code under Theano `0.7.0` and Keras `0.3.1`. Note that we installed Theano directly from git since the released version (at the time of writing) didn't contain the `relu` activation function needed by our convolutional network.

As already stated, we will use a subset of the CASIA-HWDB1.1 data set. Download from the official locations and unzip [HWDB1.1trn_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip) (1873 MB) and [HWDB1.1tst_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip) (471 MB).


## Usage
Our tests were run under a single GTX 970.

In the [Releases](https://github.com/integeruser/CASIA-HWDB1.1-cnn/releases) section we provide the (zipped) subset, the network configuration and the network weights generated in the following steps.

Convert the downloaded data set into the more convenient HDF5 binary data format (~20 min):
```
~ ➤ python2 1-gnt_to_dataset.py HWDB1.1trn_gnt HWDB1.1tst_gnt
Converting 'trn'...
Converting 'tst'...
```

Extract from the data set 200 classes of characters (starting from 3755) (~10 min):
```
~ ➤ python2 2-dataset_to_subset.py HWDB1.1.hdf5
Subsetting 'trn'...
Subsetting 'tst'...
```

Train the network using the subset (~25 min):
```
~ ➤ python2 3-train_subset.py HWDB1.1subset.hdf5
Using Theano backend.
Using gpu device 0: GeForce GTX 970 (CNMeM is disabled)

Train on 40000 samples, validate on 8020 samples
Epoch 1/15
40000/40000 [==============================] - 115s - loss: 4.1788 - acc: 0.1709 - val_loss: 2.0648 - val_acc: 0.4894
Epoch 2/15
40000/40000 [==============================] - 115s - loss: 1.4478 - acc: 0.6390 - val_loss: 0.7809 - val_acc: 0.7959
Epoch 3/15
40000/40000 [==============================] - 115s - loss: 0.7354 - acc: 0.8032 - val_loss: 0.4175 - val_acc: 0.8888
Epoch 4/15
40000/40000 [==============================] - 115s - loss: 0.5097 - acc: 0.8593 - val_loss: 0.5074 - val_acc: 0.8672
Epoch 5/15
40000/40000 [==============================] - 115s - loss: 0.4238 - acc: 0.8812 - val_loss: 0.3405 - val_acc: 0.9045
Epoch 6/15
40000/40000 [==============================] - 115s - loss: 0.3645 - acc: 0.8980 - val_loss: 0.3127 - val_acc: 0.9135
Epoch 7/15
40000/40000 [==============================] - 115s - loss: 0.3443 - acc: 0.9099 - val_loss: 0.5292 - val_acc: 0.8608
Epoch 8/15
40000/40000 [==============================] - 115s - loss: 0.2915 - acc: 0.9172 - val_loss: 0.3327 - val_acc: 0.9182
Epoch 9/15
40000/40000 [==============================] - 115s - loss: 0.2474 - acc: 0.9287 - val_loss: 0.3143 - val_acc: 0.9137
Epoch 10/15
40000/40000 [==============================] - 116s - loss: 0.2480 - acc: 0.9302 - val_loss: 0.3448 - val_acc: 0.9132
Epoch 11/15
40000/40000 [==============================] - 115s - loss: 0.2054 - acc: 0.9401 - val_loss: 0.3582 - val_acc: 0.9140
Epoch 12/15
40000/40000 [==============================] - 115s - loss: 0.2075 - acc: 0.9396 - val_loss: 0.2929 - val_acc: 0.9284
Epoch 13/15
40000/40000 [==============================] - 115s - loss: 0.2132 - acc: 0.9423 - val_loss: 0.3578 - val_acc: 0.9196
Epoch 14/15
40000/40000 [==============================] - 115s - loss: 0.1955 - acc: 0.9430 - val_loss: 0.2828 - val_acc: 0.9308
Epoch 15/15
40000/40000 [==============================] - 115s - loss: 0.1970 - acc: 0.9456 - val_loss: 0.2772 - val_acc: 0.9390
Test score: 0.238051910567
Test accuracy: 0.945760441952
```

(Optional) Generate report of some classifications (~20 sec):
```
~ ➤ python2 4-draw_results.py HWDB1.1subset.hdf5 model.json weights.hdf5
Using Theano backend.
Using gpu device 0: GeForce GTX 970 (CNMeM is disabled)

Evaluating the network on the test set...
Test score: 0.238051910567
Test accuracy: 0.945760441952
Extracting some results...
```
Example [here](https://integeruser.github.io/misc/CASIA-HWDB1.1-cnn-results.html).


## License
The MIT License (MIT)

Copyright (c) 2016 Alessandro Torcinovich, Francesco Cagnin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
