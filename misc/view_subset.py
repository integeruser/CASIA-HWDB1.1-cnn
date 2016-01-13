#!/usr/bin/env python2
# This script is useful to check if the CASIA HWDB1.1 subset was created correctly
import random
import sys

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
    print 'Usage: %s subset_filepath' % sys.argv[0]
    exit()

subset_filepath = sys.argv[1]

with h5py.File(subset_filepath, 'r') as f:
    while True:
        dset = random.choice(['trn', 'tst'])
        i = random.randint(0, len(f[dset+'/x']))

        bitmap = f[dset+'/x'][i]
        print bitmap
        print bitmap.shape
        print np.mean(bitmap)
        assert sum(f[dset+'/y'][i]) == 1

        plt.imshow(np.squeeze(bitmap, axis=0), cmap=cm.Greys_r)
        plt.show()
