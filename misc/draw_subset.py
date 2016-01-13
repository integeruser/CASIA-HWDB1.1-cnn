#!/usr/bin/env python2
# This script saves in a temporary folder the pictures of all the characters
# of the subset, grouped by tagcode
import os
import sys
import tempfile
from collections import defaultdict

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

if len(sys.argv) != 2:
    print 'Usage: %s subset_filepath' % sys.argv[0]
    sys.exit(1)

subset_filepath = sys.argv[1]

with h5py.File(subset_filepath, 'r') as f:
    dirpath = tempfile.mkdtemp()
    print 'Saving images in \'%s\'...' % dirpath

    os.chdir(dirpath)

    for name in ['trn', 'vld', 'tst']:
        label_to_indexes = defaultdict(list)
        for i in range(len(f[name+'/x'])):
            y = f[name+'/y'][i]
            label = np.argmax(y)
            label_to_indexes[label].append(i)

        os.mkdir('out-'+name)
        for label, indexes in label_to_indexes.iteritems():
            bitmap = np.array([]).reshape((0, 64*20))

            bitmaps = [np.squeeze(f[name+'/x'][index], axis=0) for index in indexes]
            for i in range(0, len(bitmaps), 20):
                chunk = bitmaps[i:i+20]
                row = np.hstack(chunk) if len(chunk) == 20 else np.hstack(chunk + [np.zeros((64, 64*(20-len(chunk))))])
                bitmap = np.vstack((bitmap, row))

            im = PIL.Image.fromarray(bitmap)
            im.convert('RGB').save(os.path.join('out-'+name, str(label)+'.png'))
