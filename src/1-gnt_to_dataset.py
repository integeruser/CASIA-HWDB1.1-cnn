#!/usr/bin/env python2
import sys

import h5py

import utils

if len(sys.argv) != 3:
    print 'Usage: %s trn_dirpath tst_dirpath' % sys.argv[0]
    sys.exit(1)

trn_dirpath = sys.argv[1]
tst_dirpath = sys.argv[2]

with h5py.File('HWDB1.1.hdf5', 'w') as f:
    for name, size, dirpath in [('trn', 897758, trn_dirpath), ('tst', 223991, tst_dirpath)]:
        print 'Converting \'%s\'...' % name

        grp = f.create_group(name)
        dset_bitmap  = grp.create_dataset('bitmap',  (size, 1, 64, 64), dtype='uint8')
        dset_tagcode = grp.create_dataset('tagcode', (size, 1),         dtype='uint16')

        for i, (bitmap, tagcode) in enumerate(utils.read_gnt_in_directory(dirpath)):
            dset_bitmap[i]  = utils.normalize_bitmap(bitmap)
            dset_tagcode[i] = tagcode
