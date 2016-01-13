#!/usr/bin/env python2
# This script counts the characters of the CASIA HWDB1.1 data set
import sys
from collections import Counter, defaultdict

import utils

if len(sys.argv) != 3:
    print 'Usage: %s trn_dirpath tst_dirpath' % sys.argv[0]
    exit()

trn_dirpath = sys.argv[1]
tst_dirpath = sys.argv[2]
frequencies = defaultdict(Counter)

for bitmap, tagcode in utils.read_gnt_in_directory(trn_dirpath):
    tagcode_unicode = utils.tagcode_to_unicode(tagcode)
    frequencies[tagcode_unicode].update(trn=1)
for bitmap, tagcode in utils.read_gnt_in_directory(tst_dirpath):
    tagcode_unicode = utils.tagcode_to_unicode(tagcode)
    frequencies[tagcode_unicode].update(tst=1)

with open('frequencies.txt', 'w') as f:
    for k, v in sorted(frequencies.iteritems(), key=lambda (k, v): v['trn'], reverse=True):
        f.write('%s: %d, %d\n' % (k.encode('utf-8'), v['trn'], v['tst']))
