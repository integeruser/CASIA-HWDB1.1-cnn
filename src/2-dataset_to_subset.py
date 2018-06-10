#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from collections import defaultdict
from itertools import count

import h5py
import keras.utils.np_utils
import numpy as np

import utils

if len(sys.argv) != 2:
    print('Usage: %s dataset_filepath' % sys.argv[0])
    sys.exit(1)

dataset_filepath = sys.argv[1]

allowed = set(utils.unicode_to_tagcode(c) for c in ['谈','般','盏','坤','膀','脂','型','骏','童','挟','损','恋','婴','读','账','服','任','茸','张','亢','耀','涉','个','随','挂','抗','贞','瞥','瘤','作','河','欲','侵','吸','眺','线','捂','倾','牌','筒','渊','拥','话','赞','知','除','巩','惫','揭','扬','驼','绿','渔','榆','辊','应','儡','假','崩','抬','是','讲','刷','鸿','契','寒','录','教','也','艾','囤','秦','峨','括','诲','滴','凶','须','孽','巾','沉','餐','暂','蒙','攘','键','厄','的','芭','岳','惜','椰','足','伴','离','笼','临','胁','泉','晚','迟','汞','级','跳','轴','偶','啸','移','贾','老','节','蜗','堑','帕','肖','伟','渝','撮','臀','吉','汉','反','双','坏','翔','胖','绪','固','舀','再','咏','堂','尔','沟','符','涵','水','误','岿','所','摄','广','结','学','苫','臭','恬','诱','递','烷','硼','茁','标','越','吏','笑','馒','耗','氟','加','砧','稻','晃','臂','其','配','城','筑','痹','揖','江','连','卡','狠','瓤','乳','赵','仿','睹','相','好','屿','争','袭','王','吃','疏','粕','涟','垣','逢','锤','覆','薯','贴','冷','霸','聂','糕','占'])
assert len(allowed) == 200

counter = count()
tagcode_to_label = dict()
label_to_categorical = dict()

with h5py.File(dataset_filepath, 'r') as f1, h5py.File('HWDB1.1subset.hdf5', 'w') as f2:
    print('Subsetting \'trn\'...')
    # find allowed characters in the data set
    tagcode_to_count = defaultdict(int)
    indexes = list()
    for i in range(897758):
        tagcode = f1['trn/tagcode'][i][0]
        if tagcode in allowed:
            indexes.append((i, tagcode_to_count[tagcode] < 200))
            tagcode_to_count[tagcode] += 1

    # create the subsets
    trn_size = sum(1 if is_trn else 0 for index, is_trn in indexes)
    assert trn_size == 200*200
    trn_grp = f2.create_group('trn')
    trn_x = trn_grp.create_dataset('x', (trn_size, 1, 64, 64), dtype='uint8')
    trn_y = trn_grp.create_dataset('y', (trn_size, 200),       dtype='uint16')

    vld_size = len(indexes)-trn_size
    vld_grp = f2.create_group('vld')
    vld_x = vld_grp.create_dataset('x', (vld_size, 1, 64, 64), dtype='uint8')
    vld_y = vld_grp.create_dataset('y', (vld_size, 200),       dtype='uint16')

    # populate subsets
    trn_i = vld_i = 0
    for index, is_trn in indexes:
        bitmap, tagcode = f1['trn/bitmap'][index], f1['trn/tagcode'][index][0]

        # compute label and categorical for current tagcode
        if tagcode not in tagcode_to_label:
            tagcode_to_label[tagcode] = next(counter)
        label = tagcode_to_label[tagcode]
        if label not in label_to_categorical:
            label_to_categorical[label] = keras.utils.np_utils.to_categorical([label], 200).reshape(200)
            assert sum(label_to_categorical[label] == 1)

        if is_trn:
            trn_x[trn_i] = utils.preprocess_bitmap(bitmap)
            trn_y[trn_i] = label_to_categorical[label]
            trn_i += 1
        else:
            vld_x[vld_i] = utils.preprocess_bitmap(bitmap)
            vld_y[vld_i] = label_to_categorical[label]
            vld_i += 1
        tagcode_to_count[tagcode] += 1

    ############################################################################

    print('Subsetting \'tst\'...')
    # find allowed characters in the data set
    indexes = [i for i in range(223991) if f1['tst/tagcode'][i][0] in allowed]
    new_size = len(indexes)

    # create the subset
    tst_grp = f2.create_group('tst')
    tst_x = tst_grp.create_dataset('x', (new_size, 1, 64, 64), dtype='uint8')
    tst_y = tst_grp.create_dataset('y', (new_size, 200),       dtype='uint16')
    tst_t = tst_grp.create_dataset('t', (new_size, 1),         dtype='uint16')

    # populate subset
    for i, index in enumerate(indexes):
        bitmap, tagcode = f1['tst/bitmap'][index], f1['tst/tagcode'][index][0]

        label = tagcode_to_label[tagcode]

        tst_x[i] = utils.preprocess_bitmap(bitmap)
        tst_y[i] = label_to_categorical[label]
        tst_t[i] = tagcode
