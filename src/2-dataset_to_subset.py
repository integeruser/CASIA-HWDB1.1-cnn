#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
from collections import defaultdict
from itertools import count

import h5py
import keras.utils.np_utils
import numpy as np

import misc.utils

if len(sys.argv) != 2:
    print 'Usage: %s dataset_filepath' % sys.argv[0]
    sys.exit(1)

dataset_filepath = sys.argv[1]

allowed = set(misc.utils.unicode_to_tagcode(c) for c in [u'谈',u'般',u'盏',u'坤',u'膀',u'脂',u'型',u'骏',u'童',u'挟',u'损',u'恋',u'婴',u'读',u'账',u'服',u'任',u'茸',u'张',u'亢',u'耀',u'涉',u'个',u'随',u'挂',u'抗',u'贞',u'瞥',u'瘤',u'作',u'河',u'欲',u'侵',u'吸',u'眺',u'线',u'捂',u'倾',u'牌',u'筒',u'渊',u'拥',u'话',u'赞',u'知',u'除',u'巩',u'惫',u'揭',u'扬',u'驼',u'绿',u'渔',u'榆',u'辊',u'应',u'儡',u'假',u'崩',u'抬',u'是',u'讲',u'刷',u'鸿',u'契',u'寒',u'录',u'教',u'也',u'艾',u'囤',u'秦',u'峨',u'括',u'诲',u'滴',u'凶',u'须',u'孽',u'巾',u'沉',u'餐',u'暂',u'蒙',u'攘',u'键',u'厄',u'的',u'芭',u'岳',u'惜',u'椰',u'足',u'伴',u'离',u'笼',u'临',u'胁',u'泉',u'晚',u'迟',u'汞',u'级',u'跳',u'轴',u'偶',u'啸',u'移',u'贾',u'老',u'节',u'蜗',u'堑',u'帕',u'肖',u'伟',u'渝',u'撮',u'臀',u'吉',u'汉',u'反',u'双',u'坏',u'翔',u'胖',u'绪',u'固',u'舀',u'再',u'咏',u'堂',u'尔',u'沟',u'符',u'涵',u'水',u'误',u'岿',u'所',u'摄',u'广',u'结',u'学',u'苫',u'臭',u'恬',u'诱',u'递',u'烷',u'硼',u'茁',u'标',u'越',u'吏',u'笑',u'馒',u'耗',u'氟',u'加',u'砧',u'稻',u'晃',u'臂',u'其',u'配',u'城',u'筑',u'痹',u'揖',u'江',u'连',u'卡',u'狠',u'瓤',u'乳',u'赵',u'仿',u'睹',u'相',u'好',u'屿',u'争',u'袭',u'王',u'吃',u'疏',u'粕',u'涟',u'垣',u'逢',u'锤',u'覆',u'薯',u'贴',u'冷',u'霸',u'聂',u'糕',u'占'])
assert len(allowed) == 200

counter = count()
tagcode_to_label = dict()
label_to_categorical = dict()

with h5py.File(dataset_filepath, 'r') as f1, h5py.File('HWDB1.1subset.hdf5', 'w') as f2:
    print 'Subsetting \'trn\'...'
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
            trn_x[trn_i] = misc.utils.preprocess_bitmap(bitmap)
            trn_y[trn_i] = label_to_categorical[label]
            trn_i += 1
        else:
            vld_x[vld_i] = misc.utils.preprocess_bitmap(bitmap)
            vld_y[vld_i] = label_to_categorical[label]
            vld_i += 1
        tagcode_to_count[tagcode] += 1

    ############################################################################

    print 'Subsetting \'tst\'...'
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

        tst_x[i] = misc.utils.preprocess_bitmap(bitmap)
        tst_y[i] = label_to_categorical[label]
        tst_t[i] = tagcode
