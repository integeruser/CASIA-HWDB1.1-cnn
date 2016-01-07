import os
import struct

import numpy as np
import scipy.misc
import skimage.exposure


def read_gnt_in_directory(gnt_dirpath):
    def samples(f):
        header_size = 10

        # read samples from f until no bytes remaining
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break

            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            assert header_size + width*height == sample_size

            bitmap = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield bitmap, tagcode

    for file_name in os.listdir(gnt_dirpath):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples(f):
                    yield bitmap, tagcode


def normalize_bitmap(bitmap):
    # pad the bitmap to make it squared
    pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
    if bitmap.shape[0] < bitmap.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    bitmap = scipy.misc.imresize(bitmap, (64 - 4*2, 64 - 4*2))
    bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert bitmap.shape == (64, 64)

    bitmap = np.expand_dims(bitmap, axis=0)
    assert bitmap.shape == (1, 64, 64)
    return bitmap

def preprocess_bitmap(bitmap):
    # contrast stretching
    p2, p98 = np.percentile(bitmap, (2, 98))
    assert abs(p2-p98) > 10
    bitmap = skimage.exposure.rescale_intensity(bitmap, in_range=(p2, p98))

    # from skimage.filters import threshold_otsu
    # thresh = threshold_otsu(bitmap)
    # bitmap = bitmap > thresh
    return bitmap


def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')

def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]
