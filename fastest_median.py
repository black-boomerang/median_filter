import sys
import time
from collections import Counter

import numpy as np


def fastest_median(image, filter_size):
    '''
    :param image: исходное серое изображение с 1 каналом
    :param filter_size: размер фильтра
    :return: изображение после применения медианного фильтра
    '''
    start_time = time.time()
    target_image = image.copy()
    if isinstance(filter_size, int):
        filter_height = filter_width = filter_size
    else:
        filter_height, filter_width = filter_size
    image = np.pad(image, ((filter_height // 2, filter_height // 2), (filter_width // 2, filter_width // 2)), 'edge')
    image_height, image_width = image.shape

    image_pad = np.pad(image, ((1, 0), (0, 0)), 'constant', constant_values=0)
    th = filter_height * filter_width // 2
    hist = Counter()
    col_hist = [Counter() for i in range(image_width)]
    for j in range(filter_height):
        for k in range(filter_width):
            hist[image_pad[j][k]] += 1
        for k in range(image_width):
            col_hist[k][image_pad[j][k]] += 1

    hist_start = hist
    max_memory = 0
    for i in range(filter_height // 2 + 1, image_height - filter_height // 2 + 1):
        hist = hist_start
        for j in range(filter_width):
            g1 = image_pad[i - filter_height // 2 - 1][j]
            hist[g1] -= 1
            col_hist[j][g1] -= 1
            g1 = image_pad[i + filter_height // 2][j]
            hist[g1] += 1
            col_hist[j][g1] += 1
        hist_start = hist

        tmdn = 0
        c_keys = sorted(list(hist.keys()))
        mdn_i = 0
        while tmdn + hist[c_keys[mdn_i]] <= th:
            tmdn += hist[c_keys[mdn_i]]
            mdn_i += 1
        target_image[i - filter_height // 2 - 1, 0] = c_keys[mdn_i]

        for j in range(filter_width // 2 + 1, image_width - filter_width // 2):
            col_hist[j + filter_width // 2][image_pad[i - filter_height // 2 - 1][j + filter_width // 2]] -= 1
            col_hist[j + filter_width // 2][image_pad[i + filter_height // 2][j + filter_width // 2]] += 1
            hist = hist + col_hist[j + filter_width // 2] - col_hist[j - filter_width // 2 - 1]

            max_memory = max(max_memory, sys.getsizeof(hist) +
                             sum([sys.getsizeof(col_hist[i]) for i in range(image_width)]))

            tmdn = 0
            c_keys = sorted(list(hist.keys()))
            mdn_i = 0
            while tmdn + hist[c_keys[mdn_i]] <= th:
                tmdn += hist[c_keys[mdn_i]]
                mdn_i += 1
            target_image[i - filter_height // 2 - 1, j - filter_width // 2] = c_keys[mdn_i]

    algorithm_time = time.time() - start_time

    return target_image, algorithm_time * 1e9 / target_image.size, max_memory
