import sys
import time
from collections import Counter

import numpy as np


def huang_etal_median(image, filter_size):
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

    th = filter_height * filter_width // 2
    max_memory = 0
    for i in range(filter_height // 2, image_height - filter_height // 2):
        hist = Counter()
        for j in range(i - filter_height // 2, i + filter_height // 2 + 1):
            for k in range(filter_width):
                hist[image[j][k]] += 1

        tmdn = 0
        mdn = 0
        while tmdn + hist[mdn] <= th:
            tmdn += hist[mdn]
            mdn += 1
        target_image[i - filter_height // 2, 0] = mdn

        for j in range(filter_width // 2 + 1, image_width - filter_width // 2):
            left = image[i - filter_height // 2:i + filter_height // 2 + 1, j - filter_width // 2 - 1]
            right = image[i - filter_height // 2:i + filter_height // 2 + 1, j + filter_width // 2]
            for k in range(filter_height):
                g1 = left[k]
                hist[g1] -= 1
                if g1 < mdn:
                    tmdn -= 1
                g1 = right[k]
                hist[g1] += 1
                if g1 < mdn:
                    tmdn += 1

            max_memory = max(max_memory, sys.getsizeof(hist) + sys.getsizeof(left), sys.getsizeof(right))

            if tmdn > th:
                while tmdn > th:
                    mdn -= 1
                    tmdn -= hist[mdn]
            else:
                while tmdn + hist[mdn] <= th:
                    tmdn += hist[mdn]
                    mdn += 1

            target_image[i - filter_height // 2, j - filter_width // 2] = mdn

    algorithm_time = time.time() - start_time

    return target_image, algorithm_time * 1e9 / target_image.size, max_memory
