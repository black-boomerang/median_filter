import sys
import time

import numpy as np


def trivial_median(image, filter_size):
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

    max_memory = 0
    for i in range(filter_height // 2, image_height - filter_height // 2):
        for j in range(filter_width // 2, image_width - filter_width // 2):
            values = image[i - filter_height // 2:i + filter_height // 2 + 1,
                     j - filter_width // 2:j + filter_width // 2 + 1].ravel()
            values.sort()
            max_memory = max(max_memory, sys.getsizeof(values))
            target_image[i - filter_height // 2, j - filter_width // 2] = values[filter_height * filter_width // 2]

    algorithm_time = time.time() - start_time

    return target_image, algorithm_time * 1e9 / target_image.size, max_memory
