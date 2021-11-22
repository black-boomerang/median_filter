import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from fastest_median import fastest_median
from huang_etal_median import huang_etal_median
from trivial_median import trivial_median

if __name__ == '__main__':
    time_trivial = np.zeros(49)
    time_huang = np.zeros(49)
    time_const = np.zeros(49)
    time_cv = np.zeros(49)
    mem_trivial = np.zeros(49)
    mem_huang = np.zeros(49)
    mem_const = np.zeros(49)
    correct_algo = 0
    for filter_size in range(3, 100, 2):
        original_image = cv2.imread('original.bmp', cv2.IMREAD_GRAYSCALE)
        image1, time_algo, mem = trivial_median(original_image, filter_size=filter_size)
        time_trivial[filter_size // 2 - 1] = time_algo
        mem_trivial[filter_size // 2 - 1] = mem
        image2, time_algo, mem = huang_etal_median(original_image, filter_size=filter_size)
        time_huang[filter_size // 2 - 1] = time_algo
        mem_huang[filter_size // 2 - 1] = mem
        image3, time_algo, mem = fastest_median(original_image, filter_size=filter_size)
        time_const[filter_size // 2 - 1] = time_algo
        mem_const[filter_size // 2 - 1] = mem

        start_time = time.time()
        image_cv = cv2.medianBlur(original_image, filter_size)
        algorithm_time = time.time() - start_time
        time_cv[filter_size // 2 - 1] = algorithm_time * 1e9 / image_cv.size

        correct_algo += np.allclose(image1, image2) and np.allclose(image1, image3) and np.allclose(image1, image_cv)

    if correct_algo == 49:
        print('Алгоритмы корректны')

    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(3, 100, 2), time_trivial, label='Тривиальный')
    plt.plot(np.arange(3, 100, 2), time_huang, label='Huang')
    plt.plot(np.arange(3, 100, 2), time_const, label='Константный')
    plt.plot(np.arange(3, 100, 2), time_cv, label='OpenCV')
    plt.title('Зависимость времени работы алгоритмов от размера фильтра')
    plt.xlabel('Размер фильтра')
    plt.ylabel('Время работы (мсек/мегапиксель)')
    plt.grid()
    plt.legend()
    plt.savefig('time.png')
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(3, 100, 2), mem_trivial, label='Тривиальный')
    plt.plot(np.arange(3, 100, 2), mem_huang, label='Huang')
    plt.plot(np.arange(3, 100, 2), mem_const, label='Константный')
    plt.title('Зависимость потребляемой памяти от размера фильтра')
    plt.xlabel('Размер фильтра')
    plt.ylabel('Потребляемая память (в байтах)')
    plt.grid()
    plt.legend()
    plt.savefig('memory.png')
    plt.show()
