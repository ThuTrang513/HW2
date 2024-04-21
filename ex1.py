import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    # Need to implement here
    img_padded = np.zeros((np.shape(img)[0] + filter_size // 2 * 2, np.shape(img)[1] + filter_size // 2 * 2), dtype = np.float32)
    img_padded[filter_size//2:-filter_size//2 + 1, filter_size//2:-filter_size//2 + 1] = img
    img_padded[:filter_size//2, filter_size//2:-filter_size//2 + 1] = img[0]
    img_padded[-filter_size//2 + 1:, filter_size//2:-filter_size//2  + 1] = img[-1]
    img_padded[filter_size//2:-filter_size//2 + 1, :filter_size//2] = img[:, 0:1]
    img_padded[filter_size//2:-filter_size//2 + 1, -filter_size//2 + 1:] = img[:, -1:np.shape(img)[1]]

    img_padded[:filter_size//2, :filter_size//2] = img[0, 0]
    img_padded[:filter_size//2, -filter_size//2 + 1:] = img[0, -1]
    img_padded[-filter_size//2 + 1:, :filter_size//2] = img[-1, 0]
    img_padded[-filter_size//2 + 1:, -filter_size//2 + 1:] = img[-1, -1]
    return img_padded

def mean_filter(img, filter_size=3):
  # Need to implement here
    img_padded = padding_img(img, filter_size)
    img_filtered = np.zeros(np.shape(img), dtype = np.float32)
    for i in range(len(img_filtered)):
        for j in range(len(img_filtered[i])):
            img_filtered[i][j] = np.average(img_padded[i:i+filter_size, j:j+filter_size])
    return img_filtered

def median_filter(img, filter_size=3):
  # Need to implement here
    img_padded = padding_img(img, filter_size)
    img_filtered = np.zeros(np.shape(img), dtype = np.float32)
    for i in range(len(img_filtered)):
        for j in range(len(img_filtered[i])):
            img_filtered[i, j] = np.median(img_padded[i:i+filter_size, j:j+filter_size])
    return img_filtered


def psnr(gt_img, smooth_img):
    # Need to implement here
    gt_img = gt_img.astype(np.float32)
    mse_score = np.mean((gt_img - smooth_img) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse_score)



def show_res(before_img, after_img):
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()

if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    img_tg = read_img(img_gt)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img_tg, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img_tg, median_smoothed_img))

