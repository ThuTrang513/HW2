import numpy as np
from skimage import io as io_url
import matplotlib.pyplot as plt
import cv2



def DFT_slow(data):
  # You need to implement the DFT here
  N = np.shape(data)[0]
  matrix = np.zeros((N, N), dtype = np.complex64)
  for n in range(N):
      for s in range(N):
          matrix[n][s] = np.exp(-2j * np.pi * n * s / N)
  return np.dot(matrix, data)


def show_img(origin, row_fft, row_col_fft):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    axs[0].imshow(origin, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.log(np.abs(np.fft.fftshift(row_fft))), cmap='gray')
    axs[1].set_title('Row-wise FFT')
    axs[1].axis('off')
    axs[2].imshow((np.log(np.abs(np.fft.fftshift(row_col_fft)))), cmap='gray')
    axs[2].set_title('Column-wise FFT')
    axs[2].axis('off')
    plt.show()


def DFT_2D(gray_img):
    # You need to implement the DFT here
    H, W = np.shape(gray_img)
    row_fft = np.zeros((H, W), dtype = np.complex64)
    for h in range(H):
      row_fft[h] = DFT_slow(gray_img[h])
    row_col_fft = row_fft.T.copy()
    for w in range(W):
      row_col_fft[w] = DFT_slow(row_col_fft[w])
    row_col_fft = row_col_fft.T
    return row_fft, row_col_fft



if __name__ == '__main__':
  
    # ex1
    # x = np.random.random(1024)
    # print(np.allclose(DFT_slow(x), np.fft.fft(x)))
  # ex2
    img = io_url.imread('https://img2.zergnet.com/2309662_300.jpg')
    gray_img = np.mean(img, -1)
    row_fft, row_col_fft = DFT_2D(gray_img)
    show_img(gray_img, row_fft, row_col_fft)

 



