import numpy as np
from math import sqrt
import cv2

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    h,w,c = input_image.shape

    new_image = np.zeros((h, w, 4))
    # ligtness method
    new_image[:,:,0] = input_image.max(axis=2) + input_image.min(axis=2) / 2

    # average method
    new_image[:,:,1] = input_image.mean(axis=2)

    # luminosity method
    new_image[:,:,2] = np.average(input_image, weights=[0.21, 0.72, 0.07], axis=2)

    # built-in opencv function 

    new_image[:,:,3] = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space

    new_image = np.zeros(input_image.shape)
    #O_1 = (R - G) / sqrt(2)
    new_image[:,:,0] = (input_image[:,:,0] - input_image[:,:,1]) / sqrt(2)
    #O_2 = ((R + G) - 2B) / sqrt(6))
    new_image[:,:,1] = ((input_image[:,:,0] + input_image[:,:,1]) - 2 * input_image[:,:,2]) / sqrt(6)
    #O_3 = (R + G + B) / sqrt(3)
    new_image[:,:,2] = input_image.sum(axis=2) / sqrt(3)

    new_image = new_image.astype(np.float32)
    return new_image

def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    new_image = np.zeros(input_image.shape)
    rgb_sum = input_image.sum(axis=2)
    for i in range(3):
        new_image[:,:,i] = input_image[:,:,i] / rgb_sum
    new_image = new_image.astype(np.float32)

    return new_image
