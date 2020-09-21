import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import signal

#Working code
def compute_gradient(image):
    # load image as pixel array
    image = plt.imread(image)

    # get data of the image
    data_image=np.array(image)

    Gx=sp.signal.convolve2d(data_image,np.array([[1, 0, -1],[2,0,-2],[1,0,-1]]))
    Gy=sp.signal.convolve2d(data_image,np.array([[1,2, 1],[0,0,0],[-1,-2,-1]]))
    im_magnitude=np.sqrt((np.square(Gx)+(np.square(Gy))))
    im_direction=np.arctan(Gy/Gx)

    return Gx, Gy, im_magnitude,im_direction
