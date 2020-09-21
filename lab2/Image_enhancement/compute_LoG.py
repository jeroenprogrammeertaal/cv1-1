import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import image
import matplotlib.pyplot as plt
from Image_enhancement.gauss1D import gauss1D
from Image_enhancement.gauss2D import gauss2D

def magn(x,y):
    return x**2+y**2

# Creating a LoG kernel
def log_kernel(sigma, kernel_size):
    G = np.zeros((kernel_size, kernel_size)) #Setting up G

    if kernel_size % 2 == 0: #Error for even kernel size
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    
    x1 = np.arange(-(kernel_size//2), kernel_size//2+1, 1)
    x2 = x1.reshape((kernel_size,1))
    x = (magn(x1,x2)-sigma**2)/(sigma**4)
    G = x*gauss2D(sigma, kernel_size) #non-normalised 
    return G


def compute_LoG(imageName, LOG_type):

    data_image=np.array(image.imread(imageName))

    if LOG_type == 1:
        #method 1
        Gaussian_Kernel=gauss2D(0.5,5)
        imInterm=sp.signal.convolve2d(data_image,Gaussian_Kernel)
        imOut=sp.signal.convolve2d(imInterm,np.array([[0,1,0],[1,-4,1],[0,1,0]]), mode="same")

    elif LOG_type == 2:
        #method 2
        imOut=sp.signal.convolve2d(data_image,np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]]), mode="same")#log_kernel(0.5,5))

    elif LOG_type == 3:
        #method 3
        sigma1=0.8 #Using a ratio of 1.6
        sigma2=0.5
        kernel=(gauss2D(sigma1,5) - gauss2D(sigma2,5))
        imOut=sp.signal.convolve2d(data_image,kernel, mode="same")

    return imOut