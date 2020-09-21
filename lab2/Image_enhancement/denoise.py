import cv2
import matplotlib.pyplot as plt
from myPSNR import *

def denoise( image, kernel_type, **kwargs):
    if kernel_type == 'box':
        imOu = cv2.blur(image, kwargs['kernel_size'])
    elif kernel_type == 'median':
        imOu = cv2.medianBlur(image, kwargs['kernel_size'])
    elif kernel_type == 'gaussian':
        imOu = cv2.GaussianBlur(image, ksize=kwargs['kernel_size'],sigmaX=kwargs['sigma'],sigmaY=kwargs['sigma'])
    else:
        print('Operation Not implemented')
    return imOu

if __name__ == "__main__":
    image = cv2.imread('images/image1_gaussian.jpg', 0)
    orig = cv2.imread('images/image1.jpg', 0)
    method = 'box'
    kernel_sizes = [(3,3), (5,5), (7,7)]
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    for i, ksize in enumerate(kernel_sizes):
        denoised = denoise(image, method, kernel_size=ksize)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(str(ksize), fontsize=20)
        axs[i].imshow(denoised, cmap='gray')
        psnr = myPSNR(orig / 255, denoised / 255)
        print("psnr score of {}, with kernelsize {}: {}".format(method, ksize, psnr))
    plt.savefig(method+'gaussian.jpg')
    plt.show()
    
    method = 'gaussian'
    kernel_sizes = [(3,3), (5,5), (7,7)]
    plt.figure()
    stds = np.linspace(0.00001, 3, 5)
    kernel_sizes = [(3,3), (5,5), (7,7)]
    for ksize in kernel_sizes:
        for std in stds:
            denoised = denoise(image, method, kernel_size=ksize, sigma=std)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(str(ksize), fontsize=20)
            axs[i].imshow(denoised, cmap='gray')
            psnr = myPSNR(orig / 255, denoised / 255)
            print("psnr score of {}, with kernelsize {} and standard deviation {}: {}".format(method, ksize, std, psnr))

    method = 'gaussian'
    kernel_sizes = [(3,3), (5,5), (7,7)]
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    for i, ksize in enumerate(kernel_sizes):
        denoised = denoise(orig, method, kernel_size=ksize, sigma=5)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(str(ksize), fontsize=20)
        axs[i].imshow(denoised, cmap='gray')
        psnr = myPSNR(orig / 255, denoised / 255)
        print("psnr score of {}, with kernelsize {}: {}".format(method, ksize, psnr))
    plt.savefig(method+'gaussian.jpg')
    plt.show()

    method = "median"
    kernel_sizes = [3, 5, 7]
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    for i, ksize in enumerate(kernel_sizes):
        denoised = denoise(image, method, kernel_size=ksize)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(str(ksize), fontsize=20)
        axs[i].imshow(denoised, cmap='gray')
        psnr = myPSNR(orig / 255, denoised / 255)
        print("psnr score of {}, with kernelsize {}: {}".format(method, ksize, psnr))
    plt.savefig(method+'saltpepper.jpg')
    plt.show()
        

           