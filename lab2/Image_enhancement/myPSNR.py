import numpy as np
import cv2

def myPSNR( orig_image, approx_image ):
    h,w = orig_image.shape
    mse = 0
    
    for y in range(h):
        for x in range(w):
            mse += (orig_image[y,x] - approx_image[y,x]) ** 2
    mse = mse / (h*w)

    I_max = orig_image.max()

    PSNR = 20 * np.log10(I_max / np.sqrt(mse))

    return PSNR


if __name__ == "__main__":
    orig_image = cv2.imread("images/image1.jpg", 0) / 255
    salt_pepper = cv2.imread("images/image1_saltpepper.jpg", 0) / 255
    gaussian = cv2.imread("images/image1_gaussian.jpg", 0) / 255
    salt_pepper_psnr = myPSNR(orig_image, salt_pepper)
    gaussian_psnr = myPSNR(orig_image, gaussian)
    print("PSNR score between image1.jpg and image1_saltpepper.jpg: ", salt_pepper_psnr)
    print("PSNR score between image1.jpg and image1_gaussian.jpg: ", gaussian_psnr)