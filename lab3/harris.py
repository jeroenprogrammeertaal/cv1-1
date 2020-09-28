import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, rotate
import cv2
import sys

class Image(object):
	def __init__(self, image, mode='grayscale'):
		if mode == 'grayscale':
			self.im = Image.RGB2Gray(image)
		else:
			self.im = image

		self.shape = self.im.shape

	@classmethod
	def readFrom(cls, p_im):
		im = plt.imread(p_im)
		return cls(im)
	
	@staticmethod
	def RGB2Gray(im):
		#factor = np.expand_dims(np.array([0.2989, 0.5870, 0.1140]).reshape(1,3),0)
		#img = np.sum(im*factor, axis=2)
		img = im[:,:,0]
		return img

	def gradient(self):
		imx = self.im.flatten()
		imy = self.im.flatten('F')
		G = np.array([-1, 0, 1])
		Gx = signal.convolve(imx, G, mode='same').reshape(self.im.shape)
		Gy = signal.convolve(imy, G, mode='same').reshape((self.im.shape[1], self.im.shape[0])).T
		return Gx, Gy


class HarrisCornerDetector(object):
	def __init__(self, window_size, threshold):
		self.window_size = window_size
		self.threshold = threshold



	def get_Hvalues(self, img, sigma=3):
		Ix, Iy = img.gradient()
		Ixx, Iyy, Ixy = Ix**2, Iy**2, np.multiply(Ix,Iy)
		#assert img.shape == Ix.shape == Iy.shape == Ixx.shape == Iyy.shape == Ixy.shape
		Hvalues = np.zeros(img.shape)
		Ixx_window = gaussian_filter(Ixx, sigma=sigma)
		Iyy_window = gaussian_filter(Iyy, sigma=sigma)
		Ixy_window = gaussian_filter(Ixy, sigma=sigma)
		H = (Iyy_window*Ixx_window - Ixy_window**2) - 0.04*(Ixx_window + Iyy_window)**2
		Hvalues= H
		return Hvalues


	def evaluate(self, H):
		h = int(self.window_size/2)
		mask = np.zeros((H.shape[0], H.shape[1]))
		for y in range(h, H.shape[0]-h):
			for x in range(h, H.shape[1]-h):
				val = np.max(H[y-h:y+h+1, x-h:x+h+1])
				if val == H[y,x] and val > self.threshold:
					mask[y,x] = 1
		return mask

def show_results(Ix, Iy, I, r, c, titles):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(Ix, cmap="gray")
    axs[0].set_title(titles[0])
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(Iy, cmap="gray")
    axs[1].set_title(titles[1])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].scatter(c, r, s=5)
    axs[2].imshow(I, cmap="gray")
    axs[2].set_title(titles[2])
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.show()

def show_results_2(p1, p2, p3, Images, titles):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].scatter(p1[1], p1[0], s=5)
    axs[0].imshow(Images[0], cmap="gray")
    axs[0].set_title(titles[0])
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].scatter(p2[1], p2[0], s=5)
    axs[1].imshow(Images[1], cmap="gray")
    axs[1].set_title(titles[1])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].scatter(p3[1], p3[0], s=5)
    axs[2].imshow(Images[2], cmap="gray")
    axs[2].set_title(titles[2])
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.show()

if __name__ == "__main__":
	
	#Img = Image.readFrom("person_toy/00000001.jpg")
	#Img45 = Image.readFrom("person_toy/00000001.jpg")
	#Img90 = Image.readFrom("person_toy/00000001.jpg")
	#Img45.im = rotate(Img45.im, 45)
	#Img90.im = rotate(Img90.im, 90)

	Img = Image.readFrom("pingpong/0000.jpeg")
	Img45 = Image.readFrom("pingpong/0000.jpeg")
	Img90 = Image.readFrom("pingpong/0000.jpeg")
	Img45.im = rotate(Img45.im, 45)
	Img90.im = rotate(Img90.im, 90)

	CornerAlgo1 = HarrisCornerDetector(3, 1e6)
	CornerAlgo2 = HarrisCornerDetector(3, 1e5)
	CornerAlgo3 = HarrisCornerDetector(3, 1e4)
	CornerAlgo4 = HarrisCornerDetector(3, 1e7)

	H1 = CornerAlgo1.get_Hvalues(Img, sigma=1)
	H2 = CornerAlgo2.get_Hvalues(Img, sigma=1)
	H3 = CornerAlgo3.get_Hvalues(Img, sigma=1)
	H4 = CornerAlgo4.get_Hvalues(Img, sigma=1)
	Hr45 = CornerAlgo1.get_Hvalues(Img45, sigma=1)
	Hr90 = CornerAlgo1.get_Hvalues(Img90, sigma=1)
	mask1 = CornerAlgo1.evaluate(H1)
	mask2 = CornerAlgo2.evaluate(H2)
	mask3 = CornerAlgo3.evaluate(H3)
	mask4 = CornerAlgo4.evaluate(H4)
	mask45 = CornerAlgo1.evaluate(Hr45)
	mask90 = CornerAlgo1.evaluate(Hr90)

	ind1 = np.nonzero(mask1)
	ind2 = np.nonzero(mask2)
	ind3 = np.nonzero(mask3)
	ind4 = np.nonzero(mask4)
	ind45 = np.nonzero(mask45)
	ind90 = np.nonzero(mask90)

	show_results(Img.gradient()[0], Img.gradient()[1], Img.im, ind1[0], ind1[1],["Ix","Iy","Detected corners"])

	show_results_2((ind3[0],ind3[1]), (ind2[0],ind2[1]), (ind4[0], ind4[1]), Images=[Img.im] * 3,  titles=['1e^4', '1e^5', '1e^7'])

	show_results_2((ind45[0],ind45[1]), (ind90[0],ind90[1]), (ind1[0], ind1[1]), Images=[Img45.im, Img90.im, Img.im],  titles=['45', '90', 'Original'])

	H1 = CornerAlgo1.get_Hvalues(Img, sigma=3)
	H2 = CornerAlgo2.get_Hvalues(Img, sigma=3)
	H4 = CornerAlgo4.get_Hvalues(Img, sigma=3)
	
	mask1 = CornerAlgo1.evaluate(H1)
	mask2 = CornerAlgo2.evaluate(H2)
	mask4 = CornerAlgo4.evaluate(H4)

	ind1 = np.nonzero(mask1)
	ind2 = np.nonzero(mask2)
	ind4 = np.nonzero(mask4)

	show_results_2((ind2[0],ind2[1]), (ind1[0],ind1[1]), (ind4[0], ind4[1]), Images=[Img.im] * 3,  titles=['1e^5', '1e^6', '1e^7'])
