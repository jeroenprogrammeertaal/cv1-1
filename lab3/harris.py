import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
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
		factor = np.expand_dims(np.array([0.2989, 0.5870, 0.1140]).reshape(1,3),0)
		img = np.sum(im*factor, axis=2)
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



	def get_Hvalues(self, img):
		Ix, Iy = img.gradient()
		Ixx, Iyy, Ixy = Ix**2, Iy**2, np.multiply(Ix,Iy)
		assert img.shape == Ix.shape == Iy.shape == Ixx.shape == Iyy.shape == Ixy.shape
		h = int(self.window_size/2)
		Hvalues = np.zeros(img.shape)
		for y in range(h, img.shape[0]-h):
			for x in range(h, img.shape[1]-h):
				Ixx_window = np.sum(gaussian_filter(Ixx[y-h:y+h+1, x-h:x+h+1], sigma=3))
				Iyy_window = np.sum(gaussian_filter(Iyy[y-h:y+h+1, x-h:x+h+1], sigma=3))
				Ixy_window = np.sum(gaussian_filter(Ixy[y-h:y+h+1, x-h:x+h+1], sigma=3))
				H = (Iyy_window*Ixx_window - Ixy_window**2) - 0.04*(Ixx_window + Iyy_window)**2
				Hvalues[y,x] = H
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



if __name__ == "__main__":
	args = sys.argv
	p_im = args[1]

	Img = Image.readFrom(p_im)
	CornerAlgo = HarrisCornerDetector(3, 1e9)
	H = CornerAlgo.get_Hvalues(Img)
	mask = CornerAlgo.evaluate(H)

	
	img = np.copy(plt.imread(p_im))
	ind = np.nonzero(mask)
	plt.imshow(img)
	plt.scatter(ind[1], ind[0])
	plt.show()
	
	
