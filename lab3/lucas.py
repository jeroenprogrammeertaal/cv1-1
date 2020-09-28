import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from harris import Image
import sys


#def read_images(p_im1, p_im2):
	#im1 = plt.imread(p_im1)[:,:,0]
	#im2 = plt.imread(p_im2)[:,:,0]
	#print("Shape im1: {}, im2: {}".format(im1.shape,im2.shape))
	#return im1, im2


#def gradient_(im):
	#imx = im.flatten()
	#imy = im.flatten('F')
	#G = np.array([-1, 0, 1])
	#Gx = signal.convolve(imx, G, mode='same').reshape(im.shape)
	#Gy = signal.convolve(imy, G, mode='same').reshape(im.shape).T
	#return Gx, Gy

#def lstsq(A, b):
#	return np.linalg.lstsq(A, b)[0]


def plot(im1, vectors):
	plt.imshow(im1, cmap='gray')
	X = np.arange(8, im1.shape[1], 15)
	Y = np.arange(8, im1.shape[0], 15)
	X, Y = np.meshgrid(X,Y)
	print(X.shape, Y.shape)
	plt.quiver(X, Y, vectors[::,0], vectors[::,1], angles='xy')
	plt.show()



def algo(im1, im2):
	rows, columns = im1.shape[0], im1.shape[1]
	start_row, start_col = rows%15, columns%15
	Gx, Gy = im1.gradient()
	Gt = im2.im - im1.im
	vectors = []
	for y in range(start_row, rows, 15):
		for x in range(start_col, columns, 15):
			gx, gy, gt  = Gx[y:y+15, x:x+15], Gy[y:y+15, x:x+15], Gt[y:y+15, x:x+15]
			gx, gy = gx.flatten(), gy.flatten()
			A = np.stack([gx, gy], axis=1)
			b = np.expand_dims(-1*gt.flatten(),1)
			xx = np.linalg.lstsq(A,b)[0]
			vectors.append(xx.flatten())
	return np.array(vectors)

	

class LucasKanade(object):
	def __init__(self, image1, image2):
		self.image1 = image1
		self.image2 = image2
		self.window_size = 15
	
		self.Gx, self.Gy = self.image1.gradient()
		self.Gt = self.image2.im - self.image1.im
		

	def regionAlgo(self, index):
		y_coord, x_coord = index
		h = int(self.window_size/2)

		gx = self.Gx[y_coord-h:y_coord+h+1, x_coord-h:x_coord+h+1]
		gy = self.Gy[y_coord-h:y_coord+h+1, x_coord-h:x_coord+h+1]
		gt = self.Gt[y_coord-h:y_coord+h+1, x_coord-h:x_coord+h+1]

		gx, gy, gt = gx.flatten(), gy.flatten(), gt.flatten()

		A = np.stack([gx, gy], axis=1)
		b = np.expand_dims(-1*gt,1)
		
		assert A.shape[0] == b.shape[0]

		xx = np.linalg.lstsq(A,b)[0]

		return xx

			
		
		
			














if __name__ == "__main__":
	args = sys.argv
	path_im1, path_im2 = args[1], args[2]
	Img1, Img2 = Image.readFrom(path_im1), Image.readFrom(path_im2)
	vectors = algo(Img1, Img2)
	plot(im1.im, vectors)

	
