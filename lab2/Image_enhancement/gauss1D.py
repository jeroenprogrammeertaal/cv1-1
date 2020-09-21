import numpy as np

def  gauss1D(sigma ,kernel_size):
	# your code
	G = np.zeros((1, kernel_size)) #Setting up G
	if kernel_size % 2 == 0: #Error for even kernel size
		raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
	x = np.arange(-(kernel_size//2), kernel_size//2+1, 1)
	N = np.exp(-(x**2)/(2*(sigma**2)))/(sigma*(np.sqrt(2*np.pi))) #Non normalised G
	G=N/(sum(N)) #Normalised G
	return G
