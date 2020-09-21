from Image_enhancement.gauss1D import gauss1D

def gauss2D( sigma , kernel_size ):
	## solution
	G1 = gauss1D(sigma,kernel_size)
	G2 = G1.reshape((kernel_size,1))
	G = G2 * G1
	return G