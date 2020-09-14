from iid_image_formation import *
import numpy as np
import sys
import matplotlib.pyplot as plt


def recolour(albedo, shading=None):
	r_channel, g_channel, b_channel = albedo[:,:,0], albedo[:,:,1], albedo[:,:,1]
	non_zero_els_r = r_channel[np.nonzero(r_channel)]
	non_zero_els_g = g_channel[np.nonzero(g_channel)]
	non_zero_els_b = b_channel[np.nonzero(b_channel)]


	
	if len(list(set(non_zero_els_r))) == len(list(set(non_zero_els_g))) == len(list(set(non_zero_els_b))) == 1:
		print("Surface is Uniform with R: {} G:{} B:{}".format(non_zero_els_r[0], non_zero_els_g[0], non_zero_els_b[0]))

	"""
		We will use the albedo image for recolouring

	"""
	recoloured_r_channel = np.zeros(albedo.shape[:2])
	recoloured_g_channel = np.zeros(albedo.shape[:2])
	recoloured_g_channel[np.nonzero(g_channel)] = 1.0
	recoloured_b_channel = np.zeros(albedo.shape[:2])

	new_albedo = np.stack([recoloured_r_channel, recoloured_g_channel, recoloured_b_channel], axis=2)

	return new_albedo	
	
	


def plot(original, recolored):
	fig, axs = plt.subplots(1,2)
	axs[0].imshow(original)
	axs[0].set_title("Original Image")

	axs[1].imshow(recolored)
	axs[1].set_title("Recolored Image")
	plt.savefig('recolored_image.png')
	plt.show()
	

def inspect_recolored(recolored):
	r_channel, g_channel, b_channel = recolored[:,:,0], recolored[:,:,1], recolored[:,:,1]
	non_zero_els_r = r_channel[np.nonzero(r_channel)]
	non_zero_els_g = g_channel[np.nonzero(g_channel)]
	non_zero_els_b = b_channel[np.nonzero(b_channel)]
	
	print(non_zero_els_r)
	print(non_zero_els_g)
	print(non_zero_els_b)

	
	if len(list(set(non_zero_els_r))) == len(list(set(non_zero_els_g))) == len(list(set(non_zero_els_b))) == 1:
		print("Surface is with R: {} G:{} B:{}".format(non_zero_els_r[0], non_zero_els_g[0], non_zero_els_b[0]))

	else:
		print("Surface of the recolored image in non-uniform")
		fig, axs = plt.subplots(1,3)
		axs[0].imshow(r_channel, cmap='gray')
		axs[0].set_title("R channel")

		axs[1].imshow(g_channel, cmap='gray')
		axs[1].set_title("G channel")
	
		axs[2].imshow(b_channel, cmap='gray')
		axs[2].set_title("B_channel")
		plt.savefig("recolored_image_rgb.png")
		plt.show()
	





if __name__ == "__main__":
	
	args = sys.argv
	path_to_original, path_to_albedo, path_to_shading = args[1], args[2], args[3]

	original, albedo, shading = read_images(path_to_original, path_to_albedo, path_to_shading)
	
	new_albedo = recolour(albedo)
	
	recolored = reconstruct(new_albedo,shading)
	
	plot(original, recolored)
	
	inspect_recolored(recolored)
