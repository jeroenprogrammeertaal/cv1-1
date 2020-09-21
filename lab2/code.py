import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import image
import matplotlib.pyplot as plt
from Image_enhancement.gauss1D import gauss1D
from Image_enhancement.gauss2D import gauss2D
from Image_enhancement.compute_LoG import *
from Image_enhancement.compute_gradient import compute_gradient

imageName='Image_enhancement/images/image2.jpg'
data_image=np.array(image.imread(imageName))
method1=compute_LoG(imageName, 1)
method2=compute_LoG(imageName, 2)
method3=compute_LoG(imageName, 3)

fig, axs = plt.subplots(2,2)

axs[0,0].set_title('Original image')
axs[0,0].imshow(data_image, cmap="gray")
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[0,1].set_title('Method 1')
axs[0,1].imshow(method1, cmap="gray")
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])


axs[1,0].set_title('Method 2')
axs[1,0].imshow(method2, cmap="gray")
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])

axs[1,1].set_title('Method 3')
axs[1,1].imshow(method3, cmap="gray")
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])

plt.show()

[Gx, Gy, im_magnitude,im_direction]=compute_gradient(imageName)

fig, axs = plt.subplots(2,2)

axs[0,0].set_title('Gx')
axs[0,0].imshow(Gx, cmap="gray")
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[0,1].set_title('Gy')
axs[0,1].imshow(Gy, cmap="gray")
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])

axs[1,0].set_title('im_magnitude')
axs[1,0].imshow(im_magnitude, cmap="gray")
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])

axs[1,1].set_title('im_direction')
axs[1,1].imshow(im_direction, cmap="gray")
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])

plt.show()