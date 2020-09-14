import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('./tst.pgm')
im = im[:,:,0]
h, w = im.shape
print(h,w)

img = np.zeros([h,w])

for i in range(h):
	for j in range(w):
		img[i,j] = im[i,j]

plt.imshow(img, cmap="gray")
plt.show()
