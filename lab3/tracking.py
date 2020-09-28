from harris import *
from lucas import LucasKanade
import os
import numpy as np
import matplotlib
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation


path = './person_toy'


#fig = plt.figure()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

#animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800) #

images = sorted(os.listdir(path))
print(images)

Img = Image.readFrom(os.path.join(path, images[0]))
CornerAlgo = HarrisCornerDetector(3, 1e9)
H = CornerAlgo.get_Hvalues(Img)
mask = CornerAlgo.evaluate(H)
indices = np.nonzero(mask)
indicesl = np.stack([indices[0], indices[1]], axis=0) #list(zip(indices[0].tolist(), indices[1].tolist()))
print(indicesl.shape)
frames = []
images_ = []
indices_ = []
for i in range(0, len(images)-1):
	img1 = Image.readFrom(os.path.join(path , images[i]))
	img2 = Image.readFrom(os.path.join(path , images[i+1]))
	lucas = LucasKanade(img1, img2)
	vectors = []
	updated_indices = []
	for i in range(indicesl.shape[1]):
		ind = (indicesl[0,i], indicesl[1,i])
		print(ind)
		vector = lucas.regionAlgo(ind)
		vectors.append(vector)
		vec = vector.flatten().tolist()
		print(vec)
		updated_ind = np.array([ind[0], ind[1]]) + np.array([vec[0], vec[1]]) #+ np.array([1/20, 1/20])
		print(updated_ind)
		updated_indices.append(updated_ind.astype(int))
		#break
	print(np.array(updated_indices))
	print(np.array(updated_indices).shape)
	indicesl = np.array(updated_indices).T
	frames.append(np.array(vectors))
	images_.append(img1.im)
	indices_.append(indicesl)




fig = plt.figure()
def Plot(i):
	plt.clf()
	vectors, image, indices = frames[i], images_[i], indices_[i]
	vectors = vectors.reshape((-1,2))
	plt.imshow(image, cmap="gray")
	plt.scatter(indices[1,:], indices[0,:])
	plt.quiver(indices[1,:], indices[0,:], vectors[::,0], vectors[::,1], angles='xy')
	return plt
#Plot(0)
ani = matplotlib.animation.FuncAnimation(fig, Plot, frames=len(images_), repeat=True)
ani.save('./tstt_toy.mp4', writer=writer)
