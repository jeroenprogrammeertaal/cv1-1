import numpy as np
import matplotlib.pyplot as plt 
import cv2


def grey_world(image):
    im = cv2.imread(image)
    im = im[:, :, ::-1]
    h,w,c = im.shape
    new_image = np.zeros((h,w,c))
    
    #compute mean of each channel
    avg_im = np.array([np.mean(im[:,:,0]), np.mean(im[:,:,1]), np.mean(im[:,:,2])])
    avg_grey = [128, 128, 128]
    A = np.diag(avg_grey / avg_im)
    
    for y in range(h):
        for x in range(w):
            new_image[y,x,:] = A @ im[y,x,:]

    print(new_image[:,:,0].mean(), new_image[:,:,1].mean(), new_image[:,:,2].mean())

    new_image = new_image / 255
    new_image = new_image.astype(np.float32)

    fig, axs = plt.subplots(1,2)
    
    axs[0].imshow(im)
    axs[0].set_title('Original', fontsize="20")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(new_image)
    axs[1].set_title('Grey-World', fontsize="20")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.savefig('gray_world.png')
    plt.show()

if __name__ == "__main__":
    grey_world('awb.jpg')

