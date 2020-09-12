import sys
import matplotlib.pyplot as plt
import numpy as np

def read_images(original, albedo, shading):
    """
       inputs:
          original : path to original image
          albedo : path to albedo intrinsic image
          shading : path to shading intrinsic image

    """

    albedo = plt.imread(albedo)
    shading = plt.imread(shading)
    original = plt.imread(original)
    
    return original, albedo, shading



def reconstruct(albedo, shading):
   
    ball = albedo*np.expand_dims(shading, axis=2)
    return ball


def plot(original, albedo, shading, ball):

    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(original)
    axs[0,0].set_title("Original Image")

    axs[0,1].imshow(albedo)
    axs[0,1].set_title("Albedo Image")

    axs[1,0].imshow(shading, cmap="gray")
    axs[1,0].set_title("Shading Image")

    axs[1,1].imshow(ball)
    axs[1,1].set_title("Reconstructed Image")
    
    plt.savefig('intrinsic_image.png')
    plt.show()
    



if __name__ == "__main__":
    args = sys.argv
    original, path_to_albedo, path_to_shading = args[1], args[2], args[3]
    
    O, A, S = read_images(original, path_to_albedo, path_to_shading)
    reconstructed = reconstruct(A, S)
    plot(O, A, S, reconstructed)
