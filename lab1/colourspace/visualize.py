import matplotlib.pyplot as plt


def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    shape = input_image.shape
    n_channels = shape[-1]

    if n_channels > 3:
        fig, axs = plt.subplots(2,2)

        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,0].set_title('Lightness', fontsize=20)
        axs[0,0].imshow(input_image[:,:,0], cmap='gray')

        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,1].set_title('Average', fontsize=20)
        axs[0,1].imshow(input_image[:,:,1], cmap='gray')

        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])
        axs[1,0].set_title('Luminosity', fontsize=20)
        axs[1,0].imshow(input_image[:,:,2], cmap='gray')

        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        axs[1,1].set_title('Opencv', fontsize=20)
        axs[1,1].imshow(input_image[:,:,3], cmap='gray')

        plt.show()

    else:
        fig, axs = plt.subplots(2,2)
        
        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,0].set_title('converted image', fontsize=20)
        axs[0,0].imshow(input_image)

        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,1].set_title('Y', fontsize=20)
        axs[0,1].imshow(input_image[:,:,0], cmap='gray')

        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])
        axs[1,0].set_title('CB', fontsize=20)
        axs[1,0].imshow(input_image[:,:,1], cmap='gray')

        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        axs[1,1].set_title('CR', fontsize=20)
        axs[1,1].imshow(input_image[:,:,2], cmap='gray')

        plt.show()