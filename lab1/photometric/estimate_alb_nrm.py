from __future__ import division
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def estimate_alb_nrm( image_stack, scriptV, shadow_trick=False):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])
    
    """
    ================
    Your code here
    scriptV = (#images, 3)
    image_stack = (h, w, #images)
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """

    if shadow_trick:
        for y in range(h):
            for x in range(w):
                i_xy = image_stack[y,x,:]
                scriptI = np.diag(i_xy)
                g = np.linalg.lstsq(scriptI @ scriptV, scriptI @ i_xy, rcond=-1)[0]
                albedo[y,x] = np.linalg.norm(g)
                normal[y,x,:] = g / np.linalg.norm(g)
    else:
        for y in range(h):
            for x in range(w):
                i_xy = image_stack[y,x,:]
                g = np.linalg.lstsq(scriptV, i_xy, rcond=-1)[0]
                albedo[y,x] = np.linalg.norm(g)
                normal[y,x,:] = g / np.linalg.norm(g)

    return albedo, normal
    
if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)
