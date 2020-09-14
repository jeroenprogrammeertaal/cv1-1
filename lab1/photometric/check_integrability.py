from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])
    
    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy

    """

    p = normals[:,:,0]/normals[:,:,2]
    q = normals[:,:,1]/normals[:,:,2]
    print(p.shape, q.shape)
    
    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0
    
    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    
    """
    p_unroll = p.flatten()
    q_unroll = q.flatten('F')

    p_2d = np.convolve(p_unroll, np.array([1,-1]), 'same').reshape(normals.shape[:2])
    q_2d = np.convolve(q_unroll, np.array([1,-1]), 'same').reshape(normals.shape[:2])	
    
    print(p_unroll.shape, q_unroll.shape)
      
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(p_2d, cmap='gray')
    axs[0].set_title("dp/dy")
    axs[1].imshow(q_2d, cmap='gray')
    axs[1].set_title("dq/dx")
    plt.show()
    

    SE = (p_2d - q_2d)**2

    print(p_2d.shape, q_2d.shape, SE.shape)
    
    #SE = SE.reshape(normals.shape[:2])

    
    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)
