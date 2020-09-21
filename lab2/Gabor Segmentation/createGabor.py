import numpy as np
import matplotlib.pyplot as plt

def createGabor( sigma, theta, lamda, psi, gamma ):
#CREATEGABOR Creates a complex valued Gabor filter.
#   myGabor = createGabor( sigma, theta, lamda, psi, gamma ) generates
#   Gabor kernels.  
#   - ARGUMENTS
#     sigma      Standard deviation of Gaussian envelope.
#     theta      Orientation of the Gaussian envelope. Takes arguments in
#                the range [0, pi/2).
#     lamda     The wavelength for the carriers. The central frequency 
#                (w_c) of the carrier signals.
#     psi        Phase offset for the carrier signal, sin(w_c . t + psi).
#     gamma      Controls the aspect ratio of the Gaussian envelope
#   
#   - OUTPUT
#     myGabor    A matrix of size [h,w,2], holding the real and imaginary 
#                parts of the Gabor in myGabor(:,:,1) and myGabor(:,:,2),
#                respectively.
                
    # Set the aspect ratio.
    sigma_x = sigma
    sigma_y = sigma/gamma

    # Generate a grid
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1,xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1,ymax))

    # Make sure that we get square filters. 
    xmax = max(xmax,ymax)
    ymax = max(xmax,ymax)
    xmin = -xmax 
    ymin = -ymax

    # Generate a coordinate system in the range [xmin,xmax] and [ymin, ymax]. 
    [x,y] = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    #print('x shape: {}, y shape: {}'.format(x.shape, y.shape))

    # Convert to a 2-by-N matrix where N is the number of pixels in the kernel.
    XY = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)), axis=0)
    #print(XY.shape)

    # Compute the rotation of pixels by theta.
    # \\ Hint: Use the rotation matrix to compute the rotated pixel coordinates: rot(theta) * XY.
    rotMat = generateRotationMatrix(theta)
    rot_XY = np.matmul(rotMat,XY)
    #print(rot_XY.shape)
    rot_x = rot_XY[0,:]
    rot_y = rot_XY[1,:]
    #print(rot_x.shape, rot_y.shape)


    # Create the Gaussian envelope.
    # \\ IMPLEMENT the helper function createGauss.
    gaussianEnv = createGauss(rot_x, rot_y, gamma, sigma)
    #print(gaussianEnv.shape)

    # Create the orthogonal carrier signals.
    # \\ IMPLEMENT the helper functions createCos and createSin.
    cosCarrier = createCos(rot_x, lamda, psi)
    #print(cosCarrier.shape)
    sinCarrier = createSin(rot_x, lamda, psi)
    #print(sinCarrier.shape)

    # Modulate (multiply) Gaussian envelope with the carriers to compute 
    # the real and imaginary components of the omplex Gabor filter. 
    myGabor_real = gaussianEnv*cosCarrier #None  # \\TODO: modulate gaussianEnv with cosCarrier
    myGabor_imaginary = gaussianEnv*sinCarrier #None  # \\TODO: modulate gaussianEnv with sinCarrier

    # Pack myGabor_real and myGabor_imaginary into myGabor.
    h, w = myGabor_real.shape
    myGabor = np.zeros((h, w, 2))
    myGabor[:,:,0] = myGabor_real
    myGabor[:,:,1] = myGabor_imaginary

    # Uncomment below line to see how are the gabor filters
    #fig, axs = plt.subplots(1,2)    
    #axs[0].imshow(myGabor_real)
    #axs[1].imshow(myGabor_imaginary)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1)
    #ax.imshow(myGabor_real)    # Real
    #ax.axis("off")
    #ax = fig.add_subplot(1, 2, 2)
    #ax.imshow(myGabor_imaginary)    # Real
    #ax.axis("off")
    #plt.show()
    return myGabor


# Helper Functions 
# ----------------------------------------------------------
def generateRotationMatrix(theta):
    # ----------------------------------------------------------
    # Returns the rotation matrix. 
    # \\ Hint: https://en.wikipedia.org/wiki/Rotation_matrix \\
    rotMat = np.array([[np.cos(theta), np.sin(theta)],[-1*np.sin(theta), np.cos(theta)]])
    return rotMat

# ----------------------------------------------------------
def createCos(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D cosine carrier. 
    cosCarrier = np.cos((2*np.pi*rot_x)/lamda + psi) #None  # \\TODO: Implement the cosine given rot_x, lamda and psi.

    # Reshape the vector representation to matrix.
    cosCarrier = np.reshape(cosCarrier, (int(np.sqrt(len(cosCarrier))), -1))
    return cosCarrier

# ----------------------------------------------------------
def createSin(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D sine carrier. 
    sinCarrier = np.sin((2*np.pi*rot_x)/lamda + psi) #None  # \\TODO: Implement the sine given rot_x, lamda and psi.

    # Reshape the vector representation to matrix.
    sinCarrier = np.reshape(sinCarrier, (int(np.sqrt(len(sinCarrier))), -1))
    return sinCarrier

# ----------------------------------------------------------
def createGauss(rot_x, rot_y, gamma, sigma):
    # ----------------------------------------------------------
    # Returns the 2D Gaussian Envelope. 
    gaussEnv = np.exp(-1*((np.power(rot_x, 2) + np.power(gamma, 2)*np.power(rot_y, 2))/(2*np.power(sigma, 2))))  # \\TODO: Implement the Gaussian envelope.
    #print(len(gaussEnv))
    # Reshape the vector representation to matrix.
    gaussEnv = np.reshape(gaussEnv, (int(np.sqrt(len(gaussEnv))), -1))
    return gaussEnv



def plot(exps, exps_config):
    
    fig = plt.figure(figsize=(5,2))
    for i, exp in enumerate(exps):
       ax = fig.add_subplot(1, len(exps), i+1)
       myGabor_real = exp[:,:,0]
       ax.imshow(myGabor_real)
       ax.axis("off")
       print(str(np.around(exps_config['sigma'][i], 2)))
       plt.title("{}".format(np.around(exps_config['sigma'][i], 2)), fontsize=8)
    plt.suptitle("theta: {}, lambda: {}, psi: {}, gamma:{}".format(exps_config['theta'], np.around(exps_config['lambda'],1), exps_config['psi'], exps_config['gamma']), fontsize=9)
    plt.tight_layout()
    plt.savefig('sigma_exp.png')
    plt.show()



if __name__ == "__main__":
    numRows, numCols = 50,50
    lambdaMin = 4/np.sqrt(2)
    lambdaMax = np.sqrt(abs(numRows)**2 + abs(numCols)**2)
    n = np.floor(np.log2(lambdaMax/lambdaMin))
    lambdas = 2**np.arange(0, (n-2)+1) * lambdaMin
    dTheta       = 2 * np.pi/8                 
    orientations = np.arange(0, np.pi+dTheta, dTheta)
    sigmas = np.array([1, 2, 5, 10, 20])
    psi = 0
    gamma = 0.5
    print(orientations)
    #print(lambdas.extend(20.1, 50)
    print(sigmas)
    print("length lambdas: {}, orientations: {}, sigmas: {}".format(len(lambdas), len(orientations), len(sigmas)))
    print(sigmas[0], orientations[2], lambdas[2], psi, gamma)
    sigma_exp = []
    for sigma in sigmas:
        gb = createGabor(sigma, orientations[0], lambdas[2], psi, 0.25)
        sigma_exp.append(gb)
    sigma_exp_config = {'theta' : orientations[0], 'sigma': sigmas, 'lambda': lambdas[2], 'psi': psi, 'gamma': 0.25}
    plot(sigma_exp, sigma_exp_config)
        
