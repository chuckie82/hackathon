# Calculate the polarization factor for horizontally polarized light
def polarizationFactorLinear(coords_x,coords_y,coords_z):
    eps = np.finfo(float).eps
    phi = np.arctan2(coords_y,coords_x)
    theta = np.arctan2(np.sqrt(np.square(coords_x)+np.square(coords_y)),coords_z)
    polarization = 1. - (np.sin(theta)*np.cos(phi))**2
    badInd = np.where(polarization<eps)
    polarization[badInd] = 0
    return polarization

# To correct for polarization, I_correct = I/P
# where I is input image matrix and P is polarization factor matrix.
