import numpy as np

def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = np.linalg.inv(Tr)
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.

    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

    There must be at least 6 calibration points for the 3D DLT.

    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' %(nd))
    
    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(n, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

    if (n < 6):
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, n))
        
    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    # Convert A to array
    A = np.asarray(A) 

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    #print(L)
    # Camera projection matrix
    H = L.reshape(3, nd + 1)
    #print(H)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    #print(H)
    H = H / H[-1, -1]
    #print(H)
    L = H.flatten('C')
    #print(L)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    # Mean distance:
    err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

    return L, err

def DLT(xyz,uv):
    #print('xyz = ', xyz)
    #print('uv = ',uv)
    """
    # Known 3D coordinates
    xyz = [[-21.01626016,  17.96747967,   1.95121951],
         [-26.41304348,  17.88043478,   7.17391304],
         [-29.12244898,  17.83673469,   9.79591837],
         [-23.41121495,  16.0100647,    6.09633357],
         [-38.47326337,  27.35632184,   9.63518241],
         [-35.94240838,  24.21465969,  10.21690352],
         [-37.05774278,  32.41994751,   3.46456693],
         [-42.41607774,  29.11660777,  11.71378092],
         [-14.78171257,   6.62825557,   6.81281619]]
    # Known pixel coordinates
    uv = [[4262.317897, 670.0449627],
     [3597.851038, 260.6636832],
     [3265.617609, 55.97304337],
     [3269.603617, 257.1657872],
     [5625.441744, 510.1037212], 
     [4781.456862, 319.3414634],
     [7639.787591, 1225.133686],
     [5800.904622, 430.9253664],
     [872.0886965, -234.2653165]]
     """

    nd = 3
    P, err = DLTcalib(nd, xyz, uv)
    """
    print('Matrix')
    print(P)
    print('\nError')
    print(err)
    """
    return P

if __name__ == "__main__":
    DLT()
