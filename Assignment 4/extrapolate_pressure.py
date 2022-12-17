import numpy as np
def extrapolate(array, nx, ny, case):
    array_ext = np.copy(array)

    if case == 'constant gradient':

        # east face
        j = 0
        for i in np.arange(1,ny+1):
            array_ext[i,j] = 2 * array_ext[i,j+1] - 1 * array_ext[i,j+2]

        # west face
        j = nx + 1
        for i in np.arange(1,ny+1):
            array_ext[i,j] = 2 * array_ext[i, j-1] - 1 * array_ext[i,j-2]

        # north face
        i = 0
        for j in np.arange(1,nx+1):
            array_ext[i,j] = 2 * array_ext[i+1, j] - 1 * array_ext[i+2, j]

        # south face
        i = ny + 1
        for j in np.arange(1,nx+1):
            array_ext[i,j] = 2 * array_ext[i-1, j] - 1 * array_ext[i-2, j]

    elif case == 'zero gradient':

        j = 0
        for i in np.arange(1, ny + 1):
            array_ext[i, j] = array_ext[i, j + 1]

        # west face
        j = nx + 1
        for i in np.arange(1, ny + 1):
            array_ext[i, j] = array_ext[i, j - 1]

        # north face
        i = 0
        for j in np.arange(1, nx + 1):
            array_ext[i, j] = array_ext[i + 1, j]

        # south face
        i = ny + 1
        for j in np.arange(1, nx + 1):
            array_ext[i, j] = array_ext[i - 1, j]
    else:
        raise Exception('chose "zero gradient" or "constant gradient"')

    return array_ext