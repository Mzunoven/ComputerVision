import numpy as np


def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    im_row = im.shape[0]
    im_col = im.shape[1]
    row, col = np.mgrid[0:output_shape[0], 0:output_shape[1]]
    invA = np.linalg.inv(A)
    x = np.round(invA[1, 0]*row + invA[1, 1]*col + invA[1, 2]).astype(int)
    y = np.round(invA[0, 0]*row + invA[0, 1]*col + invA[0, 2]).astype(int)
    # print(y[1], x[1])
    x[(x < 0) | (x >= im_col)] = im_col
    y[(y < 0) | (y >= im_row)] = im_row

    return np.pad(im, ((0, 1), (0, 1)))[(y, x)]
