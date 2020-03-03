import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    imH, imW = image1.shape

    # x, y = np.mgrid[0:imW, 0:imH]
    # x = x.reshape(1, imH*imW)
    # y = y.reshape(1, imH*imW)
    # co = np.vstack((x, y, np.ones((1, imH*imW))))
    # homop = np.dot(M, co)
    # xp = homop[0, :]
    # yp = homop[1, :]
    # xp = xp.reshape(imW*imH)
    # yp = yp.reshape(imW*imH)

    warpim1 = scipy.ndimage.affine_transform(
        image1, -M, offset=0.0, output_shape=None)
    diff = abs(warpim1 - image2)
    mask[diff > tolerance] = 1
    mask[diff < tolerance] = 0

    mask = scipy.ndimage.morphology.binary_erosion(mask)
    # mask = scipy.ndimage.morphology.binary_erosion(
    #     mask, structure=np.ones((2, 1)), iterations=1)

    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=1)

    return mask
