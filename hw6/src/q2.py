# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface, normalize
from utils import enforceIntegrability
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2


def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    u, s, v = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0.
    # print(u.shape, v.shape)
    # print(s)
    B = v[0:3, :]
    L = u[0:3, :]
    # print(L.shape, B.shape)

    return B, L


if __name__ == "__main__":

    # Put your main code here
    # Q2.b
    I, originalL, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    # print(originalL, L)

    # Q2.f
    mu = 0  # -0.5, 0.5, 1
    nu = 0  # 0.5, 1, 3
    lam = 3  # -1, 1.5, 3
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B = np.linalg.inv(G.T).dot(B)

    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Albedos Image
    # plt.imshow(albedoIm, cmap='gray')
    # # cv2.imwrite('../results/q2b_albedo.png', (albedoIm*255))
    # plt.show()

    # Normals Image
    normalIm = normalize(normalIm)
    # plt.imshow(normalIm, cmap='rainbow')
    # plt.savefig('../results/q2b_normal.png')
    # plt.show()

    # Q2.d
    surface = normalize(estimateShape(normals, s))
    # surface *= 255
    plotSurface(surface)
