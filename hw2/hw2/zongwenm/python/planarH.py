import numpy as np
import cv2
from opts import get_opts
import matplotlib.pyplot as plt
import scipy
import skimage
import pdb
from matchPics import matchPics

opts = get_opts()


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    # x1 = np.transpose(x1)
    # x2 = np.transpose(x2)
    N = x1.shape[0]
    A = np.zeros((2*N, 9))

    idx = np.random.randint(len(x1), size=4)
    for i in idx:
        x, y = x2[i]
        u, v = x1[i]

        m = np.array([[x, y, 1, 0, 0, 0, -u * x, -u * y, -u],
                      [0, 0, 0, x, y, 1, -v * x, -v * y, -v]])
        if A is None:
            A = m
        else:
            A = np.vstack((A, m))
    U, S, V = np.linalg.svd(A)
    H2to1 = V.T[:, -1].reshape([3, 3])

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    mean_1 = x1.mean(0)
    mean_2 = x2.mean(0)

    # Shift the origin of the points to the centroid
    x1 = x1 - mean_1
    x2 = x2 - mean_2

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_1 = np.max(np.linalg.norm(x1, axis=1))
    max_2 = np.max(np.linalg.norm(x2, axis=1))

    # Similarity transform 1
    s1 = 1 / (max_1 / np.sqrt(2))
    x1 = x1 * s1
    T1 = np.array([[s1, 0, -s1*mean_1[0]], [0, s1, -s1*mean_1[1]], [0, 0, 1]])

    # Similarity transform 2
    s2 = 1 / (max_2 / np.sqrt(2))
    x2 = x2 * s1
    T2 = np.array([[s2, 0, -s2*mean_2[0]], [0, s2, -s2*mean_2[1]], [0, 0, 1]])

    # Compute homography
    H2to1_hom = computeH(x1, x2)
    H2to1 = np.dot(np.linalg.inv(T1), H2to1_hom)
    H2to1 = np.dot(H2to1, T2)

    # Denormalization

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    x1 = locs1
    x2 = locs2
    x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))

    bestH2to1 = None

    inliers = None

    for idx in range(max_iters):

        H_norm = computeH_norm(x1, x2)
        x2_map = np.dot(H_norm, x2_hom.T)
        x2_map /= x2_map[2, :]
        x2_map[2, :] = np.ones(x2_map.shape[1])
        e = x1_hom.T - x2_map
        error = np.sum(e**2, axis=1)

        err = error <= inlier_tol
        inlier = err.astype(int)

        if bestH2to1 is None and inliers is None:
            bestH2to1 = H_norm
            inliers = inlier

        else:
            if np.sum(inlier == 1) > np.sum(inliers == 1):
                bestH2to1 = H_norm
                inliers = inlier

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.
    warped_img = cv2.warpPerspective(img.swapaxes(
        0, 1), H2to1, (template.shape[0], template.shape[1])).swapaxes(0, 1)
    mask = np.zeros(warped_img.shape)
    # Create mask of same size as template
    ch1, ch2, ch3 = warped_img[:, :,
                               0], warped_img[:, :, 1], warped_img[:, :, 2]

    mask[:, :, 0] = (ch1 != 0).astype(int)
    mask[:, :, 1] = (ch2 != 0).astype(int)
    mask[:, :, 2] = (ch3 != 0).astype(int)
    # Warp mask by appropriate homography

    # Warp template by appropriate homography
    mask = np.logical_not(mask).astype(int)
    # Use mask to combine the warped template and the image

    composite_img = warped_img + template * mask

    return composite_img
