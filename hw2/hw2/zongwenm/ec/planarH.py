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

    A = np.empty([2*x1.shape[0], 9])

    for ind in range(x1.shape[0]):
        u_1 = x1[ind, 0]
        v_1 = x1[ind, 1]
        u_2 = x2[ind, 0]
        v_2 = x2[ind, 1]

        A[2*ind] = [-u_1, -v_1, -1, 0, 0, 0, u_1*u_2, u_2*v_1, u_2]
        A[2*ind+1] = [0, 0, 0, -u_1, -v_1, -1, v_2*u_1, v_2*v_1, v_2]

    U, S, V_t = np.linalg.svd(A)
    eig_val = S[-1]
    eig_vect = V_t[-1, :] / V_t[-1, -1]

    H2to1 = eig_vect
    H2to1 = H2to1.reshape(3, 3)

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    mean_x1 = np.mean(x1[:, 0])
    mean_y1 = np.mean(x1[:, 1])

    mean_x2 = np.mean(x2[:, 0])
    mean_y2 = np.mean(x2[:, 1])
    sum_s_den_x1 = 0
    sum_s_den_x2 = 0
    s_x1 = np.empty((x1.shape[0]))
    s_x2 = np.empty((x2.shape[0]))

    for i in range(x1.shape[0]):
        s_x1[i] = np.sqrt((x1[i, 0]-mean_x1)**2 + (x1[i, 1]-mean_y1)**2)

    for i in range(x2.shape[0]):
        s_x2[i] = np.sqrt((x2[i, 0]-mean_x2)**2 + (x2[i, 1]-mean_y2)**2)

    s1 = np.sqrt(2) / np.max(s_x1)
    s1_mat = np.array([[s1, 0, 0], [0, s1, 0], [0, 0, 1]])
    trans1_mat = np.array([[1, 0, -mean_x1], [0, 1, -mean_y1], [0, 0, 1]])
    T1 = np.dot(s1_mat, trans1_mat)

    s2 = np.sqrt(2) / np.max(s_x2)
    s2_mat = np.array([[s2, 0, 0], [0, s2, 0], [0, 0, 1]])
    trans2_mat = np.array([[1, 0, -mean_x2], [0, 1, -mean_y2], [0, 0, 1]])
    T2 = np.dot(s2_mat, trans2_mat)

    # Similarity transform 1n
    x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x1_hom = T1@x1_hom.T

    # Similarity transform 2
    x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))
    x2_hom = T2@x2_hom.T

    # Compute homography
    H2to1_hom = computeH(x1_hom, x2_hom)
    # Denormalization
    H2to1 = np.dot(np.linalg.inv(T2), H2to1_hom)
    H2to1 = np.dot(H2to1, T1)

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    bestH2to1 = np.empty([3, 3])
    rand_1 = np.empty([2, 4])
    rand_2 = np.empty([2, 4])
    max_inliers = -1

    x1 = locs1
    x2 = locs2
    x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))

    for ind in range(max_iters):
        tot_inliers = 0
        ind_rand = np.random.choice(locs1.shape[0], 4)

        rand_1 = locs1[ind_rand, :]
        rand_2 = locs2[ind_rand, :]

        H_norm = computeH_norm(rand_1, rand_2)

        for i in range(x2_hom.shape[0]):

            pred_x2 = np.dot(H_norm, x1_hom[i].T)
            pred_x2[0] = pred_x2[0]/pred_x2[2]
            pred_x2[1] = pred_x2[1]/pred_x2[2]

            error_1 = (x2_hom[i][0] - pred_x2[0])
            error_2 = (x2_hom[i][1] - pred_x2[1])
            error = [error_1, error_2]
            error = np.linalg.norm(error)
            if error <= inlier_tol:
                tot_inliers += 1

        if tot_inliers > max_inliers:
            bestH2to1 = H_norm
            max_inliers = tot_inliers

    inliers = max_inliers
    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.
    mask_ones = np.ones(template.shape)
    mask_ones = cv2.transpose(mask_ones)
    warp_mask = cv2.warpPerspective(
        mask_ones, H2to1, (img.shape[0], img.shape[1]))
    template = cv2.transpose(template)

    warp_mask = cv2.transpose(warp_mask)
    non_zero_ind = np.nonzero(warp_mask)

    warp_template = cv2.warpPerspective(
        template, H2to1, (img.shape[0], img.shape[1]))
    warp_template = cv2.transpose(warp_template)
    img[non_zero_ind] = warp_template[non_zero_ind]
    # composite_img = img.astype('uint8')
    # composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
    composite_img = img

    return composite_img
