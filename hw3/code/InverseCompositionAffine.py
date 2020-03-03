import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage


def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hei, wid = It.shape
    p = np.zeros(6)
    grad = np.gradient(It)
    grad_x = grad[1]
    grad_y = grad[0]

    H = np.zeros((6, 6))
    for i in range(hei):
        for j in range(wid):
            jacobian = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
            cur_grad = np.array([grad_x[i, j], grad_y[i, j]])
            A = np.dot(cur_grad, jacobian)[np.newaxis, :]
            cur_H = np.dot(A.T, A)
            H = H + cur_H
    change = 1
    count = 1
    while change > threshold and count < num_iters:
        warped_It1 = scipy.ndimage.affine_transform(It1, M, offset=0.0)
        bias = warped_It1 - It
        b = np.zeros((6, 1))
        for i in range(hei):
            for j in range(wid):
                jacobian = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
                cur_grad = np.array([grad_x[i, j], grad_y[i, j]])
                A = np.dot(cur_grad, jacobian)[np.newaxis, :]
                cur_b = np.transpose(A) * bias[i, j]
                b = b + cur_b
        dp = np.dot(np.linalg.pinv(H), b)
        dM = np.array([[1.0+dp[0][0], dp[2][0], dp[4][0]],
                       [dp[1][0], 1.0+dp[3][0], dp[5][0]]])
        dM = np.concatenate((dM, np.array([[0, 0, 1]])), axis=0)
        M = np.concatenate((M, np.array([[0, 0, 1]])), axis=0)
        M = np.dot(M, np.linalg.pinv(dM))
        M = M[0:2, :]
        p = (p + dp.T).ravel()
        change = np.linalg.norm(dp)

        count += 1

    return M
