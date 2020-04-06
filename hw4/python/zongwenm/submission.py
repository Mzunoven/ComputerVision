"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import pdb
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    A = np.empty((pts1.shape[0], 9))

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1,  y2 * y1,
                   y2, x1, y1, np.ones(pts1.shape[0]))).T
    u, s, v = np.linalg.svd(A)

    F = v[-1].reshape(3, 3)
    F = helper.refineF(F, pts1, pts2)
    F = helper._singularize(F)

    F = np.dot((np.dot(T.T, F)), T)

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    A = np.empty((pts1.shape[0], 9))

    T = np.array([1/M, 0, 0], [0, 1/M, 0], [0, 0, 1])
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1,  y2 * y1,
                   y2, x1, y1, np.ones(pts1.shape[0]))).T
    u, s, v = np.linalg.svd(A)

    f1 = v[-1].reshape(3, 3)
    f2 = v[-2].reshape(3, 3)

    def dF(coeff): return np.linalg.det(coeff*f1 + (1-coeff)*f2)

    a0 = dF(0)
    a1 = 2*(dF(1) - dF(-1))/3 - (dF(2) - dF(-2))/12
    a2 = (dF(1) + dF(-1)) / 2 - a0
    a3 = (dF(1) + dF(-1)) / 2 - a1

    sol = np.roots([a3, a2, a1, a0])

    mat = [root*f1 + (1-root)*f2 for root in sol]
    mat = [helper.refineF(F, pts1, pts2) for F in mat]

    F = [np.dot((np.dot(T.T, F)), T) for F in mat]

    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.dot((np.dot(K2.T, F)), K1)

    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n, temp = pts1.shape
    P = np.zeros((n, 3))
    Phomo = np.zeros((n, 4))
    for i in range(n):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        A1 = x1*C1[2, :] - C1[0, :]
        A2 = y1*C1[2, :] - C1[1, :]
        A3 = x2*C2[2, :] - C2[0, :]
        A4 = y2*C2[2, :] - C2[1, :]
        A = np.vstack((A1, A2, A3, A4))
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        P[i, :] = p[0:3]
        Phomo[i, :] = p
        # print(p)
    p1_proj = np.matmul(C1, Phomo.T)
    lam1 = p1_proj[-1, :]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2, Phomo.T)
    lam2 = p2_proj[-1, :]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0, 1], :].T-pts1)**2)
    err2 = np.sum((p2_proj[[0, 1], :].T-pts2)**2)
    err = err1 + err2

    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1 = int(x1)
    y1 = int(y1)
    rect_size = 20  # window size to adjust
    im1_sect = im1[(y1 - rect_size//2): (y1 + rect_size//2 + 1),
                   (x1 - rect_size//2): (x1 + rect_size//2 + 1), :]

    im2_h, im2_w, _ = im2.shape

    p1 = np.array([x1, y1, 1])

    epi_line = np.dot(F, p1)
    ep_l = epi_line / np.linalg.norm(epi_line)
    a, b, c = ep_l

    ep2_y = np.arange(im2_h)
    ep2_x = np.rint(-(ep_l[1]*ep2_y + ep_l[2]) / ep_l[0])

    # Guassian
    rect_v = np.arange(-rect_size//2, rect_size//2+1, 1)
    rect_x, rect_y = np.meshgrid(rect_v, rect_v)
    dev = 7
    weight = np.dot((np.exp(-((rect_x**2 + rect_y**2) / (2 * (dev**2))))), 1)
    weight = weight / np.sqrt(2*np.pi*dev**2)
    weight = np.sum(weight)
    cur_err = 1e5

    for y2 in range((y1 - rect_size//2), (y1 + rect_size//2 + 1)):
        x2 = int((-b*y2-c) / a)
        if (x2 >= rect_size//2 and x2 + rect_size//2 < im2_w and y2 >= rect_size//2 and y2 + rect_size//2 < im2_h):
            im2_sect = im2[y2-rect_size//2:y2+rect_size //
                           2+1, x2-rect_size//2:x2+rect_size//2+1, :]
            err = np.linalg.norm((im1_sect - im2_sect) * weight)
            if err < cur_err:
                cur_err = err
                x2_opt = x2
                y2_opt = y2
    return x2_opt, y2_opt


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    max_inliers = -1

    p1_hom = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
    p2_hom = np.vstack((pts2.T, np.ones((1, pts1.shape[0]))))

    for idx in range(nIters):
        total_inliers = 0
        rand_idx = np.random.choice(pts1.shape[0], 8)
        rand1 = pts1[rand_idx, :]
        rand2 = pts2[rand_idx, :]

        F = eightpoint(rand1, rand2, M)
        pred_x2 = np.dot(F, p1_hom)
        pred_x2 = pred_x2 / np.sqrt(np.sum(pred_x2[:2, :]**2, axis=0))

        err = abs(np.sum(p2_hom*pred_x2, axis=0))
        n_inliers = err < tol
        # print(n_inliers)
        total_inliers = n_inliers[n_inliers.T].shape[0]
        if total_inliers > max_inliers:
            F_opt = F
            max_inliers = total_inliers
            inliers = n_inliers

    return F_opt, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation
    t = np.linalg.norm(r)
    if t == 0:
        return np.eye(3)
    u = r / t
    u1 = u[0, 0]
    u2 = u[1, 0]
    u3 = u[2, 0]
    ucross = np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]])
    R = np.eye(3) * np.cos(t) + (1-np.cos(t)) * \
        np.dot(u, u.T) + np.sin(t) * ucross
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation
    tol = 1e-2
    A = (R - R.T) / 2
    rh = np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])
    s = np.linalg.norm(rh)
    c = (np.trace(R) - 1) / 2
    if s < tol and (c-1) < tol:
        return np.zeros((3, 1))
    elif s < tol and (c+1) < tol:
        v_tmp = R + np.eye(3)
        for i in range(R.shape[0]):
            if np.count_nonzero(v_tmp[:, i]) > 0:
                v = v_tmp[:, i]
                break
        u = v/np.linalg.norm(v)
        u_pi = u*np.pi

        if (np.linalg.norm(u_pi) == np.pi) and (u_pi[0, 0] == u_pi[1, 0] == 0 and u_pi[2, 0] < 0) or (u_pi[0, 0] == 0 and u_pi[1, 0] < 0) or (u_pi[0, 0] < 0):
            r = -u_pi
        else:
            r = u_pi
        return r

    else:
        u = rh / s
        t = np.arctan2(s, c)
        r = t * u
        return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    C1 = np.dot(K1, M1)
    P = x[:-6].reshape(-1, 3)
    r2 = x[-6:-3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2)).reshape(3, 4)  # Extrinsics of camera 2
    C2 = np.dot(K2, M2)
    P_hom = np.vstack((P.T, np.ones((1, P.shape[0]))))

    p1_hat = np.zeros((2, P_hom.shape[1]))
    p2_hat = np.zeros((2, P_hom.shape[1]))

    x1_hom = np.dot(C1, P_hom)
    x2_hom = np.dot(C2, P_hom)

    p1_hat[0, :] = (x1_hom[0, :] / x1_hom[2, :])
    p1_hat[1, :] = (x1_hom[1, :]/x1_hom[2, :])
    p2_hat[0, :] = (x2_hom[0, :]/x2_hom[2, :])
    p2_hat[1, :] = (x2_hom[1, :] / x2_hom[2, :])
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T

    residuals = np.concatenate(
        [(p1 - p1_hat).reshape(-1), (p2 - p2_hat).reshape(-1)])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2_0 = M2_init[:, 0:3]
    t2_0 = M2_init[:, 3]
    r2_0 = invRodrigues(R2_0)
    def fun(x): return (rodriguesResidual(K1, M1, p1, K2, p2, x))
    x0 = P_init.flatten()
    x0 = np.append(x0, r2_0.flatten())
    x0 = np.append(x0, t2_0.flatten())

    x_opt, _ = scipy.optimize.leastsq(fun, x0)
    P2 = x_opt[0:-6].reshape(-1, 3)
    r2 = x_opt[-6:-3].reshape(3, 1)
    t2 = x_opt[-3:].reshape(3, 1)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2


'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    P, err = triangulate(C1, pts1[:, :2], C2, pts2[:, :2])
    return P, err


if __name__ == "__main__":

    # 2.1
    # data = np.load('../data/some_corresp.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)

    # F = eightpoint(pts1, pts2, M) # EightPoint algrithm to find F

    # print(F)
    # np.savez('q2_1.npz', F=F, M=M)
    # helper.displayEpipolarF(im1, im2, F) # Visualize result

    # 3.1
    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']
    # E = essentialMatrix(F, K1, K2)
    # print(E)

    # 4.1
    # x1 = pts1[10,0]
    # y1 = pts1[10,1]
    # x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
    # helper.epipolarMatchGUI(im1, im2, F)
    # np.savez('q4_1.npz', F = F, pts1 = pts1, pts2 = pts2)

    # 5.1
    # data = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # print(pts1.shape, pts2.shape)
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)
    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']

    # nIters = 100  # iteration number
    # tol = 0.8
    # F, inliers = ransacF(pts1, pts2, M, nIters, tol)
    # F = eightpoint(pts1, pts2, M)  # without RANSAC
    # print(inliers)
    # print("Acccuracy: ", (np.count_nonzero(inliers)/len(inliers)))
    # helper.displayEpipolarF(im1, im2, F)

    # F:
    # [[ 1.44845968e-08 -3.12242026e-07  1.07078108e-03]
    # [ 1.67046292e-07  1.00324202e-08 -8.51505084e-05]
    # [-1.04288634e-03  9.23538949e-05 -2.07767875e-03]]

    # 5.3

    # E = essentialMatrix(F, K1, K2)
    # M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # M2_all = helper.camera2(E)

    # C1 = np.dot(K1, M1)
    # err_val = np.inf

    # for i in range(M2_all.shape[2]):

    #     C2 = np.dot(K2, M2_all[:, :, i])
    #     w, err = triangulate(C1, pts1, C2, pts2)

    #     if err < err_val:
    #         err_val = err
    #         M2 = M2_all[:, :, i]
    #         C2_opt = C2
    #         w_best = w

    # P_init, err = triangulate(C1, pts1, C2_opt, pts2)
    # print('Original reprojection error: ', err)

    # bundleAdjustment
    # M2_opt, P2 = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_init)

    # C2_opt = np.dot(K2, M2_opt)
    # w_hom = np.hstack((P2, np.ones([P2.shape[0], 1])))
    # C2 = np.dot(K2, M2)
    # err_opt = 0

    # for i in range(pts1[inliers, :].shape[0]):
    #     pts1_hat = np.dot(C1, w_hom[i, :].T)
    #     pts2_hat = np.dot(C2_opt, w_hom[i, :].T)

    #     # Normalizing
    #     p1_hat_norm = (np.divide(pts1_hat[0:2], pts1_hat[2])).T
    #     p2_hat_norm = (np.divide(pts2_hat[0:2], pts2_hat[2])).T
    #     err1 = np.square(pts1[:, 0] - p1_hat_norm[0]) + \
    #         np.square(pts1[:, 1] - p1_hat_norm[0])
    #     err2 = np.square(pts2[:, 0] - p2_hat_norm[0]) + \
    #         np.square(pts2[:, 1] - p2_hat_norm[0])
    #     err_opt += np.sum((p1_hat_norm - pts1[i])
    #                       ** 2 + (p2_hat_norm - pts2[i])**2)

    # print('Error with optimized 3D points: ', err_opt)

    # fig1 = plt.figure()
    # ax1 = Axes3D(fig1)
    # ax1.set_xlim3d(np.min(P_init[:, 0]), np.max(P_init[:, 0]))
    # ax1.set_ylim3d(np.min(P_init[:, 1]), np.max(P_init[:, 1]))
    # ax1.set_zlim3d(np.min(P_init[:, 2]), np.max(P_init[:, 2]))
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')
    # ax1.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2])
    # plt.show()

    # fig2 = plt.figure()
    # ax2 = Axes3D(fig2)
    # ax2.set_xlim3d(np.min(P2[:, 0]), np.max(P2[:, 0]))
    # ax2.set_ylim3d(np.min(P2[:, 1]), np.max(P2[:, 1]))
    # ax2.set_zlim3d(np.min(P2[:, 2]), np.max(P2[:, 2]))
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')
    # ax2.scatter(P2[:, 0], P2[:, 1], P2[:, 2])
    # plt.show()

    # Original reprojection error:  514840.53051875724
    # Error with optimized 3D points:  321560.6321697503

    # 6.1

    # time_0 = np.load('../data/q6/time'+str(0)+'.npz')
    # pts1 = time_0['pts1']  # Nx3 matrix
    # pts2 = time_0['pts2']  # Nx3 matrix
    # pts3 = time_0['pts3']  # Nx3 matrix
    # M1_0 = time_0['M1']
    # M2_0 = time_0['M2']
    # M3_0 = time_0['M3']
    # K1_0 = time_0['K1']
    # K2_0 = time_0['K2']
    # K3_0 = time_0['K3']

    # 6.2
    connections_3d = [[0, 1], [1, 3], [2, 3], [2, 0], [4, 5], [6, 7], [8, 9], [9, 11], [
        10, 11], [10, 8], [0, 4], [4, 8], [1, 5], [5, 9], [2, 6], [6, 10], [3, 7], [7, 11]]
    colors = ['blue', 'blue', 'blue', 'blue', 'red', 'magenta', 'green', 'green', 'green',
              'green', 'red', 'red', 'red', 'red', 'magenta', 'magenta', 'magenta', 'magenta']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        time = np.load('../data/q6/time'+str(i)+'.npz')
        pts1 = time['pts1']
        pts2 = time['pts2']
        pts3 = time['pts3']
        M1_0 = time['M1']
        M2_0 = time['M2']
        M3_0 = time['M3']
        K1_0 = time['K1']
        K2_0 = time['K2']
        K3_0 = time['K3']
        C1_0 = np.dot(K1_0, M1_0)
        C2_0 = np.dot(K1_0, M2_0)
        C3_0 = np.dot(K1_0, M3_0)
        Thres = 200
        P_mv, err_mv = MultiviewReconstruction(
            C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)
        M2_opt, pts_3d = bundleAdjustment(
            K2_0, M2_0, pts2[:, :2], K3_0, M3_0, pts3[:, :2], P_mv)
        num_points = pts_3d.shape[0]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
        np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    # helper.plot_3d_keypoint(P2_opt)
    # np.savez('q6_1.npz', M=M2_opt, w=P2_opt)
    # img = plt.imread('../data/q6/cam3_time0.jpg')
    # helper.visualize_keypoints(img, pts3, Thres)
