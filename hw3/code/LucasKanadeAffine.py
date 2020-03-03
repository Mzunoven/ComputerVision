import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p0 = np.zeros(6)
    p = p0
    imH0, imW0 = It.shape
    imH1, imW1 = It1.shape
    wid = imW0
    hei = imH0
    spline0 = RectBivariateSpline(np.linspace(
        0, imH0, num=imH0, endpoint=False), np.linspace(0, imW0, num=imW0, endpoint=False), It)
    spline1 = RectBivariateSpline(np.linspace(
        0, imH1, num=imH1, endpoint=False), np.linspace(0, imW1, num=imW1, endpoint=False), It1)

    change = 1
    count = 1
    x, y = np.mgrid[0:imW0, 0:imH0]
    x = x.reshape(1, hei*wid)
    y = y.reshape(1, hei*wid)
    co = np.vstack((x, y, np.ones((1, hei*wid))))

    while change > threshold and count < num_iters:
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]]])
        homop = np.dot(M, co)
        xp = homop[0]
        yp = homop[1]
        x_res = (np.where(xp >= imW0) or np.where(xp < 0))
        y_res = (np.where(yp >= imH0) or np.where(yp < 0))

        if np.shape(x_res)[1] == 0 and np.shape(y_res)[1] == 0:
            res = []
        elif np.shape(x_res)[1] != 0 and np.shape(y_res)[1] == 0:
            res = x_res
        elif np.shape(x_res)[1] == 0 and np.shape(y_res)[1] != 0:
            res = y_res
        else:
            res = np.unique(np.concatenate((x_res, y_res), 0))

        x_new = np.delete(x, res)
        y_new = np.delete(y, res)
        xp = np.delete(xp, res)
        yp = np.delete(yp, res)
        dpx = spline1.ev(yp, xp, dy=1).flatten()
        dpy = spline1.ev(yp, xp, dx=1).flatten()
        It1_p = spline1.ev(yp, xp).flatten()
        It_p = spline0.ev(y_new, x_new).flatten()

        x_new = x_new.reshape(len(x_new), 1)
        y_new = y_new.reshape(len(y_new), 1)
        xp = xp.reshape(len(xp), 1)
        yp = yp.reshape(len(yp), 1)
        dpx = dpx.reshape(len(dpx), 1)
        dpy = dpy.reshape(len(dpy), 1)

        A1 = np.multiply(x_new, dpx)
        A2 = np.multiply(y_new, dpx)
        A3 = np.multiply(x_new, dpy)
        A4 = np.multiply(y_new, dpy)

        A = np.hstack((A1, A2, dpx, A3, A4, dpy))
        b = np.reshape(It_p-It1_p, (len(xp), 1))

        dp = np.linalg.pinv(A).dot(b)
        change = np.linalg.norm(dp)
        p = (p + dp.T).ravel()

        count += 1
    M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]]])

    return M
