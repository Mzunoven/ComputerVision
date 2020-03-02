import numpy as np
from numpy import matlib
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0
    imH0, imW0 = It.shape
    imH1, imW1 = It1.shape

    x1, y1, x2, y2 = rect

    wid = int(x2 - x1)
    hei = int(y2 - y1)

    spline0 = RectBivariateSpline(np.linspace(
        0, imH0, num=imH0, endpoint=False), np.linspace(0, imW0, num=imW0, endpoint=False), It)
    spline1 = RectBivariateSpline(np.linspace(
        0, imH1, num=imH1, endpoint=False), np.linspace(0, imW1, num=imW1, endpoint=False), It1)

    change = 1
    counter = 1
    x, y = np.mgrid[x1:x2+1:wid*1j, y1:y2+1:hei*1j]
    while (change > threshold) and (counter < num_iters):
        dpx = spline1.ev(y+p[1], x+p[0], dy=1).flatten()
        dpy = spline1.ev(y+p[1], x+p[0], dx=1).flatten()
        It1_p = spline1.ev(y+p[1], x+p[0]).flatten()
        It_p = spline0.ev(y, x).flatten()

        A = np.zeros((wid*hei, 2*wid*hei))
        for i in range(wid*hei):
            A[i, 2*i] = dpx[i]
            A[i, 2*i+1] = dpy[i]
        A = np.dot(A, (matlib.repmat(np.eye(2), wid*hei, 1)))
        b = np.reshape(It_p-It1_p, (wid*hei, 1))
        dp = np.linalg.pinv(A).dot(b)
        change = np.linalg.norm(dp)
        p = (p + dp.T).ravel()

        counter += 1

    return p
