# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2
from mpl_toolkits import mplot3d
from matplotlib import cm


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    cx = res[0] / 2
    cy = res[1] / 2
    # y, x = np.ogrid[cx-r:cx+r+1, cy-r:cy+r+1]
    x, y = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    x = pxSize * (x - cx) + center[0]
    y = pxSize * (y - cy) + center[1]
    z = rad ** 2 - x ** 2 - y ** 2
    mask = z < 0
    z[mask] = 0.
    z = np.sqrt(z)

    pts = np.stack((x, y, z), axis=2).reshape((res[0]*res[1], -1))
    pts = (pts.T / np.linalg.norm(pts, axis=1).T).T
    # print(pts[:, 0], pts[:, 1], pts[:, 2])

    image = np.dot(pts, light).reshape((res[1], res[0]))
    # print(image.shape)
    image[mask] = 0.
    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    P = 0
    for i in range(7):
        img_path = path + "input_{}.tif".format(i+1)
        lin_img = cv2.imread(img_path)
        lin_img_gray = cv2.cvtColor(lin_img, cv2.COLOR_BGR2GRAY)
        if I is None:
            h, w = lin_img_gray.shape
            P = h * w
            I = np.zeros((7, P))
        I[i, :] = np.reshape(lin_img_gray, (1, P))

    l_vec = np.load(path + "sources.npy")
    L = l_vec.T
    # print(L.shape)

    s = (h, w)
    # print(s)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(np.dot(L, L.T)).dot(L).dot(I)
    return B


def estimateAlbedosNormals(B):
    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    epsilon = 1e-6
    normals = B / (albedos + epsilon)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape((albedos/np.max(albedos)), s)
    normalIm = np.reshape(((normals+1.)/2.).T, (s[0], s[1], 3))

    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    epsilon = 1e-6
    zx = np.reshape(normals[0, :]/(-normals[2, :] + epsilon), s)
    zy = np.reshape(normals[1, :]/(-normals[2, :] + epsilon), s)
    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):
    """
    Question 1 (i)

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    h, w = surface.shape
    y, x = np.arange(h), np.arange(w)
    fig = plt.figure()
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, surface, edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Surface Plot')
    plt.show()


def normalize(img):
    min_v, max_v = np.min(img), np.max(img)
    img = (img - min_v) / (max_v - min_v)
    return img


if __name__ == '__main__':

    # Put your main code here

    # Q1.b
    center = np.asarray([0., 0., 0.])
    rad = 7.5
    lights = np.asarray([[1, 1, 1]/np.sqrt(3), [1, -1, 1] /
                         np.sqrt(3), [-1, -1, 1]/np.sqrt(3)])
    pxSize = 7e-3
    res = np.asarray([3840, 2160])
    # for i in range(len(lights)):
    # print(light)
    # image = renderNDotLSphere(center, rad, lights[i], pxSize, res)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.imwrite('../results/q1b_{}.png'.format(i+1), (image*255))

    # Q1.c
    I, L, s = loadData()

    # Q1.d
    u, v, vh = np.linalg.svd(I, full_matrices=False)
    # print(v)

    # Q1.e
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Albedos Image
    # plt.imshow(albedoIm, cmap='gray')
    # cv2.imwrite('../results/q1e_albedo.png', (albedoIm*255))
    # plt.show()

    # Normals Image
    normalIm = normalize(normalIm)
    # plt.imshow(normalIm, cmap='rainbow')
    # plt.savefig('../results/q1e_normal.png')
    # plt.show()

    # Q1.i
    surface = normalize(estimateShape(normals, s))
    # surface *= 255
    # plotSurface(surface)
