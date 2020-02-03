import os
import multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from multiprocessing import Pool
from opts import get_opts


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # ----- TODO -----
    # check input image channels
    if len(img.shape) < 3:
        img = np.matlib.repmat(img, 3, 1)
    if img.shape[-1] > 3:
        img = img[:, :, 0:3]

    lab_img = skimage.color.rgb2lab(img)
    scales = opts.filter_scales

    scale_size = len(scales)
    filter_num = 4
    filter_bank = scale_size * filter_num  # F

    hei, wid = img.shape[0], img.shape[1]
    filter_responses = np.zeros((hei, wid, 3*filter_bank))

    for i in range(scale_size):
        for c in range(3):
            filter_responses[:, :, i*filter_num*3+c+3*0] = scipy.ndimage.gaussian_filter(
                lab_img[:, :, c], scales[i])
            filter_responses[:, :, i*filter_num*3+c+3*1] = scipy.ndimage.gaussian_laplace(
                lab_img[:, :, c], scales[i])
            filter_responses[:, :, i*filter_num*3+c+3*2] = scipy.ndimage.gaussian_filter(
                lab_img[:, :, c], scales[i], [1, 0])
            filter_responses[:, :, i*filter_num*3+c+3*3] = scipy.ndimage.gaussian_filter(
                lab_img[:, :, c], scales[i], [0, 1])

    return filter_responses


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    # Load and process images
    opts = get_opts()
    index, alpha, train_files = args
    img = Image.open('../data/'+(train_files))
    img = np.array(img).astype(np.float32) / 255

    # randomly collect pixels
    filter_response = extract_filter_responses(opts, img)
    random_y = np.random.choice(filter_response.shape[0], int(alpha))
    random_x = np.random.choice(filter_response.shape[1], int(alpha))

    sub_img = filter_response[random_y, random_x, :]
    np.save(os.path.join('../temp/', str(index) + '.npy'), sub_img)


def compute_dictionary(opts, n_worker=8):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(
        join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    alpha = opts.alpha
    img_size = len(train_files)
    img_list = np.arange(img_size)
    alpha_list = np.ones(img_size) * alpha

    worker = Pool(n_worker)
    args = list(zip(img_list, alpha_list, train_files))
    worker.map(compute_dictionary_one_image, args)

    filter_responses = []
    for i in range(img_size):
        temp_files = np.load('../temp/' + str(i)+'.npy')
        filter_responses.append(temp_files)

    filter_responses = np.concatenate(filter_responses, axis=0)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dict = kmeans.cluster_centers_

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dict)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    # ----- TODO -----
    hei = img.shape[0]
    wid = img.shape[1]

    filted_img = extract_filter_responses(opts, img)
    filted_img = filted_img.reshape(hei*wid, dictionary.shape[-1])
    dis = scipy.spatial.distance.cdist(filted_img, dictionary, 'euclidean')
    res = np.argmin(dis, axis=1)
    wordmap = res.reshape(hei, wid)
    return wordmap
