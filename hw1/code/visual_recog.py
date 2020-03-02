import os
import math
import multiprocessing
from os.path import join
from copy import copy
from opts import get_opts

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir

    hist, _ = np.histogram(wordmap.flatten(), bins=np.arange(0, K+1))
    hist = hist / np.sum(hist)

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    # ----- TODO -----
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)

    hist_all = np.array([])
    for l in range(L, -1, -1):
        layer = pow(2, l)
        if l > 1:
            weight = pow(2, l-L-1)
        else:
            weight = pow(2, -L)
        sub_map = [sub_mat for sub_r in np.array_split(
            wordmap, layer, axis=0) for sub_mat in np.array_split(sub_r, layer, axis=1)]
        for total in range(layer * layer):
            hist = get_feature_from_wordmap(opts, sub_map[total])
            hist_all = np.hstack(hist)
        hist_all = hist_all * weight

    if np.sum(hist_all) > 0:
        return hist_all/np.sum(hist_all)
    else:
        return hist_all


def get_image_feature(para_list):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    i, img_path, labels = para_list
    opts = get_opts()
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)

    img = Image.open('../data/'+(img_path))
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    features = get_feature_from_wordmap_SPM(opts, wordmap)

    word_hist = get_feature_from_wordmap(opts, wordmap)
    np.savez("../temp/" + "train_"+str(i)+".npz",
             features=features, labels=labels, allow_pickle=True)

    # return feature


def build_recognition_system(opts, n_worker=8):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)

    img_len = len(train_files)
    img_list = np.arange(img_len)
    features = []
    para_list = list(zip(img_list, train_files, train_labels))

    # ----- TODO -----
    pool = multiprocessing.Pool(n_worker)
    pool.map(get_image_feature, para_list)

    for i in range(img_len):
        temp = np.load('../temp/'+'train_'+str(i)+'.npz', allow_pickle=True)
        feat = temp['features']
        features.append(feat)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
                        )


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    hist_min = np.minimum(word_hist, histograms)
    sim = np.sum(hist_min, axis=1)

    return sim


def evaluate_recognition_system(opts, n_worker=8):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    dict_size = len(dictionary)

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    features = trained_system['features']
    train_labels = trained_system['labels']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    test_len = len(test_files)

    # ----- TODO -----
    conf = np.zeros((8, 8), dtype=int)

    for i in range(test_len):
        img = Image.open(data_dir + '/' + (test_files[i]))
        img = np.array(img).astype(np.float32)/255
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        test_feat = get_feature_from_wordmap_SPM(opts, wordmap)
        sim = distance_to_set(test_feat, features)
        pre_label = train_labels[np.argmax(sim)]
        true_label = test_labels[i]
        conf[true_label, pre_label] += 1

    acc = np.trace(conf) / np.sum(conf)
    # print('conf:', conf)
    # print('accuracy', acc)

    return conf, acc
