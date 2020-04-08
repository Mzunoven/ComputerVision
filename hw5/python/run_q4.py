import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(
        skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1)
    print('\n')
    print(img)

    # plt.imshow(1-bw, cmap='gray')
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                         fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    box_all = []
    box_all.append([])
    line_num = 1
    bboxes.sort(key=lambda x: x[2])
    bot = bboxes[0][2]
    for b in bboxes:
        minr, minc, maxr, maxc = b
        if(minr >= bot):
            bot = maxr
            box_all.append([])
            line_num += 1
        box_all[line_num-1].append(b)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    ##########################
    ##### your code here #####
    ##########################
    for row in box_all:
        line = ""
        row.sort(key=lambda x: x[1])
        r = row[0][3]
        for box in row:
            minr, minc, maxr, maxc = box
            if (img == '01_list.jpg'):
                if(minc-r > 1.2*(maxc-minc)):
                    line += " "
            else:
                if(minc-r > 0.8*(maxc-minc)):
                    line += " "
            r = maxc

            letter = bw[minr:maxr, minc:maxc]
            # hei, wid = letter.shape
            # print(letter.shape)

            if (img == '01_list.jpg'):
                letter = np.pad(letter, ((20, 20), (20, 20)),
                                'constant', constant_values=0.0)
                letter = skimage.transform.resize(letter, (32, 32))
                letter = skimage.morphology.dilation(
                    letter, skimage.morphology.square(1))
            else:
                letter = np.pad(letter, ((50, 50), (50, 50)),
                                'constant', constant_values=0.0)
                letter = skimage.transform.resize(letter, (32, 32))
                letter = skimage.morphology.dilation(
                    letter, skimage.morphology.square(2))
            letter = 1.0 - letter
            # plt.imshow(letter)
            # plt.show()
            letter = letter.T
            print(letter)

            x = letter.reshape(1, 32*32)
            h = forward(x, params, 'layer1')
            probs = forward(h, params, 'output', softmax)
            idx = np.argmax(probs[0, :])
            line += letters[idx]
        print(line)
