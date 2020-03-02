import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt

import imageio
# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from loadVid import loadVid
from planarH import compositeH
import time


def ar_func(ind, ar_src, cv_cover, cv_cov_vid, opts):
    im = ar_src[44:311, :, :]
    cropped_im = cv2.resize(im, (im.shape[1], cv_cover.shape[0]))
    print('Frame Number Function: ', ind)
    matches, locs1, locs2 = matchPics(cv_cover, cv_cov_vid, opts)
    locs1 = locs1[matches[:, 0], 0:2]
    locs2 = locs2[matches[:, 1], 0:2]
    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
    resize_im = cv2.resize(im, (im.shape[1], cv_cover.shape[0]))
    cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2))
                                  : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)), :]

    composite_img = compositeH(bestH2to1, cropped_im, cv_cov_vid)
    return composite_img


opts = get_opts()

frm_np = np.zeros(())
t = time.time()

ar_src = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
book = loadVid('../data/book.mov')
elapsed = time.time() - t
print("Elapsed time for loading videos: ", elapsed)


t1 = time.time()
writer1 = imageio.get_writer('../result/norm_for_final.avi', fps=25)

for frame_num in range(ar_src.shape[0]):
    if (frame_num < 50 or frame_num > 75):
        if (frame_num != 81 and frame_num != 119 and frame_num != 149 and frame_num != 151 and frame_num != 178 and frame_num != 179 and frame_num != 202 and frame_num != 253 and frame_num != 322 and frame_num != 364 and frame_num != 378 and frame_num != 384 and frame_num != 388 and frame_num != 393 and frame_num != 397 and frame_num != 398 and frame_num != 399 and frame_num != 422 and frame_num != 435 and frame_num != 436 and frame_num != 437 and frame_num != 438):
            print("Iteration: ", frame_num)
            composite_img = ar_func(
                frame_num, ar_src[frame_num], cv_cover, book[frame_num], opts)
            writer1.append_data(composite_img)

writer1.close()
elapsed1 = time.time() - t1
print("Total time for creating videos for loop: ", elapsed1)
