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
    print(frame_num)
    f = book[frame_num]
    p = ar_src[frame_num]
    h = cv_cover.shape[0]
    w = cv_cover.shape[1]

    resized = np.zeros(cv_cover.shape)
    resized[:, :, 0] = cv2.resize(p[:, :, 0], (w, h))
    resized[:, :, 1] = cv2.resize(p[:, :, 1], (w, h))
    resized[:, :, 2] = cv2.resize(p[:, :, 2], (w, h))
    matches, l1, l2 = matchPics(f, cv_cover, opts)
    locs1, locs2 = [], []

    for i in range(len(matches[:, 0])):
        locs1.append(l1[matches[:, 0][i]])
        locs2.append(l2[matches[:, 1][i]])

    locs1 = np.array(locs1)
    locs2 = np.array(locs2)

    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)

    composite_img = compositeH(bestH2to1, f, resized)
    composite_img = composite_img.astype('uint8')
    composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
    writer1.append_data(composite_img)

writer1.close()
elapsed1 = time.time() - t1
print("Total time for creating videos for loop: ", elapsed1)
