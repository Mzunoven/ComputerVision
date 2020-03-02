import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts


# Import necessary functions
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

# Write script for Q2.2.4
opts = get_opts()

cv_desk = cv2.imread('../data/cv_desk.png')
cv_cover = cv2.imread('../data/cv_cover.jpg')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

hp_cover_resize = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

locs1 = locs1[matches[:, 0], 0:2]
locs2 = locs2[matches[:, 1], 0:2]
bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
composite_img = compositeH(bestH2to1, hp_cover_resize, cv_desk)

plt.imshow(composite_img)
plt.show()
