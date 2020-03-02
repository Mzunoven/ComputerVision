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

h = cv_cover.shape[0]
w = cv_cover.shape[1]

resized = np.zeros(cv_cover.shape)
resized[:, :, 0] = cv2.resize(hp_cover[:, :, 0], (w, h))
resized[:, :, 1] = cv2.resize(hp_cover[:, :, 1], (w, h))
resized[:, :, 2] = cv2.resize(hp_cover[:, :, 2], (w, h))

matches, l1, l2 = matchPics(cv_desk, cv_cover, opts)
locs1, locs2 = [], []

for i in range(len(matches[:, 0])):
    locs1.append(l1[matches[:, 0][i]])
    locs2.append(l2[matches[:, 1][i]])

locs1 = np.array(locs1)
locs2 = np.array(locs2)

bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)

composite_img = compositeH(bestH2to1, cv_desk, resized)
composite_img = composite_img.astype('uint8')
composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
plt.imshow(composite_img)
plt.show()
