import numpy as np
import cv2
# Import necessary functions
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
import matplotlib.pyplot as plt

# Write script for Q4.2x

opts = get_opts()
left_img = cv2.imread('../data/pano_left.jpg')
right_img = cv2.imread('../data/pano_right.jpg')
imH1, imW1, _ = left_img.shape
imH2, imW2, _ = right_img.shape
width = round(max(imW1, imW2)*1.2)

im2 = cv2.copyMakeBorder(right_img, 0, imH2 - imH1, width-imW2,
                         0, cv2.BORDER_CONSTANT, 0)
matches, locs1, locs2 = matchPics(left_img, im2, opts)
locs1 = locs1[matches[:, 0], 0:2]
locs2 = locs2[matches[:, 1], 0:2]
bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
pano_im = compositeH(bestH2to1, left_img, im2)

pano_im = np.maximum(im2, pano_im)

cv2.imwrite('../../results/pana.png', pano_im)
