import numpy as np
import cv2
from matchPics import matchPics
import scipy
from opts import get_opts
import matplotlib.pyplot as plt
from helper import plotMatches


# Q2.1.6
# Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
opts = get_opts()
match_num = []

for i in range(36):
    # Rotate Image
    rot_angle = i * 10
    cv_rot = scipy.ndimage.rotate(cv_cover, rot_angle, reshape=False)

    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, cv_rot, opts)
    match_num.append(len(matches))

# Update histogram
# Display histogram

plt.figure(1)
plt.bar(np.arange(36), match_num, color='blue', alpha=0.75)
plt.xlabel('rotation angles/10')
plt.ylabel('Number of matches')
plt.title('Number of Matches for different angles')

plt.show()
