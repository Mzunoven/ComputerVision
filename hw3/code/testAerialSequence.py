import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3,
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2,
                    help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2,
                    help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

imH, imW, frame_num = seq.shape

for i in range(frame_num-1):
    print(i+1, frame_num)
    image1 = seq[:, :, i]
    image2 = seq[:, :, i+1]
    mask = SubtractDominantMotion(
        image1, image2, threshold, num_iters, tolerance)
    objects = np.where(mask == 0)

    if (i+1) % 30 == 0 and i < 120:
        fig = plt.figure()
        plt.imshow(image2, cmap='gray')
        plt.axis('off')
        plt.plot(objects[1], objects[0], '.',
                 markerfacecolor='blue', markeredgecolor='None')
        fig.savefig('../results/aerialseq'+str(i+1) +
                    '.png', bbox_inches='tight')
