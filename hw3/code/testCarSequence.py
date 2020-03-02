import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4,
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2,
                    help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
imH, imW, frames = seq.shape

rects = rect[:]
for i in range(frames-1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    rects = np.vstack((rects, rect))

    if (i+1) % 100 == 0 or i == 0:
        fig = plt.figure()
        plt.imshow(It1, cmap='gray')
        plt.axis('off')
        plt.axis('tight')
        patch = patches.Rectangle((rect[0], rect[1]), (rect[2]-rect[0]),
                                  (rect[3]-rect[1]), edgecolor='r', facecolor='none', linewidth=2)
        ax = plt.gca()
        ax.add_patch(patch)
        fig.savefig('../results/carseq'+str(i+1)+'.png', bbox_inches='tight')
np.save('carseqrects.npy', rects)
