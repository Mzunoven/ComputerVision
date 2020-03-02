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
parser.add_argument('--template_threshold', type=float, default=5,
                    help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
t_rect = rect[:]
rects = rect[:]
imH, imW, frame_num = seq.shape
update = True
t = 5
It = seq[:, :, 0]
p0 = np.zeros(2)

for i in range(frame_num-1):
    print(i+1, frame_num)
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, t_rect, threshold, num_iters, p0=p0)
    p_all = p + [t_rect[0] - rect[0], t_rect[1]-rect[1]]
    p_star = LucasKanade(seq[:, :, 0], It1, rect,
                         threshold, num_iters, p0=p_all)
    change = np.linalg.norm(p_all - p_star)

    if change < t:
        p_res = (p_star - [t_rect[0]-rect[0], t_rect[1] - rect[1]])
        t_rect[0] += p_res[0]
        t_rect[1] += p_res[1]
        t_rect[2] += p_res[0]
        t_rect[3] += p_res[1]
        It = seq[:, :, i+1]
        rects = np.vstack((rects, t_rect))
        p0 = np.zeros(2)
    else:
        rects = np.vstack(
            (rects, [t_rect[0]+p[0], t_rect[1]+p[1], t_rect[2]+p[0], t_rect[3]+p[1]]))
        p0 = p

np.save('girlseqrects-wcrt.npy', rects)
carrec = np.load('girlseqrects.npy')
carrec_wcrt = np.load('girlseqrects-wcrt.npy')
r_num = [1, 20, 40, 60, 80]
for idx in range(len(r_num)):
    i = r_num[idx]
    fig = plt.figure()
    frame = seq[:, :, i]
    rec = carrec[i, :]
    rec_wcrt = carrec_wcrt[i, :]
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    patch1 = patches.Rectangle((rec[0], rec[1]), (rec[2]-rec[0]),
                               (rec[3]-rec[1]), edgecolor='b', facecolor='none', linewidth=2)
    patch2 = patches.Rectangle((rec_wcrt[0], rec_wcrt[1]), (rec_wcrt[2]-rec_wcrt[0]),
                               (rec_wcrt[3]-rec_wcrt[1]), edgecolor='r', facecolor='none', linewidth=2)
    ax = plt.gca()
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    fig.savefig('../results/girlseq_wrct'+str(i)+'.png', bbox_inches='tight')
