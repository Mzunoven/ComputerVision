'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import helper
import submission

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(im1.shape)

pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']

K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']
# print(K1.shape)

F = submission.eightpoint(pts1, pts2, M)
E = submission.essentialMatrix(F, K1, K2)

M1 = np.eye(3)
M1 = np.hstack((M1, np.zeros((3, 1))))
M2_all = np.zeros((3, 4, 4))
# print(M2_all)
M2_all = helper.camera2(E)

C1 = np.dot(K1, M1)
cur_err = np.inf

for i in range(M2_all.shape[2]):
	C2 = np.dot(K2, M2_all[:, :, i])
	w, err = submission.triangulate(C1, pts1, C2, pts2)

	if err < cur_err:
		cur_err = err
		M2 = M2_all[:, :,i]
		C2_opt = C2
		w_opt = w

np.savez('q3_3.npz', M2 = M2, C2 = C2_opt, P = w_opt)