'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import helper
import submission
from mpl_toolkits.mplot3d import Axes3D

data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(im1.shape)

K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']

temple_coord = np.load('../data/templeCoords.npz')
x1 = temple_coord['x1']
y1 = temple_coord['y1']
temple1 = np.hstack((x1, y1))

F = submission.eightpoint(pts1, pts2, M)
E = submission.essentialMatrix(F, K1, K2)
x2 = np.empty((x1.shape[0], 1))
y2 = np.empty((x1.shape[0], 1))

for i in range(x1.shape[0]):
	corresp = submission.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
	x2[i] = corresp[0]
	y2[i] = corresp[1]
temple2 = np.hstack((x2, y2))

M1 = np.hstack((np.eye(3), np.zeros((3,1))))
M2_mat = helper.camera2(E)
C1 = np.dot(K1, M1)

cur_err = np.inf
for i in range(M2_mat.shape[2]):
	C2 = np.dot(K2, M2_mat[:, :, i])
	w, err = submission.triangulate(C1, temple1, C2, temple2)

	if err<cur_err and np.min(w[:, 2])>=0:
		cur_err = err
		M2 = M2_mat[:, :, i]
		C2_opt = C2
		w_opt = w

np.savez('q4_2.npz', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2_opt)

fig = plt.figure()
res = Axes3D(fig)
res.set_xlim3d(np.min(w_opt[:,0]),np.max(w_opt[:,0]))
res.set_ylim3d(np.min(w_opt[:,1]),np.max(w_opt[:,1]))
res.set_zlim3d(np.min(w_opt[:,2]),np.max(w_opt[:,2]))
res.set_xlabel('X')
res.set_ylabel('Y')
res.set_zlabel('Z')
res.scatter(w_opt[:,0],w_opt[:,1],w_opt[:,2])
plt.show()