from skspatial.objects import Plane
from skspatial.objects import Points
from skspatial.plotting import plot_3d
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

import numpy as np
# points = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]])
# plydata = PlyData.read("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_tryy/points/sample.ply")
plydata = PlyData.read("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_normcam2_confcolordir_KNN8_LRelu_grid800_scan115_wbg_dmsk/points/light_brown_sample.ply")
vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
pts = np.zeros([vertex_data.size, 3])
color = np.zeros_like(pts)
pts[:, 0] = vertex_data['x']
pts[:, 1] = vertex_data['y']
pts[:, 2] = vertex_data['z']
# print(vertex_data)
color[:, 0] = vertex_data['red']
color[:, 1] = vertex_data['green']
color[:, 2] = vertex_data['blue']

points = Points(pts)


plane = Plane.best_fit(points)
print("plane", plane, pts.shape)
print("plane", plane.point, plane.normal)
print("color average", np.mean(color, axis=0))

coord=plane.point
coeff=plane.normal


# Plane(point=Point([-0.49666997,  0.52160616,  3.6239593 ]), normal=Vector([-0.11364093,  0.38778102,  0.91471942])) (15565, 3)

# a(x − x0) + b(y − y0) + c(z − z0) = 0


r=8

# a,b,c = coeff[0], coeff[1], coeff[2]
# x0,y0,z0=coord[0],coord[1],coord[2],
# xy = r * np.random.rand(int(1e5),2) - r/2
# z = (a*(xy[...,0]-x0) + b*(xy[...,1]-y0))/(-c) + z0
# gen_pnts = np.stack([xy[...,0], xy[...,1], z], axis=-1)
#
# np.savetxt('/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_tryy/points/planes_white.txt', gen_pnts, delimiter=';')




# plane Plane(point=Point([ 0.20770223, -0.74818161,  3.98697683]), normal=Vector([-0.11165793,  0.3806543 ,  0.91795142])) (14801, 3)
# plane [ 0.20770223 -0.74818161  3.98697683] [-0.11165793  0.3806543   0.91795142]
# color average [150.72447808  99.68367002  63.40976961]

a,b,c = coeff[0], coeff[1], coeff[2]
x0,y0,z0=coord[0],coord[1],coord[2],
xy = r * np.random.rand(int(1e5),2) - r/2
z = (a*(xy[...,0]-x0) + b*(xy[...,1]-y0))/(-c) + z0
gen_pnts = np.stack([xy[...,0], xy[...,1], z], axis=-1)
color = np.ones_like(gen_pnts) * np.mean(color, axis=0, keepdims=True)
gen_pnts = np.concatenate([gen_pnts, color], axis=-1)
np.savetxt('/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_normcam2_confcolordir_KNN8_LRelu_grid800_scan115_wbg_dmsk/points/planes_light_brown.txt', gen_pnts, delimiter=';')




# plane Plane(point=Point([-0.04889537, -0.84123057,  4.03164617]), normal=Vector([-0.11154823,  0.3783277 ,  0.91892608])) (30712, 3)
# plane [-0.04889537 -0.84123057  4.03164617] [-0.11154823  0.3783277   0.91892608]

# a,b,c = coeff[0], coeff[1], coeff[2]
# x0,y0,z0=coord[0],coord[1],coord[2],
# xy = r * np.random.rand(int(1e5),2) - r/2
# z = (a*(xy[...,0]-x0) + b*(xy[...,1]-y0))/(-c) + z0
# gen_pnts = np.stack([xy[...,0], xy[...,1], z], axis=-1)
# color = np.ones_like(gen_pnts) * np.mean(color, axis=0, keepdims=True)
# gen_pnts = np.concatenate([gen_pnts, color], axis=-1)
# np.savetxt('/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_normcam2_confcolordir_KNN8_LRelu_grid800_scan115_wbg_dmsk/points/planes_brown.txt', gen_pnts, delimiter=';')




# gen_points = Points(gen_pnts)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_3d(
#     gen_points.plotter(c='k', s=50, depthshade=False),
#     plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
# )
#
# plt.show()