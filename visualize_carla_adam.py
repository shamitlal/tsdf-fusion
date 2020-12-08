import numpy as np
import ipdb 
st = ipdb.set_trace
import matplotlib.pyplot as plt 
a = np.load('/Users/shamitlal/Desktop/temp/tsdf/city_01_vehicles_150_episode_0039_2020-05-02_cam0_startframe000580_obj8.npz')
for rgb in a['rgb_camRs']:
    plt.imshow(rgb)
    plt.show(block=True)

for rgb in a['rgb_cam0s']:
    plt.imshow(rgb)
    plt.show(block=True)

for rgb in a['rgb_cam1s']:
    plt.imshow(rgb)
    plt.show(block=True)

for rgb in a['rgb_cam2s']:
    plt.imshow(rgb)
    plt.show(block=True)

for rgb in a['rgb_cam3s']:
    plt.imshow(rgb)
    plt.show(block=True)
st()
aa=1