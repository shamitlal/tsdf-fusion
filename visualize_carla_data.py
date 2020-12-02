import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import ipdb 
st = ipdb.set_trace
fname = "/Users/shamitlal/Desktop/temp/tsdf/raw/_cvpr21_closecam/episode_0__e61381a8-3591-11eb-981a-50e0859310cf/vehicle.audi.a2/vehicle-0_frame-2250.p"
data = pickle.load(open(fname, 'rb'))
rgb_data = data['rgb_data'][-2:]
for i in range(18):
    rgb = rgb_data[i]
    plt.imshow(rgb)
    plt.show(block=True)
st()
aa=1