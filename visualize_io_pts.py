import pickle
import open3d as o3d 
import pydisco_utils
import numpy as np

data = pickle.load(open('/Users/shamitlal/Desktop/insideoutside.p', 'rb'))
inside = data['inside']
outside = data['outside']

pydisco_utils.visualize_o3d(np.expand_dims(inside, axis=0))
pydisco_utils.visualize_o3d(np.expand_dims(outside, axis=0))