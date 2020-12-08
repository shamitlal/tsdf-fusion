"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

# import cv2
import numpy as np
# import open3d
import fusion
import pydisco_utils
import torch 
import pickle 
import matplotlib.pyplot as plt
import ipdb 
st = ipdb.set_trace
import os 
import socket
hostname = socket.gethostname()

def prepare_complete_data(data):
    num_cams = 9
    rgb_list = []
    xyz_list = []
    extr_list = []
    intr_list = []
    for i in range(num_cams):
        xyz_list.append(data[f'xyz_cam{i}s'])
        extr_list.append(data[f'origin_T_cam{i}s'])
        rgb_list.append(data[f'rgb_cam{i}s'])
        intr_list.append(data['pix_T_cams'])

    data['origin_T_camXs_raw'] = np.stack(extr_list, axis=1)
    data['rgb_camXs_raw'] = np.stack(rgb_list, axis=1)
    data['xyz_camXs_raw'] = np.stack(xyz_list, axis=1)
    data['pix_T_camXs'] = np.stack(intr_list, axis=1)
    return data

if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    if 'Shamit' in hostname:
        base_modfile = "/Users/shamitlal/Desktop/temp/tsdf/tsdf.txt"
        basepath = "/Users/shamitlal/Desktop/temp/tsdf/drone_ae_processed"
        sdfpath = "/Users/shamitlal/Desktop/temp/tsdf/drone_ae_processed_tsdf"
        device = torch.device('cpu')
        voxel_size = 1
    else:
        # base_modfile = "/projects/katefgroup/datasets/shamit_carla_correct/npys/mc_cart.txt"
        basepath = "/home/mprabhud/dataset/carla/processed/npzs/new_complete_ad_s10_i1"
        sdfpath = "/projects/katefgroup/datasets/shamit_carla_correct/tsdfs/carla_complete_droneae_car_tsdf"
        device = torch.device('cuda')
        voxel_size = 0.05

    # max_depth = 10 # for mct
    max_depth = 13 # for drone_ae
    print(f"Setup: voxelsize: {voxel_size}, max_depth: {max_depth}")
    base_modfile_f = [fname for fname in os.listdir(basepath) if fname.endswith('.npz')]

    count = 0
    for filename in base_modfile_f:
        count+=1
        print(f"Processing file {filename} ... file count {count}")
        filename = filename.strip()
        picklename = filename.split('.')[0] + ".p"
        picklepath = os.path.join(basepath, filename)
        # data = pickle.load(open(picklepath, "rb"))
        data = dict(np.load(picklepath))
        print("Estimating voxel volume bounds...")
        data = prepare_complete_data(data)
        st()
        seqlen = data['rgb_cam0s'].shape[0]
        rgb =  data['rgb_cam0s']
        for seqnum in range(seqlen):
            print("Processing sequence num: ", seqnum)
            cam_intr = data['pix_T_cams'][0, :3, :3]
            cam_poses = data['origin_T_camXs_raw'][seqnum]
            xyz_camXs = data['xyz_camXs_raw'][seqnum]
            rgb_camXs = data['rgb_camXs_raw'][seqnum]
            
            cam_poses_torch = torch.tensor(cam_poses).to(device) # origin_T_cam
            camX0_T_camXI = pydisco_utils.safe_inverse(cam_poses_torch[0:1]) @ cam_poses_torch
            cam_poses = camX0_T_camXI.cpu().numpy()
            # st()
            # xyz_cam0 = pydisco_utils.apply_4x4(cam_poses, torch.tensor(xyz_camXs).float())
            # xyz_cam0 = xyz_cam0.reshape(1, -1, 3)
            # # pydisco_utils.visualize_o3d(xyz_cam0)
            depth_camXs, _ = pydisco_utils.create_depth_image(torch.tensor(data['pix_T_camXs'][0]).float().to(device), torch.tensor(xyz_camXs).float().to(device), 256, 256)
            xyz_camXs = pydisco_utils.depth2pointcloud(depth_camXs, torch.tensor(data['pix_T_camXs'][0]).float().to(device))

            pix_T_camX = pydisco_utils.get_4x4(cam_intr)
            n_imgs = depth_camXs.shape[0]
            vol_bnds = np.zeros((3,2))
            pcd_list = []
            for i in range(n_imgs):
                # Read depth image and camera pose
                depth_im = depth_camXs[i, 0]
                depth_im[depth_im >= max_depth] = 0  # set invalid depth to 0 (specific to carla mc dataset)
                cam_pose = cam_poses[i]
                # pydsico stuff
                xyz_camX = pydisco_utils.depth2pointcloud(torch.tensor(depth_im).unsqueeze(0).unsqueeze(0).float().to(device), pix_T_camX.float().to(device))
                origin_T_camX = torch.tensor(cam_pose).unsqueeze(0).float()
                xyz_camOrigin = pydisco_utils.apply_4x4(origin_T_camX.to(device), xyz_camX.to(device))
                pcd_list.append(xyz_camOrigin.cpu().numpy())
                # pydisco_utils.visualize_o3d(xyz_camOrigin)    
                # pydisco_utils.visualize_o3d(xyz_camX)    

                # Compute camera view frustum and extend convex hull
                view_frust_pts = fusion.get_view_frustum(depth_im.cpu().numpy(), cam_intr, cam_pose)
                vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

            # pcd_list = np.concatenate(pcd_list, axis=0).reshape(1,-1,3)
            # pydisco_utils.visualize_o3d(pcd_list)
            # ======================================================================================================== #

            # ======================================================================================================== #
            # Integrate
            # ======================================================================================================== #
            # Initialize voxel volume
            print("Initializing voxel volume...")
            print("bounds are: ", vol_bnds)
            # tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
            tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

            # Loop through RGB-D images and fuse them together
            t0_elapse = time.time()
            for i in range(n_imgs):
                print("Fusing frame %d/%d"%(i+1, n_imgs))

                # Read RGB-D image and camera pose
                # color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
                color_image = rgb_camXs[i]
                depth_im = depth_camXs[i, 0]
                depth_im[depth_im >= max_depth] = 0  # set invalid depth to 0 (specific to carla mc dataset)

                cam_pose = cam_poses[i]
                # Integrate observation into voxel volume (assume color aligned with depth)
                tsdf_vol.integrate(color_image, depth_im.cpu().numpy(), cam_intr, cam_pose, obs_weight=1.)

            fps = n_imgs / (time.time() - t0_elapse)
            print("Average FPS: {:.2f}".format(fps))

            # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving mesh to mesh.ply...")
            verts, faces, norms, colors = tsdf_vol.get_mesh() # somehow this is required for inside outside to work
            # fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

            # # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving point cloud to pc.ply...")
            # point_cloud = tsdf_vol.get_point_cloud()
            # fusion.pcwrite("pc.ply", point_cloud)

            # Get volume and sample points near surface
            sdf_1_points = pydisco_utils.get_freespace_points(torch.tensor(xyz_camXs).to(device), torch.tensor(cam_poses).to(device))
            inside_pts = tsdf_vol.visualize_inside_points()
            outside_pts = tsdf_vol.visualize_outside_points()
            # print("sdf1 vis")
            # pydisco_utils.visualize_o3d(sdf_1_points.reshape(1,-1,3))
            # print("Vis inside")
            # pydisco_utils.visualize_o3d(inside_pts.reshape((1,-1,3)))
            # print("Vis outside")
            # pydisco_utils.visualize_o3d(outside_pts.reshape((1,-1,3)))
            print(f"Saving {inside_pts[::3].shape[0]} inside points, {outside_pts[::3].shape} outside points, {sdf_1_points[::5].shape} sfd=1 points")
            datasave = {"inside":inside_pts[::3], "outside":outside_pts[::3], "sdf1":sdf_1_points[::5].cpu().numpy()}
            pickle.dump(datasave,open(f"{sdfpath}/tsdf_seq{seqnum}_{picklename}","wb"))