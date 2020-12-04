
import torch 
import socket 
import ipdb 
st = ipdb.set_trace
hostname = socket.gethostname()
if 'Shamit' in hostname:
    import open3d as o3d

def sub2ind(height, width, y, x):
    return y*width + x

def create_depth_image_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    #print(hashit(xy),hashit(z))
    xy = torch.round(xy).long()
    #print(hashit(xy))
    depth = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    #print(hashit(depth))
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)
    #print(hashit(valid))
    # st()

    # gather these up
    xy = xy[valid]
    z = z[valid]

    #print(hashit(xy),hashit(z))

    inds = sub2ind(H, W, xy[:,1], xy[:,0]).long()

    #print(hashit(inds))
    depth[inds] = z
    #print(hashit(depth))
    # st()
    valid = (depth > 0.0).float()
    # print(torch.sum(depth))
    depth[torch.where(depth == 0.0)] = 100.0
    # print(torch.sum(depth))
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS=1e-6
    x = (x*fx)/(z+EPS)+x0
    y = (y*fy)/(z+EPS)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy
    
def create_depth_image(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    for b in range(B):
        depth[b], valid[b] = create_depth_image_single(xy[b], z[b], H, W)
    return depth, valid


def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def visualize_o3d(xyz_camX):
    pcd = make_pcd(xyz_camX[0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd


def get_4x4(arr):
    out = torch.eye(4)
    out[:3, :3] = torch.tensor(arr)
    return out.unsqueeze(0)

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = meshgrid2D(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz


def meshgrid2D(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X
    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def normalize_grid2D(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def get_freespace_points(xyz_camXs, origin_T_camXs):
    # S, N, _ = xyz_camXs.shape
    # xyz_camXs = xyz_camXs.reshape(S,N,3,1)
    alpha = torch.linspace(0.1, 0.9, 10).reshape(1,1,-1).to(torch.device('cuda'))
    # samples = xyz_camXs*alpha
    samples_list = []
    for i, xyz_camX in enumerate(xyz_camXs):
        x, y, z = torch.unbind(xyz_camX, dim=-1)
        valid = torch.where(z<100)[0]
        x = x[valid]
        y = y[valid]
        z = z[valid]
        xyz_camX = torch.stack([x,y,z], dim=-1)
        N, _ = xyz_camX.shape
        xyz_camX = xyz_camX.reshape(N,3,1)
        samples = xyz_camX*alpha
        samples = samples.permute(0,2,1)
        samples = samples.reshape(-1, 3)
        samples = apply_4x4(origin_T_camXs[i:i+1].float(), samples.unsqueeze(0))[0]
        samples_list.append(samples)

    samples_list = torch.cat(samples_list, dim=0)
    return samples_list