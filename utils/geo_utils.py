import torch

def get_relative_pose(cam_1, cam_2):
    '''
    cam_1: pose of camera 1 in world frame, in shape [4,4]
    cam_2: pose of camera 2 in world frame, in shape [t,4,4] for t cameras
    In math, we have:
        P^W = T^w_wToc1 @ P^c1,     T^w_wToc1 is the pose of camera in world frame
        P^W = T^w_wToc2 @ P^c2,     ...
    We want to get T^c1_c1Toc2, which is the relative pose of camera 2 to camera 1, we have
        P^c1 = T^c1_c1Toc2 @ P^c2
    => T^c1_c1Toc2 = T^w_wToc1.inv() @ T^w_wToc2
    If we denote camera pose as |R, t|, we have
                                |0, 1|
        T^c1_c1Toc2 = |R1, t1|-1 @ |R2, t2| = |R1.T @ R2, R1.T @ (t2 - t1)|
                      |0,  1 |     |0,  1 |   |0,         1               |
    '''
    assert len(cam_2.shape) == 3
    b = cam_2.shape[0]

    if len(cam_1.shape) == 2:
        cam_1 = cam_1.unsqueeze(0).repeat(b,1,1)
    
    R1 = cam_1[:,:3,:3]   # [t,3,3]
    t1 = cam_1[:,:3,3]    # [t,3]
    R2 = cam_2[:,:3,:3]   # [t,3,3]
    t2 = cam_2[:,:3,3]    # [t,3]

    R1_T = R1.permute(0,2,1)    # [t,3,3]
    R = torch.matmul(R1_T, R2)  # [t,3,3]
    t = torch.matmul(R1_T, (t2 - t1).view(b,3,1)).squeeze(-1)  # [t,3]

    pose = torch.zeros(b,4,4)      # T_c1_to_c2
    pose[:,:3,:3] = R
    pose[:,:3,3] = t
    pose[:,3,3] = 1.0

    return pose


def quat2mat_transform(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat
