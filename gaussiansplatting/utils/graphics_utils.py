#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)



def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0,1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Optional, Tuple
import torch
import math
import numpy as np
from typing import NamedTuple
from kornia.core import Tensor, concatenate
from kornia.core import stack, zeros_like

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = zeros_like(x0)
    cross_product_matrix_flat = stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)

@torch.no_grad()
def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
        raise AssertionError(P1.shape)
    if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
        raise AssertionError(P2.shape)
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat(
        [
            X1Y1.det().reshape(-1, 1),
            X2Y1.det().reshape(-1, 1),
            X3Y1.det().reshape(-1, 1),
            X1Y2.det().reshape(-1, 1),
            X2Y2.det().reshape(-1, 1),
            X3Y2.det().reshape(-1, 1),
            X1Y3.det().reshape(-1, 1),
            X2Y3.det().reshape(-1, 1),
            X3Y3.det().reshape(-1, 1),
        ],
        dim=1,
    )

    return F_vec.view(*P1.shape[:-2], 3, 3)

def essential_from_Rt(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    r"""Get the Essential matrix from Camera motion (Rs and ts).

    Reference: Hartley/Zisserman 9.6 pag 257 (formula 9.12)

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        The Essential matrix with the shape :math:`(*, 3, 3)`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # first compute the camera relative motion
    R, t = relative_camera_motion(R1, t1, R2, t2)

    # get the cross product from relative translation vector
    Tx = cross_product_matrix(t[..., 0])

    return Tx @ R

def relative_camera_motion(
    R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the relative camera motion between two cameras.

    Given the motion parameters of two cameras, computes the motion parameters of the second
    one assuming the first one to be at the origin. If :math:`T1` and :math:`T2` are the camera motions,
    the computed relative motion is :math:`T = T_{2}T^{-1}_{1}`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        A tuple with the relative rotation matrix and
        translation vector with the shape of :math:`[(*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return R, t

def motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The possible solutions are :math:`[R1,t], [R1,-t], [R2,t], [R2,-t]`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = stack([R1, R1, R2, R2], dim=-3)
    Ts = stack([t, -t, t, -t], dim=-3)

    return Rs, Ts

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0,1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W):
    NDC_2_pixel = torch.tensor([[current_W / 2, 0, current_W / 2], [0, current_H / 2, current_H / 2], [0, 0, 1]]).cuda()
    # NDC_2_pixel_inversed = torch.inverse(NDC_2_pixel)
    NDC_2_pixel = NDC_2_pixel.float()
    cam_1_tranformation = cam1.full_proj_transform[:, [0,1,3]].T.float()
    cam_2_tranformation = cam2.full_proj_transform[:, [0,1,3]].T.float()
    cam_1_pixel = NDC_2_pixel@cam_1_tranformation
    cam_2_pixel = NDC_2_pixel@cam_2_tranformation

    # print(NDC_2_pixel.dtype, cam_1_tranformation.dtype, cam_2_tranformation.dtype, cam_1_pixel.dtype, cam_2_pixel.dtype)

    cam_1_pixel = cam_1_pixel.float()
    cam_2_pixel = cam_2_pixel.float()
    # print("cam_1", cam_1_pixel.dtype, cam_1_pixel.shape)
    # print("cam_2", cam_2_pixel.dtype, cam_2_pixel.shape)
    # print(NDC_2_pixel@cam_1_tranformation, NDC_2_pixel@cam_2_tranformation)
    return fundamental_from_projections(cam_1_pixel, cam_2_pixel)
    # return fundamental_from_projections(NDC_2_pixel.float()@cam1.full_proj_transform[:, [0,1,3]].T.float(), NDC_2_pixel.float()@cam2.full_proj_transform[:, [0,1,3]].T.float()).half()

# def get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W):
#     RT1 = torch.zeros((3, 4)).cuda()
#     RT2 = torch.zeros((3, 4)).cuda()

#     W2C1 = torch.from_numpy(getWorld2View2(cam1.R, cam1.T)).cuda()
#     W2C2 = torch.from_numpy(getWorld2View2(cam2.R, cam2.T)).cuda()

#     RT1[:, :3] = W2C1[:3, :3]
#     RT1[:, 3] = W2C1[:3, 3]
#     RT2[:, :3] = W2C2[:3, :3]
#     RT2[:, 3] = W2C2[:3, 3]

#     K_1 = torch.zeros((3, 3)).cuda()
#     focal_x_1 = fov2focal(cam1.FoVx, current_W)
#     focal_y_1 = fov2focal(cam1.FoVy, current_H)
#     p_x_1 = current_W / 2
#     p_y_1 = current_H / 2

#     K_1[0, 0] = focal_x_1
#     K_1[1, 1] = focal_y_1
#     K_1[0, 2] = p_x_1
#     K_1[1, 2] = p_y_1
#     K_1[2, 2] = 1

#     K_2 = torch.zeros((3, 3)).cuda()
#     focal_x_2 = fov2focal(cam2.FoVx, current_W)
#     focal_y_2 = fov2focal(cam2.FoVy, current_H)
#     p_x_2 = current_W / 2
#     p_y_2 = current_H / 2

#     K_2[0, 0] = focal_x_2 
#     K_2[1, 1] = focal_y_2 
#     K_2[0, 2] = p_x_2
#     K_2[1, 2] = p_y_2
#     K_2[2, 2] = 1

#     return fundamental_from_projections(K_1 @ RT1, K_2 @ RT2)
    
@torch.no_grad()
def compute_differece_matrix(_3d_positions, cam1, cam2):
    # 3d_positions: (N, 4)
    # cam1: (4, 4)
    # cam2: (3, 4)
    # return: (N, 2)
    # import pdb; pdb.set_trace()
    W2C1 = torch.from_numpy(getWorld2View2(cam1.R, cam1.T)).cuda()
    W2C2 = torch.from_numpy(getWorld2View2(cam2.R, cam2.T)).cuda()
    W2C1_Rotation = W2C1[:3, :3]
    W2C2_Rotation = W2C2[:3, :3]
    W2C1_Translation = W2C1[:3, 3]
    W2C2_Translation = W2C2[:3, 3]
    _3d_poisitons_1 =  _3d_positions @ W2C1_Rotation.T + W2C1_Translation
    _3d_poisitons_2 = _3d_positions @ W2C2_Rotation.T + W2C2_Translation
    K_1 = torch.zeros((3, 3)).cuda()
    focal_x_1 = fov2focal(cam1.FoVx, cam1.image_width)
    focal_y_1 = fov2focal(cam1.FoVy, cam1.image_height)
    p_x_1 = cam1.image_width / 2
    p_y_1 = cam1.image_height / 2

    K_1[0, 0] = focal_x_1 / cam1.image_width
    K_1[1, 1] = focal_y_1 / cam1.image_height
    K_1[0, 2] = p_x_1 / cam1.image_width
    K_1[1, 2] = p_y_1 / cam1.image_height
    K_1[2, 2] = 1

    K_2 = torch.zeros((3, 3)).cuda()
    focal_x_2 = fov2focal(cam2.FoVx, cam2.image_width)
    focal_y_2 = fov2focal(cam2.FoVy, cam2.image_height)
    p_x_2 = cam2.image_width / 2
    p_y_2 = cam2.image_height / 2

    K_2[0, 0] = focal_x_2 / cam2.image_width
    K_2[1, 1] = focal_y_2 / cam2.image_height
    K_2[0, 2] = p_x_2 / cam2.image_width
    K_2[1, 2] = p_y_2  / cam2.image_height
    K_2[2, 2] = 1
    
    # import pdb; pdb.set_trace()

    _3d_poisitons_1 = _3d_poisitons_1 @ K_1.T
    _3d_poisitons_2 = _3d_poisitons_2 @ K_2.T

    _3d_poisitons_1 = _3d_poisitons_1[:, :2] / _3d_poisitons_1[:, [2]]
    _3d_poisitons_2 = _3d_poisitons_2[:, :2] / _3d_poisitons_2[:, [2]]
    diff = _3d_poisitons_1 - _3d_poisitons_2

    empty_black = torch.zeros(size=(diff.shape[0], 1)).cuda()
    return torch.cat([diff, empty_black], dim=1)

@torch.no_grad()
def RT_to_projection_matrix(cam1, cam2):

    W2C1 = torch.from_numpy(getWorld2View2(cam1.R, cam1.T)).cuda()
    W2C2 = torch.from_numpy(getWorld2View2(cam2.R, cam2.T)).cuda()

    K_1 = torch.zeros((3, 3)).cuda()
    focal_x_1 = fov2focal(cam1.FoVx, cam1.image_width)
    focal_y_1 = fov2focal(cam1.FoVy, cam1.image_height)
    p_x_1 = cam1.image_width / 2
    p_y_1 = cam1.image_height / 2

    K_1[0, 0] = focal_x_1
    K_1[1, 1] = focal_y_1 
    K_1[0, 2] = p_x_1
    K_1[1, 2] = p_y_1
    K_1[2, 2] = 1

    K_2 = torch.zeros((3, 3)).cuda()
    focal_x_2 = fov2focal(cam2.FoVx, cam2.image_width)
    focal_y_2 = fov2focal(cam2.FoVy, cam2.image_height)
    p_x_2 = cam2.image_width / 2
    p_y_2 = cam2.image_height / 2

    K_2[0, 0] = focal_x_2
    K_2[1, 1] = focal_y_2
    K_2[0, 2] = p_x_2
    K_2[1, 2] = p_y_2
    K_2[2, 2] = 1

    P1 = K_1 @ W2C1[:3, :]
    P2 = K_2 @ W2C2[:3, :]
    
    return P1, P2
