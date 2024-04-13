"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)


    R_r1 = torch.tensor(R_relative).float()
    T_r1 = torch.tensor(T_relative).float()
    R_r2 = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([0, 0, torch.pi/2]), "XYZ"
    )
    T_r2 = torch.tensor([0, 0, 0]).float()

    R_r3 = torch.tensor(R_relative).float()
    T_r3 = torch.tensor([0, 0, 2]).float()

    R_r4 = torch.tensor(R_relative).float()
    T_r4 = torch.tensor([0.48, -0.45, -0.1]).float()

    R_r5 = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([0, -torch.pi/2, 0]), "XYZ"
    )
    T_r5 = torch.tensor([3.0, 0, 3.0]).float()

    Rs = [R_r1, R_r2, R_r3, R_r4, R_r5]
    Ts = [T_r1, T_r2, T_r3, T_r4, T_r5]
    imgs = []
    for r, t in zip(Rs, Ts):
        R = r @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = r @ torch.tensor([0.0, 0, 3]) + t
        # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
        # we need to add R.t() to compensate that.
        renderer = get_mesh_renderer(image_size=image_size)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
        rend = renderer(meshes, cameras=cameras, lights=lights)
        imgs.append(rend[0, ..., :3].cpu().numpy())
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/transform_cow.jpg")
    args = parser.parse_args()
    imgs = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    for idx, img in enumerate(imgs):
        outpath = f'images/transform_cow{idx + 1}.jpg'
        plt.imsave(outpath, img)
