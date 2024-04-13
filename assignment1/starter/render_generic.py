"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_rgbd(
    image_size=256,
    path="data/rgbd_data.pkl",
    background_color=(1, 1, 1),
    num_views=18,
    output_path='images/',
    device=None
):
    if device is None:
        device = get_device()
    data = load_rgbd_data(path=path)
    rgb1, depth1, mask1 = torch.from_numpy(data['rgb1']), torch.from_numpy(data['depth1']), torch.from_numpy(data['mask1'])
    rgb2, depth2, mask2 = torch.from_numpy(data['rgb2']), torch.from_numpy(data['depth2']), torch.from_numpy(data['mask2'])
    camera1, camera2 = data['cameras1'], data['cameras2']

    points1, color1 = unproject_depth_image(rgb1, mask1, depth1, camera1)
    points2, color2 = unproject_depth_image(rgb2, mask2, depth2, camera2)
    points3, color3 = torch.cat([points1, points2], dim=0), torch.cat([color1, color2], dim=0)
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud1 = pytorch3d.structures.Pointclouds(points=points1.to(device).unsqueeze(0), features=color1.to(device).unsqueeze(0))
    point_cloud2 = pytorch3d.structures.Pointclouds(points=points2.to(device).unsqueeze(0), features=color2.to(device).unsqueeze(0))
    point_cloud3 = pytorch3d.structures.Pointclouds(points=points3.to(device).unsqueeze(0), features=color3.to(device).unsqueeze(0))

    pcs = [point_cloud1, point_cloud2, point_cloud3]
    azims = np.linspace(-180, 180, num_views, endpoint=False)
    R_relative = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, torch.pi]), "XYZ")
    for idx, pc in enumerate(pcs):
        imgs = []
        for azim in azims:
            R, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=azim)
            R = R_relative @ R 
            camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(pc, cameras=camera).cpu().numpy()[0, ..., :3]  
            imgs.append(rend)
        filename = output_path + f'point_cloud{idx+1}.gif'
        imageio.mimsave(filename, imgs, fps=20)



def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_torus(image_size=256, num_samples=200, device=None, R=1.8, r=1.1,
                 num_views=24, output_path='images/torus.gif'):
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * torch.pi, num_samples)
    theta = torch.linspace(0, 2 * torch.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_pc = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)
    renderer = get_points_renderer(image_size=image_size, device=device)

    T = [[0, 0, 5.5]]
    azims = np.linspace(-180, 180, num_views, endpoint=False)
    imgs = []
    for azim in azims:
            Rot, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=azim)
            camera = pytorch3d.renderer.FoVPerspectiveCameras(R=Rot, T=T, device=device)
            rend = renderer(torus_pc, cameras=camera).cpu().numpy()[0, ..., :3]  
            imgs.append(rend)
    imageio.mimsave(output_path, imgs, fps=18)


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_torus_mesh(image_size=256, voxel_size=64, device=None, R=1.8, r=1.1,
                      num_views=24, output_path='images/implicit-torus.gif'):
    if device is None:
        device = get_device()
    min_value = -3.1
    max_value = 3.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    T = [[0, 0, 5.5]]
    azims = np.linspace(-180, 180, num_views, endpoint=False)
    imgs = []
    for azim in azims:
        Rot, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=0, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(R=Rot, T=T, device=device)
        rend = renderer(mesh, cameras=camera, lights=lights)[0, ..., :3].cpu().numpy().clip(0, 1)
        imgs.append(rend)
    imageio.mimsave(output_path, imgs, fps=18)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric-sphere", "parametric-torus", "implicit-sphere", "implicit-torus", "rgbd"],
    )
    
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "rgbd":
        render_rgbd(image_size=args.image_size)
    elif args.render == "parametric-torus":
        render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit-torus":
        render_torus_mesh(image_size=args.image_size)
    else:
        if args.render == "point_cloud":
            image = render_bridge(image_size=args.image_size)
            filename = 'images/bridge.jpg'
        elif args.render == "parametric-sphere":
            image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
            filename = 'images/parametric-sphere.jpg'
        elif args.render == "implicit-sphere":
            image = render_sphere_mesh(image_size=args.image_size)
            filename = 'images/implicit-sphere.jpg'
        else:
            raise Exception("Did not understand {}".format(args.render))
        plt.imsave(filename, image)

