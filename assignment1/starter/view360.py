"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import imageio
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def problem1(cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None):
    # problem 1.1 360-degree Renders
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    # Prepare the camera:
    azims = torch.linspace(0, 180, steps=180)
    cameras = []
    for azim in azims:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        cameras.append(camera)
    
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)


    imgs = []
    for camera in cameras:
        rend = renderer(mesh, cameras=camera, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3] 
        imgs.append(rend)

    return imgs
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/my_gif.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    images = problem1(cow_path=args.cow_path, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=15)
