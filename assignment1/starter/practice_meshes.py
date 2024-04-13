import argparse

import imageio
import pytorch3d
import torch

from starter.utils import get_mesh_renderer, get_device

def tetrahedron(
    image_size=256,
    device=None,
):
    color = [0.7, 0.7, 1]
    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([torch.pi/2, 0, 0]), "XYZ"
    )
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, -1.0, 2.0], [-1.0, 1.0, 2.0]]).unsqueeze(0)
    faces = torch.tensor([[1, 3, 2], [2, 0, 3], [3, 0, 1], [0, 1, 2]], dtype=torch.int32).unsqueeze(0)
    textures = torch.ones_like(verts)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=verts @ relative_rotation,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    mesh = mesh.to(device)

    azims = torch.linspace(0, 180, steps=180)
    cameras = []
    for azim in azims:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4.5, elev=25, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        cameras.append(camera)

    lights = pytorch3d.renderer.PointLights(location=[[-0.25, 0.25, -1.5]], device=device)

    imgs = []
    renderer = get_mesh_renderer(image_size=image_size)
    for camera in cameras:
        rend = renderer(mesh, cameras=camera, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3] 
        imgs.append(rend)
    
    return imgs, 'tetrahedron'


def cube(
    image_size=256,
    device=None,
):
    color = [0.7, 0.7, 1]
    verts = torch.tensor([[-1.0, 1, -1],
        [1,   1, -1],
        [1,  -1, -1],
        [-1, -1, -1],
        [-1,  1,  1],
        [1,   1,  1],
        [1,  -1,  1],
        [-1, -1,  1],
    ]).unsqueeze(0)

    faces = torch.tensor([[0, 1, 2],
        [0, 2, 3],
        [2, 1, 5],
        [2, 5, 6],
        [3, 2, 6],
        [3, 6, 7],
        [0, 3, 7],
        [0, 7, 4],
        [1, 0, 4],
        [1, 4, 5],
        [6, 5, 4],
        [6, 4, 7]
    ], dtype=torch.int32).unsqueeze(0)
    textures = torch.ones_like(verts)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    mesh = mesh.to(device)

    azims = torch.linspace(0, 180, steps=180)
    cameras = []
    for azim in azims:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4.5, elev=25, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        cameras.append(camera)

    lights = pytorch3d.renderer.PointLights(location=[[-0.25, 0.25, -1.5]], device=device)

    imgs = []
    renderer = get_mesh_renderer(image_size=image_size)
    for camera in cameras:
        rend = renderer(mesh, cameras=camera, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3] 
        imgs.append(rend)

    return imgs, 'cube'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default='tetrahedron')
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    device = get_device()
    if args.mesh == 'cube':
        images, mesh_name = cube(device=device)
    else:
        images, mesh_name = tetrahedron(device=device)
    output_path = f'images/{mesh_name}.gif'
    imageio.mimsave(output_path, images, fps=15)
