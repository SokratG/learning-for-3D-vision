import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_mesh_renderer, get_device, load_cow_mesh

def retexture(
    cow_path="data/cow.obj",
    image_size=256,
    device=None,
):
    renderer = get_mesh_renderer(image_size=image_size)
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    z = vertices[0, :, 2]
    z_min, z_max = z.min(), z.max()
    alpha = ((z - z_min) / (z_max - z_min)).unsqueeze(1)
    color1, color2 = torch.tensor([0., 0, 1]), torch.tensor([1., 0, 0])

    color = alpha * color2 + (1 - alpha) * color1
    textures = torch.ones_like(vertices) * color

    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([0, torch.pi/2, 0]), "XYZ"
    )
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices @ relative_rotation,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_retexture.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    device = get_device()
    
    image = retexture(cow_path=args.cow_path, image_size=args.image_size, device=device)
    plt.imsave(args.output_path, image)