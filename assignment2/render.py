import numpy as np
import pytorch3d
import torch
import imageio


from utils import get_device, get_mesh_renderer, get_points_renderer

def render_mesh(voxel_mesh: pytorch3d.structures.Meshes, image_size: int = 256,
                device: torch.device = None, num_views: int = 24,
                output_path: str = 'images/voxels.gif'):
    renderer = get_mesh_renderer(image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    azims = np.linspace(-180, 180, num_views, endpoint=False)
    imgs = []
    for azim in azims:
        Rot, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0.5, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(R=Rot, T=T, device=device)
        rend = renderer(voxel_mesh, cameras=camera, lights=lights).cpu().numpy()[0, ..., :3]  
        imgs.append(rend)
    imageio.mimsave(output_path, imgs, fps=18)


def render_point_cloud(pc: pytorch3d.structures.Pointclouds, image_size: int = 256,
                       device: torch.device = None, num_views: int = 24,
                       output_path: str = 'images/voxels.gif'):
    renderer = get_points_renderer(image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    azims = np.linspace(-180, 180, num_views, endpoint=False)
    imgs = []
    for azim in azims:
        Rot, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0.5, azim=azim)
        camera = pytorch3d.renderer.FoVPerspectiveCameras(R=Rot, T=T, device=device)
        rend = renderer(pc, cameras=camera, lights=lights).cpu().numpy()[0, ..., :3]  
        imgs.append(rend)
    imageio.mimsave(output_path, imgs, fps=18)


def render_voxels(voxels: torch.Tensor, output_path: str, 
                  color: torch.Tensor = torch.tensor([0.2, 0.2, 1.0])):
    # https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.cubify
    device = get_device()
    voxel_mesh = pytorch3d.ops.cubify(voxels, thresh=0.5, device=device)
    tex_size = voxel_mesh.verts_list()[0].unsqueeze(0).shape
    textures = torch.ones(tex_size).to(device) * color.to(device)
    voxel_mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_mesh(voxel_mesh, image_size=256, device=device, output_path=output_path)


def render_pc(pts: torch.Tensor, output_path: str, 
              color: torch.Tensor = torch.tensor([0.33, 0.54, 0.95])):
    device = get_device()
    tex_size = pts.squeeze(0).shape
    colors = torch.ones(tex_size).to(device) * color.to(device)
    pc = pytorch3d.structures.Pointclouds(points=[pts.squeeze(0).detach()], features=[colors]).to(device)
    render_point_cloud(pc, image_size=256, device=device, output_path=output_path)


def render_pmesh(mesh: pytorch3d.structures.Meshes, output_path: str, 
                 color: torch.Tensor = torch.tensor([0.46, 0.77, 0.62])):
    device = get_device()
    tex_size = mesh.verts_list()[0].unsqueeze(0).shape
    textures = torch.ones(tex_size).to(device) * color.to(device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    render_mesh(mesh.detach(), image_size=256, device=device, output_path=output_path)