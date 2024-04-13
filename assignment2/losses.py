import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src: torch.Tensor, voxel_tgt: torch.Tensor):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# https://arxiv.org/pdf/1603.08637.pdf page 5:
	loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(voxel_src), voxel_tgt) 
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src: torch.Tensor, point_cloud_tgt: torch.Tensor):
	# point_cloud_src, point_cloud_src: b x n_points x 3
	p1top2 = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, norm=2, K=1).dists[..., 0]
	p2top1 = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, norm=2, K=1).dists[..., 0]
	loss_chamfer = torch.mean(p1top2) + torch.mean(p2top1)
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src: torch.Tensor):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian