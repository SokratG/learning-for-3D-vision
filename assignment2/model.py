from torchvision import models as torchvision_models
from torchvision import transforms
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (64, 2, 2, 2)), 

                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), # [-1, 32, 4, 4, 4]
                nn.BatchNorm3d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1), # [-1, 16, 8, 8, 8]
                nn.BatchNorm3d(16),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1), # [-1, 8, 16, 16, 16]
                nn.BatchNorm3d(8),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1), # [-1, 4, 32, 32, 32]
                nn.BatchNorm3d(4),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(4, 1, kernel_size=1), # [-1, 1, 32, 32, 32]
                # nn.ReLU()
            )

        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, self.n_point * 3),
                nn.Tanh(),
                nn.Unflatten(1,(self.n_point, 3))
            )         
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            out_size = mesh_pred.verts_packed().shape[0] * 3
            self.vert_size = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, out_size),
                nn.Tanh(),
                nn.Unflatten(1, (self.vert_size, 3))
            )             

    def forward(self, images, args):

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)   
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)           
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

