from typing import Callable, Optional, Tuple, Union, List, OrderedDict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .transformer_util import EmbedMLP, TransformerBlock, get_1d_sincos_pos_embed_from_grid, \
                                get_multimodal_cond_pos_embed
from dataset.const import JOINT_BOUNDS, JOINT_NAMES
from .dino_vit import load_dino_model
from utils.geometries import rot6d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d
from utils.integral import HeatmapIntegralJoint, HeatmapIntegralPose
from utils.transforms import uvz2xyz_singlepoint
from utils.urdf_robot import URDFRobot


class MaskedSequential(nn.Sequential):
    def forward(self, x, mask=None):
        for module in self:
            if hasattr(module, 'forward') and callable(module.forward):
                x = module(x, mask) if mask is not None else module(x)
            else:
                x = module(x)  # 处理无mask参数的模块
        return x

ROBOT_DOF_DICT = {
    "panda": (8, 7),
    "kuka": (7, 8),
    "baxter": (15, 17)
}

def get_mask():
    mask = torch.zeros((333, 333))
    mask[:324, :324] = 1
    mask[324:, 324:] = 1
    return mask

class FullTransformerNetwork(nn.Module):

    def __init__(self, init_param_dict, args, **kwargs):
        super().__init__()
        self.backbone_name = args.backbone_name
        robot_type = init_param_dict["robot_type"]
        assert robot_type in ["panda", "kuka", "baxter"], f"Unknown robot type {init_param_dict['robot_type']}" 
        self.robot = URDFRobot(init_param_dict["robot_type"])
        DoF, nkpt = ROBOT_DOF_DICT[init_param_dict["robot_type"]]
        self.num_keypoints = nkpt    #& 7 in panda
        self.num_joints = DoF        #& 8 in panda 
        self.vision_encoder_name = args.vision_encoder_name
        
        #* Size of input data
        self.image_size = args.image_size
        self.height_dim = 72
        self.width_dim = 72 #TODO: Here to fit the size 
        self.depth_dim = args.depth_dim
        self.bbox_3d_shape = args.bbox_3d_shape
        self.reference_keypoint_id = args.reference_keypoint_id

        #* Size of latent in Transformer
        self.latent_dim = args.latent_dim 
        

        
        #* Layer configs
        #? 1. Vision part for image processing
        self.norm_type = "softmax"
        self.vision_backbone = load_dino_model(self.vision_encoder_name)
        print(f"Loading Dino Vision Encoder... as {self.vision_encoder_name}")
        self.vision_patch_size = self.vision_backbone.patch_size #! 14 in dino
        self.patch_num = (int(self.image_size) // self.vision_patch_size) ** 2
        #& need image_size to be divisible by vision_patch_size!
        self.vision_channel = nn.Linear(self.vision_backbone.embed_dim, self.latent_dim)
        
        #& process after transformer network
        self.vision_out_1 = nn.Linear(self.latent_dim, self.height_dim * self.width_dim)
        self.vision_out_2 = nn.Conv1d(in_channels=self.patch_num, out_channels=self.num_keypoints * self.depth_dim,  kernel_size=1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.vision_out_1.weight)  # N(0, 0.01)
        nn.init.xavier_uniform_(self.vision_out_2.weight)  # N(0, 0.01)
        self.integral_layer = HeatmapIntegralPose(backbone=self.backbone_name, num_joints=self.num_keypoints, depth_dim=self.depth_dim,
                                                height_dim=self.height_dim, width_dim=self.width_dim, norm_type=self.norm_type,
                                                image_size=self.image_size, bbox_3d_shape=self.bbox_3d_shape, rootid=self.reference_keypoint_id,
                                                fixroot=args.fix_root)
        #? 2. Pose, Rotation, Root_depth
        self.multi_kp = args.multi_kp
        self.kps_need_depth = args.kps_need_depth if self.multi_kp else [args.reference_keypoint_id]
        self.depth_num = len(self.kps_need_depth)
        self.rotation_dim = args.rotation_dim
        self.pose_embed = EmbedMLP(in_dim=1, hidden_dim=args.hidden_dim, out_dim=self.latent_dim, num_blocks=args.num_in_blocks)
        self.rotation_embed = EmbedMLP(in_dim=self.rotation_dim, hidden_dim=args.hidden_dim, out_dim=self.latent_dim, num_blocks=args.num_in_blocks)
        self.root_depth_embed = EmbedMLP(in_dim=1, hidden_dim=args.hidden_dim, out_dim=self.latent_dim, num_blocks=args.num_in_blocks)
        #& [B, 7, 3] -> [B, 7, latent_dim]                        ->   [B, 7, 3]
        #& [B, 8] -> [B, 8, 1] -> [B, 8, latent_dim]              ->   [B, 8, 1]  -> squeeze [B, 8]
        #& [B, rot_dim] -> [B, 1, rot_dim] ->[B, 1, latent_dim]   ->   [B, rot_dim]
        #& [B, 1] -> [B, 1, 1] -> [B, 1, latent_dim]              ->   [B, 1]

        #? 3. Output part:
        self.deconv_dim = [128, 128]
        self.vision_deconv_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=self.deconv_dim[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.deconv_dim[0]),
            nn.ConvTranspose2d(in_channels=self.deconv_dim[0], out_channels=self.deconv_dim[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.deconv_dim[1])
        )
        self.final_conv = nn.Conv2d(in_channels=self.deconv_dim[1], out_channels=self.num_keypoints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        self.pose_outhead = EmbedMLP(in_dim=self.latent_dim, hidden_dim=args.hidden_dim, out_dim=1, num_blocks=args.num_out_blocks)
        self.rotation_outhead = EmbedMLP(in_dim=self.latent_dim, hidden_dim=args.hidden_dim, out_dim=self.rotation_dim, num_blocks=args.num_out_blocks)
        self.root_depth_outhead = EmbedMLP(in_dim=self.latent_dim, hidden_dim=args.hidden_dim, out_dim=1, num_blocks=args.num_out_blocks)
        #TODO: Multi_kp depth pred need to predict multi depth!

        #? 4. Core part of LLM
        self.num_heads = args.num_heads
        self.transformer_blocks = args.transformer_blocks
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_joints + self.patch_num, self.latent_dim))
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.latent_dim,
            mm_cond_lens=OrderedDict([
                ('rotation', 1),
                ('joints', self.num_joints),
                ('vision', self.patch_num),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        self.transformer_part = MaskedSequential(
                *[TransformerBlock(embed_dim=self.latent_dim, 
                                   num_heads=self.num_heads, ) 
                        for _ in range(self.transformer_blocks)] 
        ) 
        

        #* init pose and rotation.
        init_param_pose = init_param_dict["pose_params"]
        init_param_cam = init_param_dict["cam_params"]
        init_param_from_mean = init_param_dict["init_pose_from_mean"]
        if init_param_from_mean:
            init_pose = torch.from_numpy(np.array([init_param_pose['mean'][robot_type][k] for k in JOINT_NAMES[robot_type]])).unsqueeze(0).float()
        else:
            init_pose = torch.from_numpy(np.array([init_param_pose['zero'][robot_type][k] for k in JOINT_NAMES[robot_type]])).unsqueeze(0).float()
        if self.rotation_dim == 6:
            init_rot = rotmat_to_rot6d(torch.from_numpy(np.array(init_param_cam[:3,:3])).unsqueeze(0)).float()
        elif self.rotation_dim == 4:
            init_rot = rotmat_to_quat(torch.from_numpy(np.array(init_param_cam[:3,:3])).unsqueeze(0)).float()

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_rot', init_rot)

    def forward(self, x_reg_input, x_root_input, k_value, K, init_pose=None, init_rot=None, test_fps=False, **kwargs):

        batch_size = x_reg_input.shape[0]
        x_reg_input = x_reg_input.to(torch.float)
        x_root_input = x_root_input.to(torch.float)
        
        #~ 暂时把 depth 换成 gt 的，看看是不是 depth 的误差带动了 vision integral 的误差
        gt_keypoints3d = kwargs.get("gt_3dkp", None)
        gt_rot = kwargs.get("gt_rot", None)
        gt_root_trans = gt_keypoints3d[:,self.reference_keypoint_id,:]
        gt_root_depth = gt_root_trans[:,2].unsqueeze(-1)
        # rotation_input = kwargs.get("rotation_input", None)
        # root_depth_input = kwargs.get("root_depth_input", None)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_rot is None:
            init_rot = self.init_rot.expand(batch_size, -1)
        root_trans_from_rootnet = torch.zeros((batch_size, 3)).float()
        init_depth = nn.Parameter(torch.ones(batch_size, 1).cuda() * 0.05) 
        #TODO Init depth needs checking 
        
        #* Vision forward
        if test_fps:
            t_start_vision = time.time()
        vision_feat = self.vision_backbone.forward_features(x_reg_input)["x_norm_patchtokens"]  #& [B, 324, 384]
        vision_output = self.vision_channel(vision_feat)                                        #& [B, 324, 512]
        
        if test_fps:
            t_end_vision = time.time()
            time_vision = t_end_vision - t_start_vision
            # print(f"Vision forward time: {t_end_vision - t_start_vision}")
        #* Pose, Rot, Root_depth forward
        # if joints_input is not None and rotation_input is not None and root_depth_input is not None:
        pred_pose_embed = self.pose_embed(init_pose.unsqueeze(-1))                              
        pred_rot_embed = self.rotation_embed(init_rot).unsqueeze(1)
        pred_depth_embed = self.root_depth_embed(init_depth).unsqueeze(1)
        
        #* Transformer forward
        pred_vis_point_pose_rot_depth_embed = torch.cat([pred_rot_embed, pred_pose_embed, vision_output], dim=1)
        pred_vis_point_pose_rot_depth_embed = pred_vis_point_pose_rot_depth_embed + self.x_pos_embed
        # print(f"PE is used!")
        # pred_vis_point_pose_rot_depth_embed = torch.cat([vision_output], dim=1)

        if test_fps:
            t_start_transformer = time.time()
        # mask = get_mask().cuda()
        pred_vis_point_pose_rot_depth_embed_output = self.transformer_part(pred_vis_point_pose_rot_depth_embed)
        
        if test_fps:
            t_end_transformer = time.time()
            time_transformer = t_end_transformer - t_start_transformer
            time_whole = t_end_transformer - t_start_vision
        #* Output part
        #TODO1: Modify processing output part.
        pred_vision_feat = pred_vis_point_pose_rot_depth_embed_output[:, 1+self.num_joints:, :]
        pred_vision_feat = pred_vision_feat.transpose(1, 2)
        pred_vision_feat = pred_vision_feat.reshape(batch_size, -1, (int(self.image_size) // self.vision_patch_size), (int(self.image_size) // self.vision_patch_size))
        pred_vision_feat = self.vision_deconv_head(pred_vision_feat)
        out = self.final_conv(pred_vision_feat)
        
        # pred_vision_feat = self.vision_out_1(pred_vision_feat)
        # out = self.vision_out_2(pred_vision_feat)

        pred_pose_feat = pred_vis_point_pose_rot_depth_embed_output[:, 1:1+self.num_joints, :]
        pred_pose = self.pose_outhead(pred_pose_feat)
        pred_pose = pred_pose.squeeze(-1)
        # pred_pose = torch.randn((batch_size, self.num_joints)).cuda()

        pred_rot_feat = pred_vis_point_pose_rot_depth_embed_output[:, 0, :]
        pred_rot = self.rotation_outhead(pred_rot_feat)
        # pred_rot = torch.randn((batch_size, self.rotation_dim)).cuda()
        # pred_rot = gt_rot
        pred_depth = gt_root_depth
        # pred_depth_feat = pred_vis_point_pose_rot_depth_embed_output[:, -1, :]
        # pred_depth = self.root_depth_outhead(pred_depth_feat)

        # root_trans_from_rootnet[:,2:3] = pred_depth #! 要先预测 depth 再预测 point
        root_trans_from_rootnet[:, 2:3] = pred_depth #~ 换成 gt 的 depth 看看估计准不准
        pred_uvd, pred_xyz_int = self.integral_layer(out, root_trans=root_trans_from_rootnet, K=K)
        pred_root_uv = (pred_uvd[:,self.reference_keypoint_id,:2] + 0.5) * self.image_size

        pred_trans = uvz2xyz_singlepoint(pred_root_uv, pred_depth, K)
        
        

        if self.reference_keypoint_id == 0:
            pred_xyz_fk = self.robot.get_keypoints(pred_pose, pred_rot, pred_trans)
        else:
            pred_xyz_fk = self.robot.get_keypoints_root(pred_pose, pred_rot, pred_trans,root=self.reference_keypoint_id)
 
        if test_fps:
            return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk, (time_vision, time_transformer, time_whole)
        else:
            if self.multi_kp:
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_depths, pred_uvd, pred_xyz_int, pred_xyz_fk
            else:
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk




        

        

        # pred_uvd, pred_xyz_int = self.integral_layer(vision_output, root_trans=root_trans_from_rootnet, K=K)
        



def get_transformerModel(init_params_dict, args, **kwargs):
    model = FullTransformerNetwork(init_params_dict, args, **kwargs)
    return model


if __name__ == "__main__":
    init_param_dict = {
        "robot_type" : "panda",
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    model_name = "vits"
    model = load_dino_model(model_name)
    print(model)