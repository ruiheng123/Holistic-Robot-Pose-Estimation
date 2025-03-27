import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import time
import numpy as np
import torch
import torch.nn as nn
from dataset.const import JOINT_BOUNDS, JOINT_NAMES
from .backbones.HRnet import get_hrnet
from .backbones.Resnet import get_resnet
from utils.geometries import rot6d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d
from utils.integral import HeatmapIntegralJoint, HeatmapIntegralPose
from utils.transforms import uvz2xyz_singlepoint
from utils.urdf_robot import URDFRobot

class JointNetPart(nn.Module):
    def __init__(self, reg_joint_map, feature_channel, #~ 这部分为必须
                 backbone_name, joint_conv_dim, p_dropout, n_iter, norm_type, 
                 robot_type, DoF) -> None:
        super(JointNetPart, self).__init__()
        self.reg_joint_map = reg_joint_map
        self.feature_channel = feature_channel
        self.n_iter = n_iter

        if self.reg_joint_map:
            self.joint_conv_dim = joint_conv_dim
            assert self.joint_conv_dim != None, "self.joint_conv_dim is None!"
            self.joint_conv_layers = self._make_joint_conv_layer()
            self.joint_final_layer = nn.Conv2d(self.joint_conv_dim[2], DoF, kernel_size=1, stride=1, padding=0)
            joint_bounds = torch.tensor(JOINT_BOUNDS[robot_type]).float()
            self.joint_integral_layer = HeatmapIntegralJoint(backbone=backbone_name, dof=DoF, norm_type=norm_type, joint_bounds=joint_bounds)
        else:
            self.fc_pose_1 = nn.Linear(self.feature_channel + DoF, 1024)
            self.fc_pose_2 = nn.Linear(1024, 1024)
            self.decpose = nn.Linear(1024, DoF)
            self.drop1 = nn.Dropout(p=p_dropout)
            self.drop2 = nn.Dropout(p=p_dropout)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)

    def _make_joint_conv_layer(self):
        joint_conv_layers = []
        conv1 = nn.Conv2d(self.feature_channel, self.joint_conv_dim[0], kernel_size=3, stride=1, padding=1)
        bn1 = nn.BatchNorm2d(self.joint_conv_dim[0])
        conv2 = nn.Conv2d(self.joint_conv_dim[0], self.joint_conv_dim[1], kernel_size=3, stride=1, padding=1)
        bn2 = nn.BatchNorm2d(self.joint_conv_dim[1])
        conv3 = nn.Conv2d(self.joint_conv_dim[1], self.joint_conv_dim[2], kernel_size=3, stride=1, padding=1)
        bn3 = nn.BatchNorm2d(self.joint_conv_dim[2])
        
        joint_conv_layers.append(conv1)
        joint_conv_layers.append(bn1)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        joint_conv_layers.append(conv2)
        joint_conv_layers.append(bn2)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        joint_conv_layers.append(conv3)
        joint_conv_layers.append(bn3)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*joint_conv_layers)
    def forward(self, xf, x_out, init_pose):
        pred_pose = init_pose
        xf = xf.view(xf.size(0), -1)
        if self.reg_joint_map:
            joint_out = self.joint_conv_layers(x_out)
            joint_out = self.joint_final_layer(joint_out)
            pred_pose = self.joint_integral_layer(joint_out)
        else:
            for _ in range(self.n_iter):
                xc = torch.cat([xf, pred_pose],1).to(torch.float)
                xc = self.fc_pose_1(xc)
                xc = self.drop1(xc)
                # if i < int(self.n_iter / 2):
                #     skiplist[i] = xc
                # elif i == int(self.n_iter / 2) and self.n_iter % 2 != 0:
                #     pass
                # else:
                #     xplus = skiplist[i-int((self.n_iter+1) / 2)]
                #     xc += xplus
                xc = self.fc_pose_2(xc)
                xc = self.drop2(xc)
                pred_pose = self.decpose(xc) + pred_pose

        return pred_pose


class RotationNetPart(nn.Module):

    def __init__(self, direct_reg_rot, rot_iterative_matmul, 
                 feature_channel, rotation_dim,
                 p_dropout, n_iter) -> None:
        super(RotationNetPart, self).__init__()
        self.direct_reg_rot = direct_reg_rot
        self.rot_iterative_matmul = rot_iterative_matmul
        self.feature_channel = feature_channel
        self.rotation_dim = rotation_dim
        self.n_iter = n_iter
        if self.direct_reg_rot:
            self.fc_rot_1 = nn.Linear(self.feature_channel, 1024)
            self.fc_rot_2 = nn.Linear(1024, 1024)
            self.fc_rot_3 = nn.Linear(1024, 1024)
            self.fc_rot_4 = nn.Linear(1024, 1024)
            self.fc_rot_5 = nn.Linear(1024, 1024)
            self.fc_rot_6 = nn.Linear(1024, 1024)
            self.decrot = nn.Linear(1024, 6)
            self.drop_rot = nn.Dropout(p=0.2)
            nn.init.xavier_uniform_(self.decrot.weight, gain=0.01)

        else:
            self.fc_rot_1 = nn.Linear(self.feature_channel + self.rotation_dim, 1024)
            self.fc_rot_2 = nn.Linear(1024, 1024) 
            self.decrot = nn.Linear(1024, self.rotation_dim) # rotation representation (quaternion / 6D / 9D)
            self.drop1 = nn.Dropout(p=p_dropout)
            self.drop2 = nn.Dropout(p=p_dropout)
            nn.init.xavier_uniform_(self.decrot.weight, gain=0.01)
    def forward(self, xf, init_rot):
        pred_rot = init_rot
        xf = xf.view(xf.size(0), -1)
        if self.direct_reg_rot:
            # xc = self.relu_rot(self.bn_rot_1(self.fc_rot_1(xf)))
            # xc = self.relu_rot(self.bn_rot_2(self.fc_rot_2(xc)))
            # xc = self.relu_rot(self.bn_rot_3(self.fc_rot_3(xc)))
            # xc = self.relu_rot(self.bn_rot_4(self.fc_rot_4(xc)))
            # xc = self.relu_rot(self.bn_rot_5(self.fc_rot_5(xc) + self.fc_rot_05(xf)))
            xc1 = self.fc_rot_1(xf)
            xc2 = self.fc_rot_2(xc1)
            xc3 = self.fc_rot_3(xc2)
            xc4 = self.fc_rot_4(xc3)
            xc5 = self.fc_rot_5(xc4)
            xc6 = self.fc_rot_6(xc5)
            xc6 = xc6 + xc1
            pred_rot = self.decrot(xc6)
        else:
            if self.rot_iterative_matmul: #? 或者是additive 或这是multiply
                assert(self.rotation_dim == 6), f'Your self.rotation dim is {self.rotation_dim}, not 6!'
                for _ in range(self.n_iter):
                    xc = torch.cat([xf, pred_rot],1).to(torch.float)
                    xc = self.fc_rot_1(xc)
                    xc = self.drop1(xc)
                    xc = self.fc_rot_2(xc)
                    xc = self.drop2(xc)
                    pred_rot = rotmat_to_rot6d(rot6d_to_rotmat(self.decrot(xc)) @ rot6d_to_rotmat(pred_rot))
            else:
                for _ in range(self.n_iter):
                    xc = torch.cat([xf, pred_rot],1).to(torch.float)
                    xc = self.fc_rot_1(xc)
                    xc = self.drop1(xc)
                    xc = self.fc_rot_2(xc)
                    xc = self.drop2(xc)
                    pred_rot = self.decrot(xc) + pred_rot
            
        return pred_rot

class DepthNetPart(nn.Module):
    def __init__(self, multi_kp, kps_need_depth, add_fc, inplanes, depth_num):
        super().__init__()
        self.multi_kp = multi_kp
        self.kps_need_depth = kps_need_depth
        self.add_fc = add_fc
        self.depth_num = depth_num

        if self.add_fc:
            self.depth_dropout = nn.Dropout(p=0.2)
            self.depth_fc_d1 = nn.Linear(inplanes, 1024)
            self.depth_fc_d2 = nn.Linear(1024, 512)
            self.depth_bn = nn.BatchNorm1d(512)
            self.depth_lrelu = nn.LeakyReLU()
            self.depth_fc_u2 = nn.Linear(512, 1024)
            self.depth_fc_u1 = nn.Linear(1024, inplanes)
        self.depth_layer = nn.Conv2d(in_channels=inplanes, out_channels=self.depth_num, kernel_size=1, stride=1, padding=0)


    def forward(self, img_feat, k_value, reference_keypoint_id):
        if self.add_fc:
            img_feat1 = self.depth_fc_d1(img_feat)
            img_feat2 = self.depth_fc_d2(img_feat1)
            img_feat_mid = self.depth_bn(img_feat2)
            img_feat_mid = self.depth_lrelu(img_feat_mid)
            img_feat3 = self.depth_fc_u2(img_feat_mid)
            img_feat3 = 0.5 * (img_feat3 + img_feat1)
            img_feat4 = self.depth_fc_u1(img_feat3)
            img_feat4 = 0.5 * (img_feat4 + img_feat)
            img_feat = img_feat4 
        img_feat = torch.unsqueeze(img_feat,2)
        img_feat = torch.unsqueeze(img_feat,3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        if self.multi_kp:
            pred_depths = gamma.view(-1,self.depth_num) * k_value.view(-1,1).expand(-1, self.depth_num)
            pred_depths = pred_depths / 1000.0
            root_index = self.kps_need_depth.index(reference_keypoint_id)
            pred_depth = pred_depths[:,root_index].reshape(-1,1)    
        else:
            pred_depth = gamma * k_value.view(-1,1)
            pred_depth = pred_depth.reshape(img_feat.size(0), 1) / 1000.0
        if self.multi_kp:
            return pred_depth, pred_depths
        else:
            return pred_depth
            

class KeypointNetPart(nn.Module):
    def __init__(self, backbone_name, num_joints, depth_dim, 
                height_dim, width_dim, norm_type, image_size, bbox_3d_shape,
                reference_keypoint_id, fix_root, 
                feature_channel, deconv_dim=[256, 256, 256]):
        super(KeypointNetPart, self).__init__()
        self.backbone_name = backbone_name
        self.feature_channel = feature_channel
        self.deconv_dim = deconv_dim
        self.image_size = image_size
        self.reference_keypoint_id = reference_keypoint_id
        if backbone_name in ["resnet", "resnet34", "resnet50", "resnet101"]:
            # self.avgpool = nn.AvgPool2d(int(self.image_size/32), stride=1)
            self.deconv_layers = self._make_deconv_layer()
            self.final_layer = nn.Conv2d(deconv_dim[2], num_joints * depth_dim, kernel_size=1, stride=1, padding=0)

        self.space_integral_layer = HeatmapIntegralPose(backbone=backbone_name, num_joints=num_joints, depth_dim=depth_dim,
                                                  height_dim=height_dim, width_dim=width_dim, norm_type=norm_type,
                                                  image_size=image_size, bbox_3d_shape=bbox_3d_shape, rootid=reference_keypoint_id,
                                                  fixroot=fix_root)
    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = nn.BatchNorm2d(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = nn.BatchNorm2d(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = nn.BatchNorm2d(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)
    def forward(self, out, xf, root_trans_from_rootnet, K):
        if self.backbone_name in ["resnet", "resnet50", "resnet34"]:
            # xf = self.avgpool(out)
            out = self.deconv_layers(out)
            out = self.final_layer(out)
            pred_uvd, pred_xyz_int = self.space_integral_layer(out, root_trans=root_trans_from_rootnet, K=K)
            pred_root_uv = (pred_uvd[:,self.reference_keypoint_id,:2]+ 0.5) * self.image_size
        elif self.backbone_name in ["hrnet", "hrnet32"]:
            pred_uvd, pred_xyz_int = self.space_integral_layer(out, root_trans=root_trans_from_rootnet, K=K)
            pred_root_uv = (pred_uvd[:,self.reference_keypoint_id,:2] + 0.5) * self.image_size

        return pred_uvd, pred_xyz_int, pred_root_uv
class RootNetwithRegInt(nn.Module):
    """ RootNetwithRegInt model with ResNet backbone

        In previous works' terminology:
        pose refers to the rotation configuration of the human keypoints (npose = 24*6)
        cam refers to the human-to-camera transformation

        In our work: (in order to be consistent with previous works such as HMR)
        pose refers to the configuration(angle) values of the robot joints  (npose = DoF)
        cam refers to the robot-to-camera 6D pose/transformation,
        which is presented in rotation(dim=6 representation, Zhou et al) and translation(dim=3)

        initialization parameters dict {
            "robot_type" : str
            "pose_params": dict
            "cam_params": array(float) (as in 4x4 array)
            "init_pose_from_mean": bool
        }
    """

    def __init__(self, init_param_dict, args, **kwargs):

        super(RootNetwithRegInt, self).__init__()

        robot_type = init_param_dict["robot_type"]
        if robot_type == "panda":
            DoF = 8
            nkpt = 7
        elif robot_type == "kuka":
            DoF = 7
            nkpt = 8
        elif robot_type == "baxter":
            DoF = 15
            nkpt = 17
        else:
            raise ValueError(f"Robot type {robot_type} is not supported.")
        npose = DoF
        self.robot = URDFRobot(robot_type)
        self.backbone_name = args.backbone_name                 #& resnet 50
        self.rootnet_backbone_name = args.rootnet_backbone_name #& hrnet 32
        self.use_rpmg = args.use_rpmg
        self.n_iter = args.n_iter
        self.norm_type = "softmax"
        self.deconv_dim = [256,256,256]
        self.num_joints = nkpt
        self.image_size = args.other_image_size
        self.depth_dim = 64
        self.height_dim = int(self.image_size/4)
        self.width_dim = int(self.image_size/4)
        self.bbox_3d_shape = args.bbox_3d_shape
        self.reference_keypoint_id = args.reference_keypoint_id
        self.rotation_dim = args.rotation_dim
        if self.backbone_name in ["resnet", "resnet34", "resnet50", "resnet101"]:
            self.reg_backbone = get_resnet(self.backbone_name)
            self.feature_channel = self.reg_backbone.block.expansion * 512
            self.avgpool = nn.AvgPool2d(int(self.image_size/32), stride=1)
        elif self.backbone_name in ["hrnet", "hrnet32"]:
            self.reg_backbone = get_hrnet(type_name=32, num_joints=self.num_joints, depth_dim=self.depth_dim,
                                      pretrain=True, generate_feat=True, generate_hm=True)
            self.feature_channel = 2048
        else:
            raise(NotImplementedError)

        self.reg_joint_map = args.reg_joint_map
        self.joint_conv_dim = None
        if self.reg_joint_map:
            self.joint_conv_dim = args.joint_conv_dim

        self.direct_reg_rot = args.direct_reg_rot
        self.rot_iterative_matmul = args.rot_iterative_matmul

        if self.rootnet_backbone_name in ["resnet",  "resnet50",  "resnet34"]:
            self.rootnet_backbone = get_resnet(self.rootnet_backbone_name)
            self.inplanes = self.rootnet_backbone.block.expansion * 512
        elif self.rootnet_backbone_name in ["hrnet",  "hrnet32"]:
            self.rootnet_backbone = get_hrnet(type_name=32, num_joints=nkpt, depth_dim=self.depth_dim,
                                            pretrain=True, generate_feat=True, generate_hm=False)
            self.inplanes = 2048
        else:
            raise(NotImplementedError)
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.multi_kp = args.multi_kp
        self.kps_need_depth = args.kps_need_depth if self.multi_kp else [args.reference_keypoint_id]
        self.depth_num = len(self.kps_need_depth)
        self.add_fc = args.add_fc
        
        #TODO 这一部分是Depthnet
        
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
        self.jointnet = JointNetPart(self.reg_joint_map, self.feature_channel, self.backbone_name,
                                self.joint_conv_dim, p_dropout=args.p_dropout, n_iter=self.n_iter, 
                                norm_type=self.norm_type, robot_type=robot_type, DoF=DoF)
        self.rotationnet = RotationNetPart(self.direct_reg_rot, self.rot_iterative_matmul,
                                           self.feature_channel, self.rotation_dim,
                                           p_dropout=args.p_dropout, n_iter=self.n_iter)
        self.keypointnet = KeypointNetPart(self.backbone_name, self.num_joints,
                                           self.depth_dim, self.height_dim, self.width_dim,
                                           self.norm_type, self.image_size, self.bbox_3d_shape,
                                           self.reference_keypoint_id, args.fix_root, 
                                           feature_channel=self.feature_channel, deconv_dim=self.deconv_dim
                                           )
        self.depthnet = DepthNetPart(self.multi_kp, self.kps_need_depth, self.add_fc, 
                                     self.inplanes, self.depth_num)  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_rot', init_rot)

    
    def forward(self, x_reg_input, x_root_input, k_value, K, init_pose=None, init_rot=None, test_fps=False):
        
        batch_size = x_reg_input.shape[0]
        x_reg_input = x_reg_input.to(torch.float)
        x_root_input = x_root_input.to(torch.float)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_rot is None:
            init_rot = self.init_rot.expand(batch_size, -1)
        root_trans_from_rootnet = torch.zeros((batch_size, 3)).float()

        # root z
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #^ Step 1: x_root_input 
        if self.rootnet_backbone_name in ["resnet34", "resnet50", "resnet", "resnet101"]:
            if test_fps:
                t_start_root = time.time()
            fm = self.rootnet_backbone(x_root_input)
            img_feat = torch.mean(fm.view(fm.size(0), fm.size(1), fm.size(2)*fm.size(3)), dim=2) # global average pooling
        elif self.rootnet_backbone_name in ["hrnet", "hrnet32"]:
            if test_fps:
                t_start_root = time.time()
            img_feat = self.rootnet_backbone(x_root_input)
        
        if self.multi_kp:
            pred_depth, pred_depths = self.depthnet(img_feat, k_value, reference_keypoint_id=self.reference_keypoint_id)

        else: 
            pred_depth = self.depthnet(img_feat, k_value, reference_keypoint_id=self.reference_keypoint_id)

        
        if test_fps:
            torch.cuda.current_stream().synchronize()
            t_end_root = time.time()
            time_root = t_end_root - t_start_root 
        root_trans_from_rootnet[:,2:3] = pred_depth
        
        if test_fps:
            t_start_other = time.time() 
        #^ Step2: x_reg_input

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.backbone_name in ["resnet", "resnet50", "resnet34"]:
            x_out = self.reg_backbone(x_reg_input)
            xf = self.avgpool(x_out)
            pred_uvd, pred_xyz_int, pred_root_uv = self.keypointnet(x_out, xf, root_trans_from_rootnet, K)

        elif self.backbone_name in ["hrnet", "hrnet32"]:
            out, xf = self.reg_backbone(x_reg_input)
            pred_uvd, pred_xyz_int, pred_root_uv = self.keypointnet(out, xf, root_trans_from_rootnet, K)

        pred_trans = uvz2xyz_singlepoint(pred_root_uv, pred_depth, K)

        # joint angle/pose, rotation (iterative)
        # pred_pose = init_pose
        # pred_rot = init_rot
        xf = xf.view(xf.size(0), -1)

        pred_pose = self.jointnet(xf, x_out, init_pose)
        # skiplist, skiplist2 = {}, {}
        
        pred_rot = self.rotationnet(xf, init_rot)
        if self.reference_keypoint_id == 0:
            pred_xyz_fk = self.robot.get_keypoints(pred_pose, pred_rot, pred_trans)
        else:
            pred_xyz_fk = self.robot.get_keypoints_root(pred_pose, pred_rot, pred_trans,root=self.reference_keypoint_id)
        
        if test_fps:
            torch.cuda.current_stream().synchronize()
            t_end_other = time.time() 
            time_other = t_end_other - t_start_other
            time_whole = t_end_other - t_start_root
        
        if test_fps:
            return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk, (time_root,time_other,time_whole)
        else:
            if self.multi_kp:
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_depths, pred_uvd, pred_xyz_int, pred_xyz_fk
            else:
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk


    
def get_rootNetwithRegInt_model(init_params_dict, args, **kwargs):
    """ Constructs a rootNetwithRegInt model with ResNet/Hrnet backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.backbone_name not in ["resnet", "resnet50", "resnet34","resnet101", "hrnet", "hrnet32"]:
        raise(NotImplementedError)
    if args.rootnet_backbone_name not in ["resnet", "resnet50", "resnet34", "hrnet", "hrnet32"]:
        raise(NotImplementedError)
    
    model = RootNetwithRegInt(init_params_dict, args, **kwargs)
    
    model.reg_backbone.init_weights(args.backbone_name)
    if args.rootnet_backbone_name not in ["hrnet", "hrnet32"]:
        model.rootnet_backbone.init_weights(args.rootnet_backbone_name)
    
    if args.pretrained_rootnet is not None:
        pretrained_path = args.pretrained_rootnet
        pretrained_checkpoint = torch.load(pretrained_path)
        print(f"Using {args.pretrained_rootnet} as pretrained rootnet weights for rootNetwithRegInt pipeline. ")
        pretrained_rootnet_weights = pretrained_checkpoint["model_state_dict"]
        pretrained_weights = {}
        for k, v in pretrained_rootnet_weights.items():
            if k.startswith("backbone"):
                new_k = k.replace("backbone", "rootnet_backbone")
            else:
                new_k = k
            pretrained_weights[new_k] = v
        # print(pretrained_weights.keys())
        model.load_state_dict(pretrained_weights, strict=False)
    else:
        print(f"Not using pretrained depthnet weights for the full network training stage. ")

    
    return model
    #^ 模型输入：
    #^ reg_images              [48, 3, 256, 256]
    #^ root_images             [48, 3, 256, 256]
    #^ k_values                [48, 1]
    #^ K = Other_K             [48, 3, 3]
    
    #TODO 模型输出：
    #TODO - pred_pose              [48, 8]     |--> gt_pose         loss_pose
    #TODO - pred_rot               [48, 6]     |--> gt_root_rot     loss_rot
    #TODO - pred_trans             [48, 3]     |--> gt_root_trans   loss_trans
    #TODO - pred_root_uv           [48, 2]     |--> gt_root_uv      loss_uv
    #TODO - pred_root_depth        [48, 1]     |--> gt_root_depth   loss_depth
    #TODO - pred_uvd               [48, 7, 3]  |--> (unused)
    #TODO - pred_keypoints3d_int   [48, 7, 3]  |--> gt_keypoints3d  loss_error3d_int  --> loss_error3d_align
    #TODO - pred_keypoints3d_fk    [48, 7, 3]  |--> gt_keypoints3d  loss_error3d      <-----------|
    #TODO - pred_keypoints2d_reproj_int [48, 7, 2]                  loss_error2d_int
    #TODO - pred_keypoints2d_reproj_fk  [48, 7, 2] 
    #& gt_root_rot
    #& gt_root_trans 
    #& gt_pose: Total 8 joints, gt_pose shape: torch.Size([48, 8])
    #& gt_root_depth: root_trans 的三维vector(x, y, z)取z
    #& gt_root_uv: gt_keypoints2d[48, 7, 2] 选择对应keypoint的2d 坐标(u, v)
        #* gt_keypoints3d shape: [Batch_size, n_keypoint_names, 3], where 3 for (x, y, z) 3d point
        #* 选择好对应的reference 作为root translation vector [48, 3]
        #* rotation部分则改为6dim vector：[48, 6]

    #* robot.link_name for panda: link 0,2,3,4,6,7,hand   7 keypoints

#* isinstance(sample, dict): True
            #* sample.keys(): dict_keys(['image_id', 'scene_id', 'images_original', 'bbox_strict_bounded_original', 'bbox_gt2d_extended_original', 'TCO', 'K_original', 'jointpose', 'keypoints_2d_original', 'valid_mask', 'keypoints_3d_original', 'root', 'other'])
            
            #* sample['image_id'].shape, sample['scene_id'].shape: torch.Size([48]), torch.Size([48])
            
            #* sample['images_original'].shape: torch.Size([48, 3, 480, 640])
            
            #* sampe['bbox_strict_bounded_original'].shape, sampe['bbox_gt2d_extended_original'].shape: torch.Size([48, 4]), torch.Size([48, 4])
            
            #* sample['TCO'].shape: torch.Size([48, 4, 4])
            
            #* sample['K_original'].shape: torch.Size([48, 3, 3]
            
            #* sample['jointpose'].keys(): odict_keys(['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_joint8', 'panda_hand_joint', 'panda_finger_joint1', 'panda_finger_joint2'])
            #* [x.shape for x in sample["jointpose"].values(): [torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48]), torch.Size([48])]
            
            #* sample['keypoints_2d_original'].shape: torch.Size([48, 7, 2])
            
            #* sample['valid_mask'].shape: torch.Size([48, 7])
            
            #* sample['keypoints_3d_original'].shape: torch.Size([48, 7, 3])
            
            #* sample['root'].keys():                              dict_keys(['images', 'bbox_strict_bounded', 'bbox_gt2d_extended', 'K', 'keypoints_3d', 'keypoints_2d', 'valid_mask_crop'])
            #* [x.shape for x in sample["root"].values()]: [torch.Size([48, 3, 256, 256]), torch.Size([48, 4]), torch.Size([48, 4]), torch.Size([48, 3, 3]), torch.Size([48, 7, 3]), torch.Size([48, 7, 2]), torch.Size([48, 7])]
            
            #* sample['other'].keys(): dict_keys(['images', 'bbox_strict_bounded', 'bbox_gt2d_extended', 'K', 'keypoints_3d', 'keypoints_2d', 'valid_mask_crop'])
            #* [x.shape for x in sample["other"].values()]: [torch.Size([48, 3, 256, 256]), torch.Size([48, 4]), torch.Size([48, 4]), torch.Size([48, 3, 3]), torch.Size([48, 7, 3]), torch.Size([48, 7, 2]), torch.Size([48, 7])]