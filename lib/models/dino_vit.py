import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
dino_model_dict = {
    "vits": 'dinov2_vits14',
    "vitb": 'dinov2_vitb14',
    "vitl": 'dinov2_vitl14',
    "vitg": 'dinov2_vitg14',
    "vits_reg": 'dinov2_vits14_reg',
    "vitb_reg": 'dinov2_vitb14_reg',
    "vitl_reg": 'dinov2_vitl14_reg',
    "vitg_reg": 'dinov2_vitg14_reg',
}
dam_model_dict = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
def load_dino_model(model_name, pretrained=True):
    print(f"Loading DINO model {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', dino_model_dict[model_name], pretrained=pretrained)
    return model

def load_dam_model(model_name, checkpoint_path):
    print(f"Loading DAM model {model_name}...")
    model = DepthAnythingV2(**dam_model_dict[model_name])
    model.load_state_dict(torch.load(checkpoint_path))
    return model