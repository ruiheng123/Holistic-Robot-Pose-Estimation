import torch
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

def load_dino_model(model_name, pretrained=True):

    model = torch.hub.load('facebookresearch/dinov2', dino_model_dict[model_name], pretrained=pretrained)
    return model