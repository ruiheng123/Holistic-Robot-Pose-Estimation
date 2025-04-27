import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
torch.cuda.set_device(4)
from torchinfo import summary
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


model = load_dino_model("vits").cuda()
patch_h, patch_w = 36, 48
feat_dim = 384

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),  # 高斯模糊
    T.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
    T.CenterCrop((patch_h * 14, patch_w * 14)),  # 中心裁剪
    T.ToTensor(),  # 转换为张量
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
])

features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()


img_path = "data/dream/synthetic/panda_synth_test_dr/000005.rgb.jpg"
img = Image.open(img_path).convert('RGB')
imgs_tensor[0] = transform(img)[:3]
print(f"img_trans has shape: {transform(img).shape}, img_tensor[0] shape: {imgs_tensor[0].shape}")
with torch.no_grad():
    # 将图像张量传递给dinov2_vits14模型获取特征
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()

pca = PCA(n_components=3)
pca.fit(features)

# 对PCA转换后的特征进行归一化处理
pca_features = pca.transform(features)
print(pca_features.shape)
import time 
time.sleep(1)
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
pca_features_fg = pca_features[:, 0] > 0.3
pca_features_bg = ~pca_features_fg
b = np.where(pca_features_bg)
pca.fit(features[pca_features_fg])
pca_features_rem = pca.transform(features[pca_features_fg])
for i in range(3):
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_fg] = pca_features_rem
pca_features_rgb[b] = 0
pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
plt.subplot(121)
plt.imshow(np.array(img))
plt.subplot(122)
plt.imshow(pca_features_rgb[0][...,::-1])
plt.savefig("image2.png")
plt.show()
# summary(model, input_size=(1, 3, 252, 252))