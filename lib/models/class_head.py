import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import MixerLayer

class FCBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClassificationHead(nn.Module):

    def __init__(self,
                 in_channels: int, 
                 image_size, 
                 num_joints: int, #! input
                 conv_channels: int,
                 hidden_dim: int, 
                 num_blocks: int,
                 hidden_inter_dim: int, 
                 token_inter_dim: int, 
                 dropout: float,

                 token_num: int, 
                 token_class_num: int, 
 
                 tokenizer = None, 
                 stage: str = "classifier"
    ) -> None:
        super(ClassificationHead, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_joints = num_joints

        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.hidden_inter_dim = hidden_inter_dim
        self.token_inter_dim = token_inter_dim
        self.dropout = dropout

        self.token_num = token_num
        self.token_class_num = token_class_num

        self.tokenizer = tokenizer
        

        self.stage = stage
        if self.stage == "classifier" and (self.tokenizer):
            self.tokenizer.eval()
            for _, params in self.tokenizer.named_parameters():
                params.requires_grad = False
            total_params = sum(p.numel() for p in self.tokenizer.parameters() if p.requires_grad)
            print(f"Tokneizer 可训练参数数量: {total_params}")

        if self.stage == "classifier":
            self.conv_trans = self._make_transition_for_head(
                in_channels, self.conv_channels)
            
            
            self.conv_head = self._make_cls_head(self.conv_channels)

            input_size = (int(image_size[0]//32)) * (int(image_size[1]//32))
            self.mixer_trans = FCBlock(
                self.conv_channels * input_size, 
                self.token_num * self.hidden_dim)

            self.mixer_head = nn.ModuleList(
                [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                    self.token_num, self.token_inter_dim,  
                    self.dropout) for _ in range(self.num_blocks)])
            self.mixer_norm_layer = FCBlock(
                self.hidden_dim, self.hidden_dim)

            self.cls_pred_layer = nn.Linear(
                self.hidden_dim, self.token_class_num)

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, channel, num_blocks=3, dilation=1):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            channel,
            channel,
            num_blocks,
            dilation=dilation)
        feature_convs.append(feature_conv)
        
        
        return nn.ModuleList(feature_convs)
    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def extract_joints_feat(self, feature_map, joint_coords):
        assert self.image_size[1] == self.image_size[0], \
            'If you want to use a rectangle input, ' \
            'please carefully check the length and width below.'
        batch_size, _, _, height = feature_map.shape
        stride = self.image_size[0] / feature_map.shape[-1]
        joint_x = (joint_coords[:,:,0] / stride + 0.5).int()
        joint_y = (joint_coords[:,:,1] / stride + 0.5).int()
        joint_x = joint_x.clamp(0, feature_map.shape[-1] - 1)
        joint_y = joint_y.clamp(0, feature_map.shape[-2] - 1)
        joint_indices = (joint_y * height + joint_x).long()

        flattened_feature_map = feature_map.clone().flatten(2)
        joint_features = flattened_feature_map[
            torch.arange(batch_size).unsqueeze(1), :, joint_indices]

        return joint_features

    def forward(self, x, joints=None, train=True):
        if self.stage == "classifier":
            batch_size = x.shape[0]
            cls_feat = self.conv_head[0](self.conv_trans(x))
            cls_feat = cls_feat.flatten(2).transpose(2,1).flatten(1)
            cls_feat = self.mixer_trans(cls_feat)
            cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)
            for mixer_layer in self.mixer_head:
                cls_feat = mixer_layer(cls_feat)
            cls_feat = self.mixer_norm_layer(cls_feat)
            
            cls_logits = self.cls_pred_layer(cls_feat)
            encoding_scores = cls_logits.topk(1, dim=2)[0]
            cls_logits = cls_logits.flatten(0,1)
            cls_logits_softmax = cls_logits.clone().softmax(1)

            
            if train:
                reconstruct, cls_label, _ = self.tokenizer(joints, cls_logits_softmax, train=train)
                return cls_logits, reconstruct, cls_label
            else:
                reconstruct, cls_label, _ = self.tokenizer(joints, cls_logits_softmax, train=train)
                return reconstruct, encoding_scores
