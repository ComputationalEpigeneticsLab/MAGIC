import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention



#
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attention=False):
        norm_x = self.norm(x)
        if return_attention:
            attn_output, attn_matrix = self.attn(norm_x, return_attn_matrices=True)
            return x + attn_output, attn_matrix
        else:
            return x + self.attn(norm_x)

#
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        #
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL_wsi(nn.Module):
    def __init__(self, n_classes=1,patch_feature_dim =384,device = 'cuda:0'):
        super(TransMIL_wsi, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(patch_feature_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.device = device

    def forward(self, data: torch.Tensor,  return_attention=False):
        h = data.float()

        h = self._fc1(h)

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)

        # ---->Translayer x2
        if return_attention:
            h, attn_matrix = self.layer2(h,return_attention=return_attention)
            # ---->cls_token
            h = self.norm(h)
            h_cls = h[:, 0]
            return h_cls.squeeze(0).unsqueeze(0), attn_matrix, h.squeeze(0)

        else:
            h = self.layer2(h)
            # ---->cls_token
            h = self.norm(h)
            h_cls = h[:, 0]
            return h_cls.squeeze(0).unsqueeze(0)