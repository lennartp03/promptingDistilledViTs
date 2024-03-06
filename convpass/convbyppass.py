import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath

def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False, distilled=True):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(192, dim) 
        self.adapter_up = nn.Linear(dim, 192)  
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.distilled = distilled

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x) 
        x_down = self.act(x_down)

        sliced_token = 2 if self.distilled else 1
        x_patch = x_down[:, sliced_token:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2) 
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        zero_dist_token = torch.zeros_like(x_cls)

        if self.distilled:
          x_down = torch.cat([x_cls, zero_dist_token, x_patch], dim=1)
        else:
          x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  

        return x_up



def set_Convpass(model, method, dim=8, s=1, xavier_init=False, distilled=True):
    if method == 'convpass':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init, distilled)
                _.adapter_mlp = Convpass(dim, xavier_init, distilled)
                _.s = s
                _.drop_path = DropPath(drop_prob=0.1)
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init)
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init)
                _.s = s
                bound_method = forward_block_attn.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init)