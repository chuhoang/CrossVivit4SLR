import torch
import numpy
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple
# from .t2t import T2T, get_sinusoid_encoding
from timm.models.vision_transformer import _cfg, Mlp, Block, trunc_normal_
from timm.models.vision_transformer import Attention
# Việc cần làm:
# Xây cros attention => (Q - cls token, k, v-patch token) => linear + dropout
# cros attention block => cros attention + block(attn)??? + mlp
# Xây ViVit => concat patch emb A + cls token B and revert (đã flatten hết patch embed và cls token ra) => cros
from timm.models.layers import to_2tuple
from Backbone import I3D
class BackboneI3D(torch.nn.Module):
    def __init__(self, model, weight_pretrained):
        super().__init__()
        self.extract_feature = model
        if weight_pretrained is not None:
            
            self.checkpoint = torch.load(weight_pretrained) 
            
            del self.checkpoint['features.18.weight']
            del self.checkpoint['features.18.bias']
            self.extract_feature.load_state_dict(self.checkpoint)
    def forward(self, input):
        out = self.extract_feature(input)
        return out
    
class PatchEmbed1(torch.nn.Module):
    def __init__(self, input_shape, embed_dim, patch_size, tubelet_size):
        super().__init__()
        self.input_shape = input_shape
        patch_size = to_2tuple(patch_size) 
        self.conv3d = torch.nn.Conv3d(
            in_channels=1024,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
            )
    def forward(self, inp):
        out = self.conv3d(inp)
        return out.flatten(2).transpose(1,2)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=None, has_mlp=True):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            # self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(x))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(x))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=None):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=nn.LayerNorm))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=nn.LayerNorm,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=nn.LayerNorm,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [ act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(12, 16), tubelet =(4, 8),  in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.5, attn_drop_rate=0.1,
                 drop_path_rate=0.2, hybrid_backbone=None, norm_layer=None, weight_pretrained=None, multi_conv=False):
        super().__init__()
        self.gradients = None
        self.activations = None
        self.pretrained_weight = weight_pretrained
        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size
        num_patches = (196, 18)
        self.num_branches = len(patch_size)
        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.InceptionI3d = I3D(self.num_classes)
            self.backbone = BackboneI3D(model = self.InceptionI3d, weight_pretrained = self.pretrained_weight)
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1+num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d, tubelet in zip(img_size, patch_size, embed_dim, tubelet):
                self.patch_embed.append(PatchEmbed1(input_shape=im_s, embed_dim=d, patch_size=p, tubelet_size=tubelet))
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=0.5)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr+curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=nn.LayerNorm)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
        self.num_classes = num_classes
        self.drop_out = nn.Dropout(p = 0.5)
        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
        # self.head_auxilary =  nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def activations_hook(self, grad):
#         print('ghudfidfhids')
        self.gradients.append(grad)
#         print(self.gradients)
        
    # method for the gradient extraction
    def get_activations_gradient(self):
#         print(self.gradients)
        # print(self.gradients)
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        B, T, C, H, W = x.shape
        xs = []
        x=x.transpose(1,2)
        x_ = self.backbone(x)
        activations = []
        for i in range(self.num_branches):
            # x = torch.nn.functional.interpolate(x, size=(C, self.img_size[i], self.img_size[i]), mode='nearest-exact') if H != self.img_size[i] else x
            # x_ = self.backbone(x_)
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)

            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)
        for blk in self.blocks:
            torch.cuda.empty_cache()
            xs = blk(xs)
        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        [activations.append(x) for x in xs]
        
        return activations

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, T, C, H, W = x.shape
        xs = []
        x=x.transpose(1,2)
        x_ = self.backbone(x)
        # print(x_.shape,'==================')
        for i in range(self.num_branches):
            # x = torch.nn.functional.interpolate(x, size=(C, self.img_size[i], self.img_size[i]), mode='nearest-exact') if H != self.img_size[i] else x
            # x_ = self.backbone(x_)
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)

            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)
        for blk in self.blocks:
            torch.cuda.empty_cache()
            xs = blk(xs)
        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        cnt=0
        for x in xs:
            h = x.register_hook(self.activations_hook) 
        out = [x[:, 0] for x in xs]
        return out

    def forward(self, x):
        self.gradients = []
        # x = x.transpose(1, 2)
        feats = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(feats)]
        # ce_logits = torch.squeeze(ce_logits, dim = 1)
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        # self.get_activations(ce_logits)
        # h = ce_logits.register_hook(self.activations_hook)
        return ce_logits