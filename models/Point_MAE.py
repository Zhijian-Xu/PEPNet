import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from utils.logger import *
from tools.loss import FocalLoss, DiceLoss
    
class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel, feature_dim=11):
        super().__init__()
        self.encoder_channel = encoder_channel
        # self.first_conv = nn.Sequential( 
        #     nn.Conv1d(feature_dim, 128, 1),
        #     nn.LayerNorm(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, 1)
        # )
        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(512, 512, 1),
        #     nn.LayerNorm(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, self.encoder_channel, 1)
        # )
        self.first_conv1 = nn.Conv1d(feature_dim, 128, 1)
        self.first_ln = nn.LayerNorm(128)
        self.first_conv2 = nn.Conv1d(128, 256, 1)

        self.second_conv1 = nn.Conv1d(512, 512, 1)
        self.second_ln = nn.LayerNorm(512)
        self.second_conv2 = nn.Conv1d(512, self.encoder_channel, 1)
        
    def forward(self, point_groups): 
        bs, g, n, _ = point_groups.shape 
        input_feat = point_groups.reshape(bs * g, n, -1) 
        x = input_feat.transpose(2, 1)  

        x = self.first_conv1(x) 
        x = x.transpose(1, 2) 
        x = self.first_ln(x) 
        x = x.transpose(1, 2)   
        x = F.relu(x)
        x = self.first_conv2(x) 

        x_global = torch.max(x, dim=2, keepdim=True)[0]  
        x = torch.cat([x_global.expand(-1, -1, n), x], dim=1) 

        x = self.second_conv1(x) 
        x = x.transpose(1, 2)     
        x = self.second_ln(x)     
        x = x.transpose(1, 2)    
        x = F.relu(x)
        x = self.second_conv2(x)  

        x_global = torch.max(x, dim=2, keepdim=False)[0] 
        return x_global.reshape(bs, g, self.encoder_channel) 


class ResidueBasedGroup(nn.Module):
    def __init__(self, max_atoms_per_residue=24,atom_feature_dims=8):
        super().__init__()
        self.max_atoms_per_residue = max_atoms_per_residue
        self.atom_dims = atom_feature_dims + 3
    
    def forward(self, points, residue_starts, mask=None):
        batch_size = points.shape[0]
        device = points.device
        max_residues = max([len(r) for r in residue_starts])
        

        neighborhoods = []
        centers = []
        residue_masks = []
        for b in range(batch_size):
            sample_points = points[b]  
            sample_residue_starts = residue_starts[b]  # [num_residues]
            
            if mask is not None:
                sample_mask = mask[b]  # [N]
                valid_points = sample_points[sample_mask > 0]
            else:
                valid_points = sample_points
            residue_ends = torch.cat([sample_residue_starts[1:].to(device), torch.tensor([len(valid_points)], device=device)])
            residue_atoms = []
            residue_centers = []
            residue_mask = []
            
            for start_idx, end_idx in zip(sample_residue_starts, residue_ends):
               
                atoms = valid_points[start_idx:end_idx]  # [num_atoms, 3]
                
                if len(atoms) == 0:
                    center = torch.zeros(3, device=device)
                    relative_atoms = torch.zeros(1, self.atom_dims , device=device)
                    residue_mask.append(torch.tensor(0, device=device))
                else:
                    center = atoms[:, :3].mean(dim=0) 
                    relative_atoms = atoms.clone()
                    relative_atoms[:, :3] = atoms[:, :3] - center.unsqueeze(0)  
                    residue_mask.append(torch.tensor(1, device=device))
                
                padded_atoms = torch.zeros(self.max_atoms_per_residue, self.atom_dims , device=device)
                num_atoms = min(relative_atoms.shape[0], self.max_atoms_per_residue)
                padded_atoms[:num_atoms] = relative_atoms[:num_atoms] 
                
                residue_atoms.append(padded_atoms)
                residue_centers.append(center)
            
            num_residues = len(residue_atoms)
            if num_residues > 0: 
                sample_neighborhoods = torch.stack(residue_atoms, dim=0)
                sample_centers = torch.stack(residue_centers, dim=0)
                sample_residue_mask = torch.stack(residue_mask, dim=0)
            else:
                sample_neighborhoods = torch.zeros(0, self.max_atoms_per_residue,self.atom_dims , device=device)
                sample_centers = torch.zeros(0, 3, device=device)
                sample_residue_mask = torch.zeros(0, device=device)
            
            padded_neighborhoods = torch.zeros(max_residues, self.max_atoms_per_residue,self.atom_dims , device=device)
            padded_centers = torch.zeros(max_residues, 3, device=device)
            padded_residue_mask = torch.zeros(max_residues, device=device)
            
            if num_residues > 0:
                padded_neighborhoods[:num_residues] = sample_neighborhoods
                padded_centers[:num_residues] = sample_centers
                padded_residue_mask[:num_residues] = sample_residue_mask 
            
            neighborhoods.append(padded_neighborhoods)
            centers.append(padded_centers)
            residue_masks.append(padded_residue_mask)
        
        neighborhood = torch.stack(neighborhoods, dim=0) 
        center = torch.stack(centers, dim=0) 
        residue_mask = torch.stack(residue_masks, dim=0)

        return neighborhood, center, residue_mask


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        self.dim = dim 
        inv_freq = 1.0 / (2048 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute position encodings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # https://www.jianshu.com/p/e8be3dbfb4c5
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, seq_len_q, seq_len_k=None):
        """
        Args:
            seq_len_q: query sequence length
            seq_len_k: key sequence length (for cross attention)
        """
        if seq_len_k is None:
            seq_len_k = seq_len_q
            
        cos_q = self.cos_cached[:seq_len_q, :]
        sin_q = self.sin_cached[:seq_len_q, :]
        cos_k = self.cos_cached[:seq_len_k, :]
        sin_k = self.sin_cached[:seq_len_k, :]
        
        return cos_q, sin_q, cos_k, sin_k

def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k):
    """Apply rotary position embedding to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    # For query
    cos_q = cos_q.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, dim]
    sin_q = sin_q.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, dim]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    
    # For key
    cos_k = cos_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_k, dim]
    sin_k = sin_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_k, dim]
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., cross=False, use_rope=False,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.cross = cross
        self.use_rope = use_rope

        # --- Keep original module definitions for weight loading ---
        if not self.cross:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            # Using ModuleList to keep compatibility
            self.qkv = nn.ModuleList([
                nn.Linear(dim, dim, bias=qkv_bias), # Q proj
                nn.Linear(dim, dim, bias=qkv_bias), # K proj
                nn.Linear(dim, dim, bias=qkv_bias)  # V proj
            ])
        
        # Initialize RoPE if enabled
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(head_dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None, key_padding_mask=None):
        B, N, C = x.shape
        
        if not self.cross:
            assert k is None and v is None, "k and v should not be provided in self-attention mode"

            qkv_proj = self.qkv(x)
            q_proj, k_proj, v_proj = qkv_proj.chunk(3, dim=-1)

            q, k, v = q_proj, k_proj, v_proj

            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # Apply RoPE for self-attention
            if self.use_rope:
                cos_q, sin_q, cos_k, sin_k = self.rope(q.size(2))
                q, k = apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k)
        else:
            q_proj = self.qkv[0](x)
            k_proj = self.qkv[1](k)
            v_proj = self.qkv[2](v)

            q, k, v = q_proj, k_proj, v_proj

            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            k = k.reshape(B, k.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, v.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # Apply RoPE for cross-attention
            if self.use_rope:
                seq_len_q = q.size(2)
                seq_len_k = k.size(2)
                cos_q, sin_q, cos_k, sin_k = self.rope(seq_len_q, seq_len_k)
                q, k = apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_padding_mask == 0, -1e9)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross=False,  use_rope=False,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross = cross
        self.use_rope = use_rope
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, cross=cross, use_rope=self.use_rope,)
        
    def forward(self, x, kv=None, key_padding_mask=None):
        if not self.cross:
            x = x + self.drop_path(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        else:
            kv = self.norm2(kv)
            x = x + self.drop_path(self.attn(self.norm1(x), kv, kv, key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_rope=True):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                use_rope= use_rope,
                )
            for i in range(depth)])

    def forward(self, x, pos, key_padding_mask=None):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, key_padding_mask=key_padding_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                cross = True, 
                )
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, ag, ab, ag_pos, ab_pos, ag_padding_mask=None, ab_padding_mask=None):
        for _, block in enumerate(self.blocks):
            ag = block(ag+ag_pos, ab+ab_pos, key_padding_mask=ab_padding_mask) 
        ag = self.head(self.norm(ag)) 
        return ag
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        self.use_PLMs = config.get('use_PLMs', False) 
        self.use_rope = config.get('use_rope', True)
        self.atom_feature_dims = config.get('atom_feature_dims', 8) 
        self.aa_feature_dims = config.get('aa_feature_dims', 62) 
        self.encoder = Encoder(encoder_channel=self.trans_dim, feature_dim=3 + self.atom_feature_dims)
        if self.use_PLMs:
            self.aa_proj_dim = 64 
            self.aa_projector_antigen = nn.Sequential(
                nn.Linear(config.antigen_dims+self.aa_feature_dims, 128),
                nn.GELU(),
                nn.Linear(128, self.aa_proj_dim),
                nn.GELU(),
            )
            self.aa_projector_antibody = nn.Sequential(
                nn.Linear(config.antibody_dims+self.aa_feature_dims, 128),
                nn.GELU(),
                nn.Linear(128, self.aa_proj_dim),
                nn.GELU(),
            )
            self.aa_dropout = nn.Dropout(p=0.2)  
            self.encoder2 = Encoder(encoder_channel=self.trans_dim, feature_dim=self.trans_dim+self.aa_proj_dim) 

        else:
            self.encoder2 = Encoder(encoder_channel=self.trans_dim, feature_dim=self.trans_dim + self.aa_feature_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            use_rope = self.use_rope
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, aa_features, padding_mask=None, is_antigen=False):
        """
        Args:
            neighborhood (torch.Tensor): (B, G, max_atoms, 3+F)
            center (torch.Tensor): (B, G, 3)
            aa_features (torch.Tensor): (B, G, 62)
            padding_mask (torch.Tensor): (B, G)
            rotation_matrix (torch.Tensor, optional): (B, 3, 3). Defaults to None.
        """
        group_input_tokens = self.encoder(neighborhood)
        if self.use_PLMs:
            if is_antigen:
                aa_embed = self.aa_projector_antigen(aa_features)   
            else:
                aa_embed = self.aa_projector_antibody(aa_features)
            aa_embed = self.aa_dropout(aa_embed)
        
            group_input_tokens = torch.cat([group_input_tokens, aa_embed], dim=-1)
        else:
            group_input_tokens = torch.cat([group_input_tokens, aa_features], dim=-1)
        group_input_tokens = self.encoder2(group_input_tokens.unsqueeze(2))
        pos_geo = self.pos_embed(center)
  
        x = self.blocks(group_input_tokens, pos_geo, key_padding_mask=padding_mask)
        x = self.norm(x)
        
        return x, pos_geo
       
@MODELS.register_module()
class Point_MAE_Epitope(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim 

        self.shared_encoder = MaskTransformer(config)

        self.drop_path_rate = config.transformer_config.drop_path_rate
      
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder( 
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.max_atoms_per_residue = config.get('max_atoms_per_residue', 24)
        print_log(f'[Point_MAE_Epitope] Using residue-based grouping with max {self.max_atoms_per_residue} atoms per residue', logger='Point_MAE')
        self.group_divider = ResidueBasedGroup(max_atoms_per_residue=self.max_atoms_per_residue, atom_feature_dims=config.get('atom_feature_dims', 8))

        self.cls_head = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1) 
        )

        self.loss = config.loss
        self.focal_weight = config.get('focal_loss_weight', 20.0)
        self.dice_weight = config.get('dice_loss_weight', 1.0)
        self.bce_weight = config.get('bce_loss_weight', 1.0)
        self.rotation_weight = config.get('rotation_loss_weight', 1.0)  


        focal_alpha = config.get('focal_alpha', 0.25)
        focal_gamma = config.get('focal_gamma', 2.0)
        self.focal_loss_func = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

        # Dice Loss parameters (can be moved to config)
        dice_smooth = config.get('dice_smooth', 1.0)
        self.dice_loss_func = DiceLoss(smooth=dice_smooth)


        self.atom_dims = config.get('atom_feature_dims', 8)
        self.aa_remove = config.get('aa_feature_remove', None)

        self.sa_noise_std = config.get('sa_noise_std', 0.02) 
        self.nc_noise_std = config.get('nc_noise_std', 0.005) 

    
        self.bce_loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    # 评估函数
    def forward(self, data_dict, **kwargs):

        antigen_atom_features = data_dict['atom_features_A'][:, :, :self.atom_dims]
        antibody_atom_features = data_dict['atom_features_antibody'][:, :, :self.atom_dims]

        aa_features_antigen_full = data_dict['aa_features_A']
        aa_features_antibody_full = data_dict['aa_features_antibody']

        aa_features_antigen = aa_features_antigen_full
        aa_features_antibody = aa_features_antibody_full

        # 提取输入数据
        antigen_points = data_dict["xyz_A"]
        antibody_points = data_dict["xyz_antibody"]
        antigen_atom_mask = data_dict['atom_A_mask']
        antibody_atom_mask = data_dict['atom_ab_mask']
        residue_starts_antigen = data_dict["residue_starts_A"]
        residue_starts_antibody = data_dict["residue_starts_antibody"]
        aa_antigen_mask = data_dict['aa_A_mask']
        aa_antibody_mask = data_dict['aa_ab_mask']
        epitope = data_dict.get("epitope", None)
        device = data_dict['xyz_A'].device
        batch_size = data_dict["xyz_A"].shape[0]
  
        antigen_points_with_features = torch.cat([antigen_points, antigen_atom_features], dim=-1)
        antibody_points_with_features = torch.cat([antibody_points, antibody_atom_features], dim=-1)
        
        epitope = data_dict.get("epitope", None)


        antigen_neighborhood, antigen_center, antigen_residue_masks =  self.group_divider(
            antigen_points_with_features, residue_starts_antigen, antigen_atom_mask) #

        antibody_neighborhood, antibody_center, antibody_residue_masks = self.group_divider(
            antibody_points_with_features, residue_starts_antibody, antibody_atom_mask)

        assert torch.all(antigen_residue_masks == aa_antigen_mask), \
            f"antigen_residue_masks and aa_antigen_mask mismatch: {antigen_residue_masks.shape} vs {aa_antigen_mask.shape}"
        assert torch.all(antibody_residue_masks == aa_antibody_mask), \
            f"antibody_residue_masks and aa_antibody_mask mismatch: {antibody_residue_masks.shape} vs {aa_antibody_mask.shape}"
        
        ab_feat_rotated, ab_pos = self.shared_encoder(antibody_neighborhood, antibody_center, aa_features_antibody, antibody_residue_masks, is_antigen=False) # , rotation_matrix=predicted_rotation_matrix)

        antigen_features, ag_pos = self.shared_encoder(antigen_neighborhood, antigen_center, aa_features_antigen, antigen_residue_masks, is_antigen=True)


        if torch.isnan(antigen_features).any():
            import pdb;pdb.set_trace()

        decoded_features = self.MAE_decoder(
        antigen_features, ab_feat_rotated, 
        ag_pos, ab_pos,
        ag_padding_mask=antigen_residue_masks, 
        ab_padding_mask=antibody_residue_masks
        ) 
               
        batch_size, num_residues, feature_dim = decoded_features.shape
        flattened_features = decoded_features.reshape(-1, feature_dim)
        logits = self.cls_head(flattened_features) 
        logits = logits.reshape(batch_size, num_residues)
        valid_mask = antigen_residue_masks > 0  
        valid_logits = logits[valid_mask] 
        valid_epitope = epitope.view(batch_size, -1)[valid_mask].float() 
        if len(valid_epitope) > 0:
            epsilon = 0.2
            y_smooth = (1 - epsilon) * valid_epitope + epsilon * 0.5
            loss_focal = self.focal_loss_func(valid_logits, y_smooth)
            loss_bce = self.bce_loss_func(valid_logits, y_smooth)
            valid_probs = torch.sigmoid(valid_logits)
            loss_dice = self.dice_loss_func(valid_probs, y_smooth)

            if self.loss=='bce':
                return valid_probs, loss_bce, valid_epitope, loss_focal, loss_bce, loss_dice
            elif self.loss=='focal':
                raise NotImplementedError("Focal loss is not implemented yet.")
                return valid_probs, loss_rotation, valid_epitope, loss_focal, loss_bce, loss_dice
            elif self.loss=='dice':
                raise NotImplementedError("Dice loss is not implemented yet.")
                return valid_probs, loss_rotation, valid_epitope, loss_focal, loss_bce, loss_dice
            
@MODELS.register_module()
class Point_MAE_Epitope_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim 
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.solvent_dim = config.get('solvent_dim', 2)
        self.num_aa = config.get('num_aa_types', 20)

        # Encoders
        self.shared_encoder = MaskTransformer(config)

        self.max_atoms_per_residue = config.get('max_atoms_per_residue', 24)
        print_log(f'[Point_MAE_Epitope] Using residue-based grouping with max {self.max_atoms_per_residue} atoms per residue', logger='Point_MAE')
        self.group_divider = ResidueBasedGroup(max_atoms_per_residue=self.max_atoms_per_residue, atom_feature_dims=config.get('atom_feature_dims', 8))
        
        self.solvent_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim//2), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.trans_dim//2, self.solvent_dim)
        )
        self.pssm_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim//2), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.trans_dim//2, 20)
        )
        self.aa_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim//2), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.trans_dim//2, self.num_aa)
        )

        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

        self.sa_noise_std = config.get('sa_noise_std', 0.02)
        self.nc_noise_std = config.get('nc_noise_std', 0.005) 

        self.aa_weight = config.get('aa_weight', 1.0)
        self.sol_weight = config.get('solvent_weight', 20.0) 
        self.pssm_weight = config.get('pssm_weight', 1.0)
        self.atom_dims = self.config.get('atom_feature_dims', 8)
        self.aa_remove = self.config.get('aa_feature_remove', None)
        self.mask_ratio_for_zero = self.config.get('mask_ratio_for_zero', 0.8)
        self.use_PLMs = config.get('use_PLMs', False) 
        self.antibody_PLM_dim = config.get('antibody_dims', 512)
        self.antigen_PLM_dim = config.get('antigen_dims', 480)

        self.data_augmentation = config.get('data_augmentation', True)

    def apply_feature_ablation(self, aa_features):
        """Apply feature ablation while preserving PLM dimensions"""
        if self.aa_remove == 'PSSM':
            # Remove PSSM (positions 42:62) but keep PLMs
            if self.use_PLMs:
                return torch.cat([
                    aa_features[:, :, :42],  # One-hot + NC + SA/RSA
                    aa_features[:, :, 62:]   # PLM features
                ], dim=-1)
            else:
                return aa_features[:, :, :42]  # Just remove PSSM
                
        elif self.aa_remove == 'NC':
            # Remove neighbor frequencies (positions 20:40) but keep PLMs
            if self.use_PLMs:
                return torch.cat([
                    aa_features[:, :, :20],   # One-hot AA
                    aa_features[:, :, 40:]    # SA/RSA + PSSM + PLM
                ], dim=-1)
            else:
                return torch.cat([
                    aa_features[:, :, :20],   # One-hot AA
                    aa_features[:, :, 40:62]  # SA/RSA + PSSM
                ], dim=-1)
        else:
            return aa_features  # No ablation
    
    def apply_masking_and_corruption(self, aa_features, mask_zero_idx, mask_rand_idx, mask_rand, is_antigen=True, use_PLMs = False, data_augmentation=True):
        """Apply masking and corruption to amino acid features"""
        aa_corrupt = aa_features.clone()
        
        # Add noise to SA and NC features
        if self.aa_remove != 'NC' and data_augmentation:
            noise = torch.randn_like(aa_corrupt[:, :, 20:40]) * self.nc_noise_std
            aa_corrupt[:, :, 20:40] += noise
        solvent_start_idx = 20 if self.aa_remove == 'NC' else 40
        if data_augmentation:
            # Always add noise to SA features (at position 40:42 in original, may shift with ablation)
            noise = torch.randn_like(aa_corrupt[:, :, solvent_start_idx:solvent_start_idx+2]) * self.sa_noise_std
            aa_corrupt[:, :, solvent_start_idx:solvent_start_idx+2] += noise
        
        # Apply zero masking
        if not is_antigen:
            pssm_end_idx = self.antibody_PLM_dim
        else:
            pssm_end_idx = self.antigen_PLM_dim
        aa_corrupt[mask_zero_idx[0], mask_zero_idx[1], :20] = 0.0  # Zero AA type
        aa_corrupt[mask_zero_idx[0], mask_zero_idx[1], solvent_start_idx:solvent_start_idx+2] = 0.0  # Zero solvent
        if use_PLMs:
            aa_corrupt[mask_zero_idx[0], mask_zero_idx[1], -20-pssm_end_idx:-pssm_end_idx] = 0.0  # Zero PSSM
        else:
            aa_corrupt[mask_zero_idx[0], mask_zero_idx[1], -20:] = 0.0  # Zero PSSM
        
        # Apply random masking
        aa_corrupt[mask_rand_idx[0], mask_rand_idx[1], :20] = F.one_hot(
            torch.randint(20, (mask_rand.sum(),), device=aa_features.device), self.num_aa
        ).float()
        aa_corrupt[mask_rand_idx[0], mask_rand_idx[1], solvent_start_idx:solvent_start_idx+2] = torch.rand(
            mask_rand.sum(), self.solvent_dim, device=aa_features.device
        )
        if use_PLMs:
            aa_corrupt[mask_rand_idx[0], mask_rand_idx[1], -20-pssm_end_idx:-pssm_end_idx] = torch.randn(
                mask_rand.sum(), 20, device=aa_features.device
            )
        else:
            aa_corrupt[mask_rand_idx[0], mask_rand_idx[1], -20:] = torch.randn(
                mask_rand.sum(), 20, device=aa_features.device
            )
        
        return aa_corrupt
    
    # 评估函数
    def forward(self, data_dict):
        # Select atom features
        atom_features_A = data_dict['atom_features_A'][:, :, :self.atom_dims]
        atom_features_antibody = data_dict['atom_features_antibody'][:, :, :self.atom_dims]

        # Select amino acid features
        aa_features_A_full = data_dict['aa_features_A']
        aa_features_antibody_full = data_dict['aa_features_antibody']

        # Apply feature ablation while preserving PLM dimensions
        aaA = self.apply_feature_ablation(aa_features_A_full)
        aaB = self.apply_feature_ablation(aa_features_antibody_full)

        ptsA = torch.cat([data_dict['xyz_A'], atom_features_A], -1)
        neighA, ctrA, maskA = self.group_divider(
            ptsA, data_dict['residue_starts_A'], data_dict['atom_A_mask']
        )

        ptsB = torch.cat([data_dict['xyz_antibody'], atom_features_antibody], -1)
        neighB, ctrB, maskB = self.group_divider(
            ptsB, data_dict['residue_starts_antibody'], data_dict['atom_ab_mask']
        )
        B = maskA.shape[0]
        RA, RB = maskA.shape[1], maskB.shape[1]
        # Generate masks for antigen
        prob_mask_A = torch.rand(B, RA, device=maskA.device)
        to_mask_A = (prob_mask_A < self.mask_ratio) & maskA.bool()
        prob2_A = torch.rand(B, RA, device=maskA.device)
        mask_zero_A = to_mask_A & (prob2_A < self.mask_ratio_for_zero)
        mask_rand_A = to_mask_A & ~mask_zero_A
        
        # Generate masks for antibody
        prob_mask_B = torch.rand(B, RB, device=maskB.device)
        to_mask_B = (prob_mask_B < self.mask_ratio) & maskB.bool()
        prob2_B = torch.rand(B, RB, device=maskB.device)
        mask_zero_B = to_mask_B & (prob2_B < self.mask_ratio_for_zero)
        mask_rand_B = to_mask_B & ~mask_zero_B

        mask_zero_idx_A = mask_zero_A.nonzero(as_tuple=True)
        mask_rand_idx_A = mask_rand_A.nonzero(as_tuple=True)
       
        aa_corrupt_A = self.apply_masking_and_corruption(aaA, mask_zero_idx_A, mask_rand_idx_A, mask_rand_A, is_antigen=True, use_PLMs=self.use_PLMs, data_augmentation=self.data_augmentation)
        
        mask_zero_idx_B = mask_zero_B.nonzero(as_tuple=True)
        mask_rand_idx_B = mask_rand_B.nonzero(as_tuple=True)
        aa_corrupt_B = self.apply_masking_and_corruption(aaB, mask_zero_idx_B, mask_rand_idx_B, mask_rand_B, is_antigen=False, use_PLMs=self.use_PLMs, data_augmentation=self.data_augmentation)

        cent_corrupt_A = ctrA.clone()
        cent_corrupt_B = ctrB.clone()

        feat_A, pos_A = self.shared_encoder(neighA, cent_corrupt_A, aa_corrupt_A, maskA, is_antigen=True)
        feat_B, pos_B = self.shared_encoder(neighB, cent_corrupt_B, aa_corrupt_B, maskB, is_antigen=False)
        
        feat_cat = torch.cat([feat_A, feat_B], dim=1)
        pos_cat = torch.cat([pos_A, pos_B], dim=1)
        to_mask_cat = torch.cat([to_mask_A, to_mask_B], dim=1)
        
        rep = feat_cat + pos_cat
        rep_masked = rep[to_mask_cat]
        if self.use_PLMs:
            aa_feats_full_cat = torch.cat([aa_features_A_full[:,:,:-self.antigen_PLM_dim], aa_features_antibody_full[:, :,:-self.antibody_PLM_dim]], dim=1)
        else:
            aa_feats_full_cat = torch.cat([aa_features_A_full, aa_features_antibody_full], dim=1)
        gt_aa = aa_feats_full_cat[to_mask_cat][:, :20].argmax(-1)
        gt_solvent = aa_feats_full_cat[to_mask_cat][:, 40:42]
        gt_pssm = aa_feats_full_cat[to_mask_cat][:, -20:]

        p_aa      = self.aa_head(rep_masked)
        p_sol     = self.solvent_head(rep_masked)
        p_pssm    = self.pssm_head(rep_masked)

        L_aa  = self.ce(p_aa, gt_aa) * self.aa_weight
        L_sol = self.mse(p_sol, gt_solvent) * self.sol_weight
        L_pssm= self.mse(p_pssm, gt_pssm) * self.pssm_weight
        loss  = L_aa + L_sol + L_pssm

        return loss, L_aa, L_sol, L_pssm