from ..wan.models import ModelManager
from ..wan.models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..wan.models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..wan.models.wan_video_image_encoder import WanImageEncoder
from ..wan.models.wan_video_dit import sinusoidal_embedding_1d
from ..wan.schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline, TeaCache
from .base import modulate, rope_apply, CrossAttention, MLP, Head, WanModelStateDictConverter
from ..wan.prompters import WanPrompter
from ..wan.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..utils.builder import PIPELINES
from typing import Optional, Tuple
from einops import rearrange
from PIL import Image
from tqdm import tqdm
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, layer_idx: int = 0, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        result = flash_attn_interface.flash_attn_func(q, k, v)
        x = result[0] if isinstance(result, tuple) else result
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)        
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, layer_idx=0):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, layer_idx=layer_idx)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, layer_idx=0):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v, layer_idx)
        return self.o(x)



def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )     
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, freq_dim)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                cam_emb: torch.Tensor,
                context: torch.Tensor,
                frame_time_embedding: dict,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        for layer_idx, block in enumerate(self.blocks, 1):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            block,
                            x, context, cam_emb, frame_time_embedding, t_mod, freqs, layer_idx,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        block,
                        x, context, cam_emb, frame_time_embedding, t_mod, freqs, layer_idx,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb, frame_time_embedding, t_mod, freqs, layer_idx)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, freq_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        
        self.frame_time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.temporal_downsampler = TemporalDownsampler(dim=dim)
                
        # Camera encoding components
        self.cam_encoder = nn.Linear(12, dim)
        self.projector = nn.Linear(dim, dim)
        
        # Initialize camera encoder to zero and projector to identity
        self.cam_encoder.weight.data.zero_()
        self.cam_encoder.bias.data.zero_()
        self.projector.weight = nn.Parameter(torch.eye(dim))
        self.projector.bias = nn.Parameter(torch.zeros(dim))

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, cam_emb, frame_time_embedding, t_mod, freqs, layer_idx=0):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # encode time embedding
        src_time_embedding = frame_time_embedding['time_embedding_src']
        tgt_time_embedding = frame_time_embedding['time_embedding_tgt']

        src_time_embedding = self.frame_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, src_time_embedding))
        tgt_time_embedding = self.frame_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, tgt_time_embedding))

        # temporal downsampling: 81 -> 21 frames
        src_time_embedding = self.temporal_downsampler(src_time_embedding)
        tgt_time_embedding = self.temporal_downsampler(tgt_time_embedding)

        frame_time_embedding = torch.cat([tgt_time_embedding, src_time_embedding], dim=1)  # [B, 42, dim]
        frame_time_embedding = frame_time_embedding.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
        frame_time_embedding = rearrange(frame_time_embedding, 'b f h w d -> b (f h w) d')
        input_x = input_x + frame_time_embedding

        # encode camera
        cam_emb_tgt = self.cam_encoder(cam_emb["tgt"])
        cam_emb_src = self.cam_encoder(cam_emb["src"])
        cam_emb = torch.cat([cam_emb_tgt, cam_emb_src], dim=1)

        cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
        cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')

        input_x = input_x + cam_emb
        x = x + gate_msa * self.projector(self.self_attn(input_x, freqs, layer_idx))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution for temporal downsampling.
    Simple version without cache - just uses causal padding.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store original padding for causal behavior
        self._padding = self.padding[0] * 2  # Only pad on the left (past)
        self.padding = (0,)  # Remove default padding
    
    def forward(self, x):
        # Simple causal padding: pad only on the left (past frames)
        if self._padding > 0:
            x = F.pad(x, (self._padding, 0))  # Pad only on the left
        
        return super().forward(x)


class TemporalDownsampler(nn.Module):
    """
    Temporal downsampler following VAE chunking strategy exactly.
    Uses two-stage 2:1 compression like VAE encoder.
    81 frames -> 21 frames (matches VAE 4:1 compression via 2x 2:1 stages)
    """
    
    def __init__(self, dim=1536):
        super().__init__()
        self.dim = dim
        
        # Two-stage compression like VAE: 2:1 then 2:1 = 4:1 total
        # Stage 1: 4 frames -> 2 frames (2:1 compression)
        self.temporal_conv1 = CausalConv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(32, dim)
        self.activation1 = nn.SiLU()
        
        # Stage 2: 2 frames -> 1 frame (2:1 compression)  
        self.temporal_conv2 = CausalConv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(32, dim)
        self.activation2 = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, D] where T=81, D=1536
        
        Returns:
            Output tensor [B, T_out, D] where T_out=21, D=1536
        """
        B, T, D = x.shape
        
        # Follow VAE chunking strategy exactly
        # Calculate number of iterations: iter_ = 1 + (t - 1) // 4
        iter_ = 1 + (T - 1) // 4  # For T=81: iter_ = 1 + 80//4 = 21
        
        outputs = []
        
        for i in range(iter_):
            if i == 0:
                # First chunk: only first frame
                chunk = x[:, :1, :]  # [B, 1, D]
            else:
                # Subsequent chunks: 4 frames each
                start_idx = 1 + 4 * (i - 1)
                end_idx = min(1 + 4 * i, T)
                chunk = x[:, start_idx:end_idx, :]  # [B, 4, D] (or less for last chunk)
            
            # Process chunk using two-stage compression like VAE
            if chunk.shape[1] == 1:
                # First frame - keep as is (no compression needed)
                processed_chunk = chunk  # [B, 1, D]
            else:
                # For 4-frame chunks, compress using two-stage approach: 4→2→1
                chunk_transposed = chunk.transpose(1, 2)  # [B, D, T_chunk]
                
                # Pad to ensure consistent processing (ensure 4 frames)
                if chunk_transposed.shape[2] < 4:
                    pad_size = 4 - chunk_transposed.shape[2]
                    chunk_transposed = F.pad(chunk_transposed, (0, pad_size), mode='replicate')
                
                # Stage 1: 4 frames -> 2 frames (2:1 compression)
                stage1 = self.temporal_conv1(chunk_transposed)  # [B, D, 2]
                stage1 = self.norm1(stage1)
                stage1 = self.activation1(stage1)
                
                # Stage 2: 2 frames -> 1 frame (2:1 compression)
                stage2 = self.temporal_conv2(stage1)  # [B, D, 1]
                stage2 = self.norm2(stage2)
                stage2 = self.activation2(stage2)
                
                processed_chunk = stage2.transpose(1, 2)  # [B, 1, D]
            
            outputs.append(processed_chunk)
        
        # Concatenate all processed chunks
        result = torch.cat(outputs, dim=1)  # [B, 21, D]
        
        return result


@PIPELINES.register_module(name='spacetimepilot_1dconv')
class WanVideoReCamMasterPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        # Get the pre-trained model (just for configuration/weights)
        pretrained_dit = model_manager.fetch_model("wan_video_dit")
        
        if pretrained_dit is not None:

            config = {
                "dim": pretrained_dit.dim,
                "in_dim": pretrained_dit.patch_embedding.in_channels,
                "ffn_dim": pretrained_dit.blocks[0].ffn_dim,
                "out_dim": pretrained_dit.head.head.out_features // math.prod(pretrained_dit.patch_size),
                "text_dim": pretrained_dit.text_embedding[0].in_features,
                "freq_dim": pretrained_dit.freq_dim,
                "eps": pretrained_dit.blocks[0].norm1.eps,
                "patch_size": pretrained_dit.patch_size,
                "num_heads": pretrained_dit.blocks[0].num_heads,
                "num_layers": len(pretrained_dit.blocks),
                "has_image_input": pretrained_dit.has_image_input,
            }

            self.dit = WanModel(**config)

            state_dict = pretrained_dit.state_dict()
            missing_keys, unexpected_keys = self.dit.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoReCamMasterPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}


    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        source_video=None,
        target_camera=None,
        source_camera=None,
        src_time_embedding=None,
        tgt_time_embedding=None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        inference_t2v=False,
        inference_i2v=False,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode source video (recammaster)
        self.load_models_to_device(['vae'])
        source_video = source_video.to(dtype=self.torch_dtype, device=self.device)
        source_latents = self.encode_video(source_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        
        # Handle I2V/T2V inference modes
        if inference_t2v:
            source_latents = torch.randn_like(source_latents)
        elif inference_i2v:
            first_frame = source_latents[:, :, 0, :, :]  # [1, 16, 1, H, W] - extract first frame
            source_latents = torch.randn_like(source_latents)  # Replace all with noise
            source_latents[:, :, 0, :, :] = first_frame  # Restore first frame
        
        # Process target camera (recammaster)
        tgt_cam_emb = target_camera.to(dtype=self.torch_dtype, device=self.device)
        src_cam_emb = source_camera.to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        
        src_time_emb = src_time_embedding.to(dtype=self.torch_dtype, device=self.device)
        tgt_time_emb = tgt_time_embedding.to(dtype=self.torch_dtype, device=self.device)
        
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(["dit"])
        tgt_latent_length = latents.shape[2]
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            latents_input = torch.cat([latents, source_latents], dim=2)
            # Inference
            noise_pred_posi = model_fn_wan_video(self.dit, latents_input, timestep=timestep, 
                                                src_camera_emb=src_cam_emb, tgt_camera_emb=tgt_cam_emb,
                                                src_time_embedding=src_time_emb, tgt_time_embedding=tgt_time_emb,
                                                **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi)
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(self.dit, latents_input, timestep=timestep,
                                                    src_camera_emb=src_cam_emb, tgt_camera_emb=tgt_cam_emb,
                                                    src_time_embedding=src_time_emb, tgt_time_embedding=tgt_time_emb,
                                                    **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred[:,:,:tgt_latent_length,...], self.scheduler.timesteps[progress_id], latents_input[:,:,:tgt_latent_length,...])

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames



def model_fn_wan_video(
        dit,
        x: torch.Tensor,
        timestep: torch.Tensor,
        src_camera_emb: torch.Tensor,
        tgt_camera_emb: torch.Tensor,
        src_time_embedding: torch.Tensor,
        tgt_time_embedding: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        tea_cache: TeaCache = None,
        **kwargs,
    ):
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1)

    freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

    frame_time_embedding = {
        "time_embedding_src": src_time_embedding,
        "time_embedding_tgt": tgt_time_embedding,
    }
    camera_emb = {"tgt": tgt_camera_emb, "src": src_camera_emb}

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for layer_idx, block in enumerate(dit.blocks, 1):
            x = block(x, context, camera_emb, frame_time_embedding, t_mod, freqs, layer_idx)
        if tea_cache is not None:
            tea_cache.store(x)
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
