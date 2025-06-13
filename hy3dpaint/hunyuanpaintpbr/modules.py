# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Literal
import diffusers
from diffusers.utils import deprecate
from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.models import UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.transformers.transformer_2d import BasicTransformerBlock

from transformers import AutoImageProcessor, AutoModel




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, Literal, List, Callable
from einops import rearrange
from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor


class AttnUtils:
    """
    Shared utility functions for attention processing.

    This class provides common operations used across different attention processors
    to eliminate code duplication and improve maintainability.
    """

    @staticmethod
    def check_pytorch_compatibility():
        """
        Check PyTorch compatibility for scaled_dot_product_attention.

        Raises:
            ImportError: If PyTorch version doesn't support scaled_dot_product_attention
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    @staticmethod
    def handle_deprecation_warning(args, kwargs):
        """
        Handle deprecation warning for the 'scale' argument.

        Args:
            args: Positional arguments passed to attention processor
            kwargs: Keyword arguments passed to attention processor
        """
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored."
                "Please remove it, as passing it will raise an error in the future."
                "`scale` should directly be passed while calling the underlying pipeline component"
                "i.e., via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

    @staticmethod
    def prepare_hidden_states(
        hidden_states, attn, temb, spatial_norm_attr="spatial_norm", group_norm_attr="group_norm"
    ):
        """
        Common preprocessing of hidden states for attention computation.

        Args:
            hidden_states: Input hidden states tensor
            attn: Attention module instance
            temb: Optional temporal embedding tensor
            spatial_norm_attr: Attribute name for spatial normalization
            group_norm_attr: Attribute name for group normalization

        Returns:
            Tuple of (processed_hidden_states, residual, input_ndim, shape_info)
        """
        residual = hidden_states

        spatial_norm = getattr(attn, spatial_norm_attr, None)
        if spatial_norm is not None:
            hidden_states = spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        group_norm = getattr(attn, group_norm_attr, None)
        if group_norm is not None:
            hidden_states = group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        return hidden_states, residual, input_ndim, (batch_size, channel, height, width)

    @staticmethod
    def prepare_attention_mask(attention_mask, attn, sequence_length, batch_size):
        """
        Prepare attention mask for scaled_dot_product_attention.

        Args:
            attention_mask: Input attention mask tensor or None
            attn: Attention module instance
            sequence_length: Length of the sequence
            batch_size: Batch size

        Returns:
            Prepared attention mask tensor reshaped for multi-head attention
        """
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        return attention_mask

    @staticmethod
    def reshape_qkv_for_attention(tensor, batch_size, attn_heads, head_dim):
        """
        Reshape Q/K/V tensors for multi-head attention computation.

        Args:
            tensor: Input tensor to reshape
            batch_size: Batch size
            attn_heads: Number of attention heads
            head_dim: Dimension per attention head

        Returns:
            Reshaped tensor with shape [batch_size, attn_heads, seq_len, head_dim]
        """
        return tensor.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

    @staticmethod
    def apply_norms(query, key, norm_q, norm_k):
        """
        Apply Q/K normalization layers if available.

        Args:
            query: Query tensor
            key: Key tensor
            norm_q: Query normalization layer (optional)
            norm_k: Key normalization layer (optional)

        Returns:
            Tuple of (normalized_query, normalized_key)
        """
        if norm_q is not None:
            query = norm_q(query)
        if norm_k is not None:
            key = norm_k(key)
        return query, key

    @staticmethod
    def finalize_output(hidden_states, input_ndim, shape_info, attn, residual, to_out):
        """
        Common output processing including projection, dropout, reshaping, and residual connection.

        Args:
            hidden_states: Processed hidden states from attention
            input_ndim: Original input tensor dimensions
            shape_info: Tuple containing original shape information
            attn: Attention module instance
            residual: Residual connection tensor
            to_out: Output projection layers [linear, dropout]

        Returns:
            Final output tensor after all processing steps
        """
        batch_size, channel, height, width = shape_info

        # Apply output projection and dropout
        hidden_states = to_out[0](hidden_states)
        hidden_states = to_out[1](hidden_states)

        # Reshape back if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Apply residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Apply rescaling
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Base class for attention processors (eliminating initialization duplication)
class BaseAttnProcessor(nn.Module):
    """
    Base class for attention processors with common initialization.

    This base class provides shared parameter initialization and module registration
    functionality to reduce code duplication across different attention processor types.
    """

    def __init__(
        self,
        query_dim: int,
        pbr_setting: List[str] = ["albedo", "mr"],
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        **kwargs,
    ):
        """
        Initialize base attention processor with common parameters.

        Args:
            query_dim: Dimension of query features
            pbr_setting: List of PBR material types to process (e.g., ["albedo", "mr"])
            cross_attention_dim: Dimension of cross-attention features (optional)
            heads: Number of attention heads
            kv_heads: Number of key-value heads for grouped query attention (optional)
            dim_head: Dimension per attention head
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
            upcast_attention: Whether to upcast attention computation to float32
            upcast_softmax: Whether to upcast softmax computation to float32
            cross_attention_norm: Type of cross-attention normalization (optional)
            cross_attention_norm_num_groups: Number of groups for cross-attention norm
            qk_norm: Type of query-key normalization (optional)
            added_kv_proj_dim: Dimension for additional key-value projections (optional)
            added_proj_bias: Whether to use bias in additional projections
            norm_num_groups: Number of groups for normalization (optional)
            spatial_norm_dim: Dimension for spatial normalization (optional)
            out_bias: Whether to use bias in output projection
            scale_qk: Whether to scale query-key products
            only_cross_attention: Whether to only perform cross-attention
            eps: Small epsilon value for numerical stability
            rescale_output_factor: Factor to rescale output values
            residual_connection: Whether to use residual connections
            _from_deprecated_attn_block: Flag for deprecated attention blocks
            processor: Optional attention processor instance
            out_dim: Output dimension (optional)
            out_context_dim: Output context dimension (optional)
            context_pre_only: Whether to only process context in pre-processing
            pre_only: Whether to only perform pre-processing
            elementwise_affine: Whether to use element-wise affine transformations
            is_causal: Whether to use causal attention masking
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        AttnUtils.check_pytorch_compatibility()

        # Store common attributes
        self.pbr_setting = pbr_setting
        self.n_pbr_tokens = len(self.pbr_setting)
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention
        self.added_proj_bias = added_proj_bias

        # Validation
        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
                "Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

    def register_pbr_modules(self, module_types: List[str], **kwargs):
        """
        Generic PBR module registration to eliminate code repetition.

        Dynamically registers PyTorch modules for different PBR material types
        based on the specified module types and PBR settings.

        Args:
            module_types: List of module types to register ("qkv", "v_only", "out", "add_kv")
            **kwargs: Additional arguments for module configuration
        """
        for pbr_token in self.pbr_setting:
            if pbr_token == "albedo":
                continue

            for module_type in module_types:
                if module_type == "qkv":
                    self.register_module(
                        f"to_q_{pbr_token}", nn.Linear(self.query_dim, self.inner_dim, bias=self.use_bias)
                    )
                    self.register_module(
                        f"to_k_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                    self.register_module(
                        f"to_v_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                elif module_type == "v_only":
                    self.register_module(
                        f"to_v_{pbr_token}", nn.Linear(self.cross_attention_dim, self.inner_dim, bias=self.use_bias)
                    )
                elif module_type == "out":
                    if not self.pre_only:
                        self.register_module(
                            f"to_out_{pbr_token}",
                            nn.ModuleList(
                                [
                                    nn.Linear(self.inner_dim, self.out_dim, bias=kwargs.get("out_bias", True)),
                                    nn.Dropout(self.dropout),
                                ]
                            ),
                        )
                    else:
                        self.register_module(f"to_out_{pbr_token}", None)
                elif module_type == "add_kv":
                    if self.added_kv_proj_dim is not None:
                        self.register_module(
                            f"add_k_proj_{pbr_token}",
                            nn.Linear(self.added_kv_proj_dim, self.inner_kv_dim, bias=self.added_proj_bias),
                        )
                        self.register_module(
                            f"add_v_proj_{pbr_token}",
                            nn.Linear(self.added_kv_proj_dim, self.inner_kv_dim, bias=self.added_proj_bias),
                        )
                    else:
                        self.register_module(f"add_k_proj_{pbr_token}", None)
                        self.register_module(f"add_v_proj_{pbr_token}", None)


# Rotary Position Embedding utilities (specialized for PoseRoPE)
class RotaryEmbedding:
    """
    Rotary position embedding utilities for 3D spatial attention.

    Provides functions to compute and apply rotary position embeddings (RoPE)
    for 1D, 3D spatial coordinates used in 3D-aware attention mechanisms.
    """

    @staticmethod
    def get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 10000.0, linear_factor=1.0, ntk_factor=1.0):
        """
        Compute 1D rotary position embeddings.

        Args:
            dim: Embedding dimension (must be even)
            pos: Position tensor
            theta: Base frequency for rotary embeddings
            linear_factor: Linear scaling factor
            ntk_factor: NTK (Neural Tangent Kernel) scaling factor

        Returns:
            Tuple of (cos_embeddings, sin_embeddings)
        """
        assert dim % 2 == 0
        theta = theta * ntk_factor
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device)[: (dim // 2)] / dim))
            / linear_factor
        )
        freqs = torch.outer(pos, freqs)
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
        return freqs_cos, freqs_sin

    @staticmethod
    def get_3d_rotary_pos_embed(position, embed_dim, voxel_resolution, theta: int = 10000):
        """
        Compute 3D rotary position embeddings for spatial coordinates.

        Args:
            position: 3D position tensor with shape [..., 3]
            embed_dim: Embedding dimension
            voxel_resolution: Resolution of the voxel grid
            theta: Base frequency for rotary embeddings

        Returns:
            Tuple of (cos_embeddings, sin_embeddings) for 3D positions
        """
        assert position.shape[-1] == 3
        dim_xy = embed_dim // 8 * 3
        dim_z = embed_dim // 8 * 2

        grid = torch.arange(voxel_resolution, dtype=torch.float32, device=position.device)
        freqs_xy = RotaryEmbedding.get_1d_rotary_pos_embed(dim_xy, grid, theta=theta)
        freqs_z = RotaryEmbedding.get_1d_rotary_pos_embed(dim_z, grid, theta=theta)

        xy_cos, xy_sin = freqs_xy
        z_cos, z_sin = freqs_z

        embed_flattn = position.view(-1, position.shape[-1])
        x_cos = xy_cos[embed_flattn[:, 0], :]
        x_sin = xy_sin[embed_flattn[:, 0], :]
        y_cos = xy_cos[embed_flattn[:, 1], :]
        y_sin = xy_sin[embed_flattn[:, 1], :]
        z_cos = z_cos[embed_flattn[:, 2], :]
        z_sin = z_sin[embed_flattn[:, 2], :]

        cos = torch.cat((x_cos, y_cos, z_cos), dim=-1)
        sin = torch.cat((x_sin, y_sin, z_sin), dim=-1)

        cos = cos.view(*position.shape[:-1], embed_dim)
        sin = sin.view(*position.shape[:-1], embed_dim)
        return cos, sin

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]):
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor to apply rotary embeddings to
            freqs_cis: Tuple of (cos_embeddings, sin_embeddings) or single tensor

        Returns:
            Tensor with rotary position embeddings applied
        """
        cos, sin = freqs_cis
        cos, sin = cos.to(x.device), sin.to(x.device)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out


# Core attention processing logic (eliminating major duplication)
class AttnCore:
    """
    Core attention processing logic shared across processors.

    This class provides the fundamental attention computation pipeline
    that can be reused across different attention processor implementations.
    """

    @staticmethod
    def process_attention_base(
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        get_qkv_fn: Callable = None,
        apply_rope_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Generic attention processing core shared across different processors.

        This function implements the common attention computation pipeline including:
        1. Hidden state preprocessing
        2. Attention mask preparation
        3. Q/K/V computation via provided function
        4. Tensor reshaping for multi-head attention
        5. Optional normalization and RoPE application
        6. Scaled dot-product attention computation

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            get_qkv_fn: Function to compute Q, K, V tensors
            apply_rope_fn: Optional function to apply rotary position embeddings
            **kwargs: Additional keyword arguments passed to subfunctions

        Returns:
            Tuple containing (attention_output, residual, input_ndim, shape_info,
            batch_size, num_heads, head_dim)
        """
        # Prepare hidden states
        hidden_states, residual, input_ndim, shape_info = AttnUtils.prepare_hidden_states(hidden_states, attn, temb)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # Prepare attention mask
        attention_mask = AttnUtils.prepare_attention_mask(attention_mask, attn, sequence_length, batch_size)

        # Get Q, K, V
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query, key, value = get_qkv_fn(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = AttnUtils.reshape_qkv_for_attention(query, batch_size, attn.heads, head_dim)
        key = AttnUtils.reshape_qkv_for_attention(key, batch_size, attn.heads, head_dim)
        value = AttnUtils.reshape_qkv_for_attention(value, batch_size, attn.heads, value.shape[-1] // attn.heads)

        # Apply normalization
        query, key = AttnUtils.apply_norms(query, key, getattr(attn, "norm_q", None), getattr(attn, "norm_k", None))

        # Apply RoPE if provided
        if apply_rope_fn is not None:
            query, key = apply_rope_fn(query, key, head_dim, **kwargs)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        return hidden_states, residual, input_ndim, shape_info, batch_size, attn.heads, head_dim


# Specific processor implementations (minimal unique code)
class PoseRoPEAttnProcessor2_0:
    """
    Attention processor with Rotary Position Encoding (RoPE) for 3D spatial awareness.

    This processor extends standard attention with 3D rotary position embeddings
    to provide spatial awareness for 3D scene understanding tasks.
    """

    def __init__(self):
        """Initialize the RoPE attention processor."""
        AttnUtils.check_pytorch_compatibility()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_indices: Dict = None,
        temb: Optional[torch.Tensor] = None,
        n_pbrs=1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply RoPE-enhanced attention computation.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            position_indices: Dictionary containing 3D position information for RoPE
            temb: Optional temporal embedding tensor
            n_pbrs: Number of PBR material types
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Attention output tensor with applied rotary position encodings
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            return attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        def apply_rope(query, key, head_dim, **kwargs):
            if position_indices is not None:
                if head_dim in position_indices:
                    image_rotary_emb = position_indices[head_dim]
                else:
                    image_rotary_emb = RotaryEmbedding.get_3d_rotary_pos_embed(
                        rearrange(
                            position_indices["voxel_indices"].unsqueeze(1).repeat(1, n_pbrs, 1, 1),
                            "b n_pbrs l c -> (b n_pbrs) l c",
                        ),
                        head_dim,
                        voxel_resolution=position_indices["voxel_resolution"],
                    )
                    position_indices[head_dim] = image_rotary_emb

                query = RotaryEmbedding.apply_rotary_emb(query, image_rotary_emb)
                key = RotaryEmbedding.apply_rotary_emb(key, image_rotary_emb)
            return query, key

        # Core attention processing
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            get_qkv_fn=get_qkv,
            apply_rope_fn=apply_rope,
            position_indices=position_indices,
            n_pbrs=n_pbrs,
        )

        # Finalize output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(hidden_states.dtype)

        return AttnUtils.finalize_output(hidden_states, input_ndim, shape_info, attn, residual, attn.to_out)


class SelfAttnProcessor2_0(BaseAttnProcessor):
    """
    Self-attention processor with PBR (Physically Based Rendering) material support.

    This processor handles multiple PBR material types (e.g., albedo, metallic-roughness)
    with separate attention computation paths for each material type.
    """

    def __init__(self, **kwargs):
        """
        Initialize self-attention processor with PBR support.

        Args:
            **kwargs: Arguments passed to BaseAttnProcessor initialization
        """
        super().__init__(**kwargs)
        self.register_pbr_modules(["qkv", "out", "add_kv"], **kwargs)

    def process_single(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        token: Literal["albedo", "mr"] = "albedo",
        multiple_devices=False,
        *args,
        **kwargs,
    ):
        """
        Process attention for a single PBR material type.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            token: PBR material type to process ("albedo", "mr", etc.)
            multiple_devices: Whether to use multiple GPU devices
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed attention output for the specified PBR material type
        """
        target = attn if token == "albedo" else attn.processor
        token_suffix = "" if token == "albedo" else "_" + token

        # Device management (if needed)
        if multiple_devices:
            device = torch.device("cuda:0") if token == "albedo" else torch.device("cuda:1")
            for attr in [f"to_q{token_suffix}", f"to_k{token_suffix}", f"to_v{token_suffix}", f"to_out{token_suffix}"]:
                getattr(target, attr).to(device)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            return (
                getattr(target, f"to_q{token_suffix}")(hidden_states),
                getattr(target, f"to_k{token_suffix}")(encoder_hidden_states),
                getattr(target, f"to_v{token_suffix}")(encoder_hidden_states),
            )

        # Core processing using shared logic
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, get_qkv_fn=get_qkv
        )

        # Finalize
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = hidden_states.to(hidden_states.dtype)

        return AttnUtils.finalize_output(
            hidden_states, input_ndim, shape_info, attn, residual, getattr(target, f"to_out{token_suffix}")
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply self-attention with PBR material processing.

        Processes multiple PBR material types sequentially, applying attention
        computation for each material type separately and combining results.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor with PBR dimension
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Combined attention output for all PBR material types
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        B = hidden_states.size(0)
        pbr_hidden_states = torch.split(hidden_states, 1, dim=1)

        # Process each PBR setting
        results = []
        for token, pbr_hs in zip(self.pbr_setting, pbr_hidden_states):
            processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c").to("cuda:0")
            result = self.process_single(attn, processed_hs, None, attention_mask, temb, token, False)
            results.append(result)

        outputs = [rearrange(result, "(b n_pbrs n) l c -> b n_pbrs n l c", b=B, n_pbrs=1) for result in results]
        return torch.cat(outputs, dim=1)


class RefAttnProcessor2_0(BaseAttnProcessor):
    """
    Reference attention processor with shared value computation across PBR materials.

    This processor computes query and key once, but uses separate value projections
    for different PBR material types, enabling efficient multi-material processing.
    """

    def __init__(self, **kwargs):
        """
        Initialize reference attention processor.

        Args:
            **kwargs: Arguments passed to BaseAttnProcessor initialization
        """
        super().__init__(**kwargs)
        self.pbr_settings = self.pbr_setting  # Alias for compatibility
        self.register_pbr_modules(["v_only", "out"], **kwargs)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply reference attention with shared Q/K and separate V projections.

        This method computes query and key tensors once and reuses them across
        all PBR material types, while using separate value projections for each
        material type to maintain material-specific information.

        Args:
            attn: Attention module instance
            hidden_states: Input hidden states tensor
            encoder_hidden_states: Optional encoder hidden states for cross-attention
            attention_mask: Optional attention mask tensor
            temb: Optional temporal embedding tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Stacked attention output for all PBR material types
        """
        AttnUtils.handle_deprecation_warning(args, kwargs)

        def get_qkv(attn, hidden_states, encoder_hidden_states, **kwargs):
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            # Concatenate values from all PBR settings
            value_list = [attn.to_v(encoder_hidden_states)]
            for token in ["_" + token for token in self.pbr_settings if token != "albedo"]:
                value_list.append(getattr(attn.processor, f"to_v{token}")(encoder_hidden_states))
            value = torch.cat(value_list, dim=-1)

            return query, key, value

        # Core processing
        hidden_states, residual, input_ndim, shape_info, batch_size, heads, head_dim = AttnCore.process_attention_base(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, get_qkv_fn=get_qkv
        )

        # Split and process each PBR setting output
        hidden_states_list = torch.split(hidden_states, head_dim, dim=-1)
        output_hidden_states_list = []

        for i, hs in enumerate(hidden_states_list):
            hs = hs.transpose(1, 2).reshape(batch_size, -1, heads * head_dim).to(hs.dtype)
            token_suffix = "_" + self.pbr_settings[i] if self.pbr_settings[i] != "albedo" else ""
            target = attn if self.pbr_settings[i] == "albedo" else attn.processor

            hs = AttnUtils.finalize_output(
                hs, input_ndim, shape_info, attn, residual, getattr(target, f"to_out{token_suffix}")
            )
            output_hidden_states_list.append(hs)

        return torch.stack(output_hidden_states_list, dim=1)


class Dino_v2(nn.Module):

    """Wrapper for DINOv2 vision transformer (frozen weights).
    
    Provides feature extraction for reference images.
    
    Args:
        dino_v2_path: Custom path to DINOv2 model weights (uses default if None)
    """


    def __init__(self, dino_v2_path):
        super(Dino_v2, self).__init__()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_v2_path)
        self.dino_v2 = AutoModel.from_pretrained(dino_v2_path)

        for param in self.parameters():
            param.requires_grad = False

        self.dino_v2.eval()

    def forward(self, images):

        """Processes input images through DINOv2 ViT.
        
        Handles both tensor input (B, N, C, H, W) and PIL image lists.
        Extracts patch embeddings and flattens spatial dimensions.
        
        Returns:
            torch.Tensor: Feature vectors [B, N*(num_patches), feature_dim]
        """

        if isinstance(images, torch.Tensor):
            batch_size = images.shape[0]
            dino_proceesed_images = self.dino_processor(
                images=rearrange(images, "b n c h w -> (b n) c h w"), return_tensors="pt", do_rescale=False
            ).pixel_values
        else:
            batch_size = 1
            dino_proceesed_images = self.dino_processor(images=images, return_tensors="pt").pixel_values
            dino_proceesed_images = torch.stack(
                [torch.from_numpy(np.array(image)) for image in dino_proceesed_images], dim=0
            )
        dino_param = next(self.dino_v2.parameters())
        dino_proceesed_images = dino_proceesed_images.to(dino_param)
        dino_hidden_states = self.dino_v2(dino_proceesed_images)[0]
        dino_hidden_states = rearrange(dino_hidden_states.to(dino_param), "(b n) l c -> b (n l) c", b=batch_size)

        return dino_hidden_states


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory

    """Memory-efficient feedforward execution via chunking.
    
    Divides input along specified dimension for sequential processing.
    
    Args:
        ff: Feedforward module to apply
        hidden_states: Input tensor
        chunk_dim: Dimension to split
        chunk_size: Size of each chunk
        
    Returns:
        torch.Tensor: Reassembled output tensor
    """

    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]}"
            f"has to be divisible by chunk size: {chunk_size}."
            "Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


@torch.no_grad()
def compute_voxel_grid_mask(position, grid_resolution=8):

    """Generates view-to-view attention mask based on 3D position similarity.
    
    Uses voxel grid downsampling to determine spatially adjacent regions.
    Mask indicates where features should interact across different views.
    
    Args:
        position: Position maps [B, N, 3, H, W] (normalized 0-1)
        grid_resolution: Spatial reduction factor
        
    Returns:
        torch.Tensor: Attention mask [B, N*grid_res**2, N*grid_res**2]
    """

    position = position.half()
    B, N, _, H, W = position.shape
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask == False] = 0

    position = rearrange(
        position,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )
    valid_mask = rearrange(
        valid_mask,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked < 5] = 0

    grid_position = grid_position.permute(0, 1, 4, 2, 3)
    grid_position = rearrange(grid_position, "b n c h w -> b n (h w) c")

    grid_position_expanded_1 = grid_position.unsqueeze(2).unsqueeze(4)  # 形状变为 B, N, 1, L, 1, 3
    grid_position_expanded_2 = grid_position.unsqueeze(1).unsqueeze(3)  # 形状变为 B, 1, N, 1, L, 3

    # 计算欧氏距离
    distances = torch.norm(grid_position_expanded_1 - grid_position_expanded_2, dim=-1)  # 形状为 B, N, N, L, L

    weights = distances
    grid_distance = 1.73 / grid_resolution
    weights = weights < grid_distance

    return weights


def compute_multi_resolution_mask(position_maps, grid_resolutions=[32, 16, 8]):

    """Generates attention masks at multiple spatial resolutions.
    
    Creates pyramid of position-based masks for hierarchical attention.
    
    Args:
        position_maps: Position maps [B, N, 3, H, W]
        grid_resolutions: List of downsampling factors
        
    Returns:
        dict: Resolution-specific masks keyed by flattened dimension size
    """

    position_attn_mask = {}
    with torch.no_grad():
        for grid_resolution in grid_resolutions:
            position_mask = compute_voxel_grid_mask(position_maps, grid_resolution)
            position_mask = rearrange(position_mask, "b ni nj li lj -> b (ni li) (nj lj)")
            position_attn_mask[position_mask.shape[1]] = position_mask
    return position_attn_mask


@torch.no_grad()
def compute_discrete_voxel_indice(position, grid_resolution=8, voxel_resolution=128):

    """Quantizes position maps to discrete voxel indices.
    
    Creates sparse 3D coordinate representations for efficient hashing.
    
    Args:
        position: Position maps [B, N, 3, H, W]
        grid_resolution: Spatial downsampling factor
        voxel_resolution: Quantization resolution
        
    Returns:
        torch.Tensor: Voxel indices [B, N, grid_res, grid_res, 3]
    """

    position = position.half()
    B, N, _, H, W = position.shape
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask == False] = 0

    position = rearrange(
        position,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )
    valid_mask = rearrange(
        valid_mask,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    voxel_mask_thres = (H // grid_resolution) * (W // grid_resolution) // (4 * 4)
    grid_position[count_masked < voxel_mask_thres] = 0

    grid_position = grid_position.permute(0, 1, 4, 2, 3).clamp(0, 1)  # B N C H W
    voxel_indices = grid_position * (voxel_resolution - 1)
    voxel_indices = torch.round(voxel_indices).long()
    return voxel_indices


def calc_multires_voxel_idxs(position_maps, grid_resolutions=[64, 32, 16, 8], voxel_resolutions=[512, 256, 128, 64]):

    """Generates multi-resolution voxel indices for position encoding.
    
    Creates pyramid of quantized position representations.
    
    Args:
        position_maps: Input position maps
        grid_resolutions: Spatial resolution levels
        voxel_resolutions: Quantization levels
        
    Returns:
        dict: Voxel indices keyed by flattened dimension size, with resolution metadata
    """

    voxel_indices = {}
    with torch.no_grad():
        for grid_resolution, voxel_resolution in zip(grid_resolutions, voxel_resolutions):
            voxel_indice = compute_discrete_voxel_indice(position_maps, grid_resolution, voxel_resolution)
            voxel_indice = rearrange(voxel_indice, "b n c h w -> b (n h w) c")
            voxel_indices[voxel_indice.shape[1]] = {"voxel_indices": voxel_indice, "voxel_resolution": voxel_resolution}
    return voxel_indices


class Basic2p5DTransformerBlock(torch.nn.Module):


    """Enhanced transformer block for multiview 2.5D image generation.
    
    Extends standard transformer blocks with:
    - Material-specific attention (MDA)
    - Multiview attention (MA)
    - Reference attention (RA)
    - DINO feature integration
    
    Args:
        transformer: Base transformer block
        layer_name: Identifier for layer
        use_ma: Enable multiview attention
        use_ra: Enable reference attention
        use_mda: Enable material-aware attention
        use_dino: Enable DINO feature integration
        pbr_setting: List of PBR materials
    """

    def __init__(
        self,
        transformer: BasicTransformerBlock,
        layer_name,
        use_ma=True,
        use_ra=True,
        use_mda=True,
        use_dino=True,
        pbr_setting=None,
    ) -> None:
        
        """
        Initialization:
        1. Material-Dimension Attention (MDA):
           - Processes each PBR material with separate projection weights
           - Uses custom SelfAttnProcessor2_0 with material awareness
           
        2. Multiview Attention (MA):
           - Adds cross-view attention with PoseRoPE
           - Initialized as zero-initialized residual pathway
           
        3. Reference Attention (RA):
           - Conditions on reference view features
           - Uses RefAttnProcessor2_0 for material-specific conditioning
           
        4. DINO Attention:
           - Incorporates DINO-ViT features
           - Initialized as zero-initialized residual pathway
        """

        super().__init__()
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.use_mda = use_mda
        self.use_dino = use_dino
        self.pbr_setting = pbr_setting

        if self.use_mda:
            self.attn1.set_processor(
                SelfAttnProcessor2_0(
                    query_dim=self.dim,
                    heads=self.num_attention_heads,
                    dim_head=self.attention_head_dim,
                    dropout=self.dropout,
                    bias=self.attention_bias,
                    cross_attention_dim=None,
                    upcast_attention=self.attn1.upcast_attention,
                    out_bias=True,
                    pbr_setting=self.pbr_setting,
                )
            )

        # multiview attn
        if self.use_ma:
            self.attn_multiview = Attention(
                query_dim=self.dim,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                dropout=self.dropout,
                bias=self.attention_bias,
                cross_attention_dim=None,
                upcast_attention=self.attn1.upcast_attention,
                out_bias=True,
                processor=PoseRoPEAttnProcessor2_0(),
            )

        # ref attn
        if self.use_ra:
            self.attn_refview = Attention(
                query_dim=self.dim,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                dropout=self.dropout,
                bias=self.attention_bias,
                cross_attention_dim=None,
                upcast_attention=self.attn1.upcast_attention,
                out_bias=True,
                processor=RefAttnProcessor2_0(
                    query_dim=self.dim,
                    heads=self.num_attention_heads,
                    dim_head=self.attention_head_dim,
                    dropout=self.dropout,
                    bias=self.attention_bias,
                    cross_attention_dim=None,
                    upcast_attention=self.attn1.upcast_attention,
                    out_bias=True,
                    pbr_setting=self.pbr_setting,
                ),
            )

        # dino attn
        if self.use_dino:
            self.attn_dino = Attention(
                query_dim=self.dim,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                dropout=self.dropout,
                bias=self.attention_bias,
                cross_attention_dim=self.cross_attention_dim,
                upcast_attention=self.attn2.upcast_attention,
                out_bias=True,
            )

        self._initialize_attn_weights()

    def _initialize_attn_weights(self):

        """Initializes specialized attention heads with base weights.
        
        Uses weight sharing strategy:
        - Copies base transformer weights to specialized heads
        - Initializes newly-added parameters to zero
        """

        if self.use_mda:
            for token in self.pbr_setting:
                if token == "albedo":
                    continue
                getattr(self.attn1.processor, f"to_q_{token}").load_state_dict(self.attn1.to_q.state_dict())
                getattr(self.attn1.processor, f"to_k_{token}").load_state_dict(self.attn1.to_k.state_dict())
                getattr(self.attn1.processor, f"to_v_{token}").load_state_dict(self.attn1.to_v.state_dict())
                getattr(self.attn1.processor, f"to_out_{token}").load_state_dict(self.attn1.to_out.state_dict())

        if self.use_ma:
            self.attn_multiview.load_state_dict(self.attn1.state_dict(), strict=False)
            with torch.no_grad():
                for layer in self.attn_multiview.to_out:
                    for param in layer.parameters():
                        param.zero_()

        if self.use_ra:
            self.attn_refview.load_state_dict(self.attn1.state_dict(), strict=False)
            for token in self.pbr_setting:
                if token == "albedo":
                    continue
                getattr(self.attn_refview.processor, f"to_v_{token}").load_state_dict(
                    self.attn_refview.to_q.state_dict()
                )
                getattr(self.attn_refview.processor, f"to_out_{token}").load_state_dict(
                    self.attn_refview.to_out.state_dict()
                )
            with torch.no_grad():
                for layer in self.attn_refview.to_out:
                    for param in layer.parameters():
                        param.zero_()
                for token in self.pbr_setting:
                    if token == "albedo":
                        continue
                    for layer in getattr(self.attn_refview.processor, f"to_out_{token}"):
                        for param in layer.parameters():
                            param.zero_()

        if self.use_dino:
            self.attn_dino.load_state_dict(self.attn2.state_dict(), strict=False)
            with torch.no_grad():
                for layer in self.attn_dino.to_out:
                    for param in layer.parameters():
                        param.zero_()

        if self.use_dino:
            self.attn_dino.load_state_dict(self.attn2.state_dict(), strict=False)
            with torch.no_grad():
                for layer in self.attn_dino.to_out:
                    for param in layer.parameters():
                        param.zero_()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        """Forward pass with multi-mechanism attention.
        
        Processing stages:
        1. Material-aware self-attention (MDA)
        2. Reference attention (RA)
        3. Multiview attention (MA) with position-aware attention
        4. Text conditioning (base attention)
        5. DINO feature conditioning (optional)
        6. Position-aware conditioning
        7. Feed-forward network
        
        Args:
            hidden_states: Input features [B * N_materials * N_views, Seq_len, Feat_dim]
            See base transformer for other parameters
            
        Returns:
            torch.Tensor: Output features
        """
        # [Full multi-mechanism processing pipeline...]
        # Key processing stages:
        # 1. Material-aware self-attention (handles albedo/mr separation)
        # 2. Reference attention (conditioned on reference features)
        # 3. View-to-view attention with geometric constraints
        # 4. Text-to-image cross-attention
        # 5. DINO feature fusion (when enabled)
        # 6. Positional conditioning (RoPE-style)
        # 7. Feed-forward network with conditional normalization

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        num_in_batch = cross_attention_kwargs.pop("num_in_batch", 1)
        mode = cross_attention_kwargs.pop("mode", None)
        mva_scale = cross_attention_kwargs.pop("mva_scale", 1.0)
        ref_scale = cross_attention_kwargs.pop("ref_scale", 1.0)
        condition_embed_dict = cross_attention_kwargs.pop("condition_embed_dict", None)
        dino_hidden_states = cross_attention_kwargs.pop("dino_hidden_states", None)
        position_voxel_indices = cross_attention_kwargs.pop("position_voxel_indices", None)
        N_pbr = len(self.pbr_setting) if self.pbr_setting is not None else 1

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        if self.use_mda:
            mda_norm_hidden_states = rearrange(
                norm_hidden_states, "(b n_pbr n) l c -> b n_pbr n l c", n=num_in_batch, n_pbr=N_pbr
            )
            attn_output = self.attn1(
                mda_norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 Reference Attention
        if "w" in mode:
            condition_embed_dict[self.layer_name] = rearrange(
                norm_hidden_states, "(b n) l c -> b (n l) c", n=num_in_batch
            )  # B, (N L), C

        if "r" in mode and self.use_ra:
            condition_embed = condition_embed_dict[self.layer_name]

            #! Only using albedo features for reference attention
            ref_norm_hidden_states = rearrange(
                norm_hidden_states, "(b n_pbr n) l c -> b n_pbr (n l) c", n=num_in_batch, n_pbr=N_pbr
            )[:, 0, ...]

            attn_output = self.attn_refview(
                ref_norm_hidden_states,
                encoder_hidden_states=condition_embed,
                attention_mask=None,
                **cross_attention_kwargs,
            )  # b (n l) c
            attn_output = rearrange(attn_output, "b n_pbr (n l) c -> (b n_pbr n) l c", n=num_in_batch, n_pbr=N_pbr)

            ref_scale_timing = ref_scale
            if isinstance(ref_scale, torch.Tensor):
                ref_scale_timing = ref_scale.unsqueeze(1).repeat(1, num_in_batch * N_pbr).view(-1)
                for _ in range(attn_output.ndim - 1):
                    ref_scale_timing = ref_scale_timing.unsqueeze(-1)
            hidden_states = ref_scale_timing * attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 1.3 Multiview Attention
        if num_in_batch > 1 and self.use_ma:
            multivew_hidden_states = rearrange(
                norm_hidden_states, "(b n_pbr n) l c -> (b n_pbr) (n l) c", n_pbr=N_pbr, n=num_in_batch
            )
            position_indices = None
            if position_voxel_indices is not None:
                if multivew_hidden_states.shape[1] in position_voxel_indices:
                    position_indices = position_voxel_indices[multivew_hidden_states.shape[1]]

            attn_output = self.attn_multiview(
                multivew_hidden_states,
                encoder_hidden_states=multivew_hidden_states,
                position_indices=position_indices,
                n_pbrs=N_pbr,
                **cross_attention_kwargs,
            )

            attn_output = rearrange(attn_output, "(b n_pbr) (n l) c -> (b n_pbr n) l c", n_pbr=N_pbr, n=num_in_batch)

            hidden_states = mva_scale * attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

            # dino attn
            if self.use_dino:
                dino_hidden_states = dino_hidden_states.unsqueeze(1).repeat(1, N_pbr * num_in_batch, 1, 1)
                dino_hidden_states = rearrange(dino_hidden_states, "b n l c -> (b n) l c")
                attn_output = self.attn_dino(
                    norm_hidden_states,
                    encoder_hidden_states=dino_hidden_states,
                    attention_mask=None,
                    **cross_attention_kwargs,
                )

                hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class ImageProjModel(torch.nn.Module):

    """Projects image embeddings into cross-attention space.
    
    Transforms CLIP embeddings into additional context tokens for conditioning.
    
    Args:
        cross_attention_dim: Dimension of attention space
        clip_embeddings_dim: Dimension of input CLIP embeddings
        clip_extra_context_tokens: Number of context tokens to generate
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):

        """Projects image embeddings to cross-attention context tokens.
        
        Args:
            image_embeds: Input embeddings [B, N, C] or [B, C]
            
        Returns:
            torch.Tensor: Context tokens [B, N*clip_extra_context_tokens, cross_attention_dim]
        """

        embeds = image_embeds
        num_token = 1
        if embeds.dim() == 3:
            num_token = embeds.shape[1]
            embeds = rearrange(embeds, "b n c -> (b n) c")

        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)

        clip_extra_context_tokens = rearrange(clip_extra_context_tokens, "(b nt) n c -> b (nt n) c", nt=num_token)

        return clip_extra_context_tokens


class UNet2p5DConditionModel(ModelMixin):

    """2.5D UNet extension for multiview PBR generation.
    
    Enhances standard 2D UNet with:
    - Multiview attention mechanisms
    - Material-aware processing
    - Position-aware conditioning
    - Dual-stream reference processing
    
    Args:
        unet: Base 2D UNet model
        train_sched: Training scheduler (DDPM)
        val_sched: Validation scheduler (EulerAncestral)
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        train_sched: DDPMScheduler = None,
        val_sched: EulerAncestralDiscreteScheduler = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        self.use_ma = True
        self.use_ra = True
        self.use_mda = True
        self.use_dino = True
        self.use_position_rope = True
        self.use_learned_text_clip = True
        self.use_dual_stream = True
        self.pbr_setting = ["albedo", "mr"]
        self.pbr_token_channels = 77

        if self.use_dual_stream and self.use_ra:
            self.unet_dual = copy.deepcopy(unet)
            self.init_attention(self.unet_dual)

        self.init_attention(
            self.unet,
            use_ma=self.use_ma,
            use_ra=self.use_ra,
            use_dino=self.use_dino,
            use_mda=self.use_mda,
            pbr_setting=self.pbr_setting,
        )
        self.init_condition(use_dino=self.use_dino)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        unet_ckpt_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.bin")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        unet = UNet2DConditionModel(**config)
        unet_2p5d = UNet2p5DConditionModel(unet)
        unet_2p5d.unet.conv_in = torch.nn.Conv2d(
            12,
            unet.conv_in.out_channels,
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            dilation=unet.conv_in.dilation,
            groups=unet.conv_in.groups,
            bias=unet.conv_in.bias is not None,
        )
        unet_ckpt = torch.load(unet_ckpt_path, map_location="cpu", weights_only=True)
        unet_2p5d.load_state_dict(unet_ckpt, strict=True)
        unet_2p5d = unet_2p5d.to(torch_dtype)
        return unet_2p5d

    def init_condition(self, use_dino):

        """Initializes conditioning mechanisms for multiview PBR generation.
        
        Sets up:
        1. Learned text embeddings: Material-specific tokens (albedo, mr) initialized to zeros
        2. DINO projector: Model to process DINO-ViT features for cross-attention
        
        Args:
            use_dino: Flag to enable DINO feature integration
        """

        if self.use_learned_text_clip:
            for token in self.pbr_setting:
                self.unet.register_parameter(
                    f"learned_text_clip_{token}", nn.Parameter(torch.zeros(self.pbr_token_channels, 1024))
                )
            self.unet.learned_text_clip_ref = nn.Parameter(torch.zeros(self.pbr_token_channels, 1024))

        if use_dino:
            self.unet.image_proj_model_dino = ImageProjModel(
                cross_attention_dim=self.unet.config.cross_attention_dim,
                clip_embeddings_dim=1536,
                clip_extra_context_tokens=4,
            )

    def init_attention(self, unet, use_ma=False, use_ra=False, use_mda=False, use_dino=False, pbr_setting=None):

        """Recursively replaces standard transformers with enhanced 2.5D blocks.
        
        Processes UNet architecture:
        1. Downsampling blocks: Replaces transformers in attention layers
        2. Middle block: Upgrades central transformers
        3. Upsampling blocks: Modifies decoder transformers
        
        Args:
            unet: UNet model to enhance
            use_ma: Enable multiview attention
            use_ra: Enable reference attention
            use_mda: Enable material-specific attention
            use_dino: Enable DINO feature integration
            pbr_setting: List of PBR materials
        """

        for down_block_i, down_block in enumerate(unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                for attn_i, attn in enumerate(down_block.attentions):
                    for transformer_i, transformer in enumerate(attn.transformer_blocks):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                                transformer,
                                f"down_{down_block_i}_{attn_i}_{transformer_i}",
                                use_ma,
                                use_ra,
                                use_mda,
                                use_dino,
                                pbr_setting,
                            )

        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            for attn_i, attn in enumerate(unet.mid_block.attentions):
                for transformer_i, transformer in enumerate(attn.transformer_blocks):
                    if isinstance(transformer, BasicTransformerBlock):
                        attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                            transformer, f"mid_{attn_i}_{transformer_i}", use_ma, use_ra, use_mda, use_dino, pbr_setting
                        )

        for up_block_i, up_block in enumerate(unet.up_blocks):
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                for attn_i, attn in enumerate(up_block.attentions):
                    for transformer_i, transformer in enumerate(attn.transformer_blocks):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                                transformer,
                                f"up_{up_block_i}_{attn_i}_{transformer_i}",
                                use_ma,
                                use_ra,
                                use_mda,
                                use_dino,
                                pbr_setting,
                            )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        *args,
        added_cond_kwargs=None,
        cross_attention_kwargs=None,
        down_intrablock_additional_residuals=None,
        down_block_res_samples=None,
        mid_block_res_sample=None,
        **cached_condition,
    ):

        """Forward pass with multiview/material conditioning.
        
        Key stages:
        1. Input preparation (concat normal/position maps)
        2. Reference feature extraction (dual-stream)
        3. Position encoding (voxel indices)
        4. DINO feature projection
        5. Main UNet processing with attention conditioning
        
        Args:
            sample: Input latents [B, N_pbr, N_gen, C, H, W]
            cached_condition: Dictionary containing:
                - embeds_normal: Normal map embeddings
                - embeds_position: Position map embeddings
                - ref_latents: Reference image latents
                - dino_hidden_states: DINO features
                - position_maps: 3D position maps
                - mva_scale: Multiview attention scale
                - ref_scale: Reference attention scale
                
        Returns:
            torch.Tensor: Output features
        """

        B, N_pbr, N_gen, _, H, W = sample.shape
        assert H == W

        if "cache" not in cached_condition:
            cached_condition["cache"] = {}

        sample = [sample]
        if "embeds_normal" in cached_condition:
            sample.append(cached_condition["embeds_normal"].unsqueeze(1).repeat(1, N_pbr, 1, 1, 1, 1))
        if "embeds_position" in cached_condition:
            sample.append(cached_condition["embeds_position"].unsqueeze(1).repeat(1, N_pbr, 1, 1, 1, 1))
        sample = torch.cat(sample, dim=-3)

        sample = rearrange(sample, "b n_pbr n c h w -> (b n_pbr n) c h w")

        encoder_hidden_states_gen = encoder_hidden_states.unsqueeze(-3).repeat(1, 1, N_gen, 1, 1)
        encoder_hidden_states_gen = rearrange(encoder_hidden_states_gen, "b n_pbr n l c -> (b n_pbr n) l c")

        if added_cond_kwargs is not None:
            text_embeds_gen = added_cond_kwargs["text_embeds"].unsqueeze(1).repeat(1, N_gen, 1)
            text_embeds_gen = rearrange(text_embeds_gen, "b n c -> (b n) c")
            time_ids_gen = added_cond_kwargs["time_ids"].unsqueeze(1).repeat(1, N_gen, 1)
            time_ids_gen = rearrange(time_ids_gen, "b n c -> (b n) c")
            added_cond_kwargs_gen = {"text_embeds": text_embeds_gen, "time_ids": time_ids_gen}
        else:
            added_cond_kwargs_gen = None

        if self.use_position_rope:
            if "position_voxel_indices" in cached_condition["cache"]:
                position_voxel_indices = cached_condition["cache"]["position_voxel_indices"]
            else:
                if "position_maps" in cached_condition:
                    position_voxel_indices = calc_multires_voxel_idxs(
                        cached_condition["position_maps"],
                        grid_resolutions=[H, H // 2, H // 4, H // 8],
                        voxel_resolutions=[H * 8, H * 4, H * 2, H],
                    )
                    cached_condition["cache"]["position_voxel_indices"] = position_voxel_indices
        else:
            position_voxel_indices = None

        if self.use_dino:
            if "dino_hidden_states_proj" in cached_condition["cache"]:
                dino_hidden_states = cached_condition["cache"]["dino_hidden_states_proj"]
            else:
                assert "dino_hidden_states" in cached_condition
                dino_hidden_states = cached_condition["dino_hidden_states"]
                dino_hidden_states = self.image_proj_model_dino(dino_hidden_states)
                cached_condition["cache"]["dino_hidden_states_proj"] = dino_hidden_states
        else:
            dino_hidden_states = None

        if self.use_ra:
            if "condition_embed_dict" in cached_condition["cache"]:
                condition_embed_dict = cached_condition["cache"]["condition_embed_dict"]
            else:
                condition_embed_dict = {}
                ref_latents = cached_condition["ref_latents"]
                N_ref = ref_latents.shape[1]

                if not self.use_dual_stream:
                    ref_latents = [ref_latents]
                    if "embeds_normal" in cached_condition:
                        ref_latents.append(torch.zeros_like(ref_latents[0]))
                    if "embeds_position" in cached_condition:
                        ref_latents.append(torch.zeros_like(ref_latents[0]))
                    ref_latents = torch.cat(ref_latents, dim=2)

                ref_latents = rearrange(ref_latents, "b n c h w -> (b n) c h w")

                encoder_hidden_states_ref = self.unet.learned_text_clip_ref.repeat(B, N_ref, 1, 1)

                encoder_hidden_states_ref = rearrange(encoder_hidden_states_ref, "b n l c -> (b n) l c")

                if added_cond_kwargs is not None:
                    text_embeds_ref = added_cond_kwargs["text_embeds"].unsqueeze(1).repeat(1, N_ref, 1)
                    text_embeds_ref = rearrange(text_embeds_ref, "b n c -> (b n) c")
                    time_ids_ref = added_cond_kwargs["time_ids"].unsqueeze(1).repeat(1, N_ref, 1)
                    time_ids_ref = rearrange(time_ids_ref, "b n c -> (b n) c")
                    added_cond_kwargs_ref = {
                        "text_embeds": text_embeds_ref,
                        "time_ids": time_ids_ref,
                    }
                else:
                    added_cond_kwargs_ref = None

                noisy_ref_latents = ref_latents
                timestep_ref = 0
                if self.use_dual_stream:
                    unet_ref = self.unet_dual
                else:
                    unet_ref = self.unet
                unet_ref(
                    noisy_ref_latents,
                    timestep_ref,
                    encoder_hidden_states=encoder_hidden_states_ref,
                    class_labels=None,
                    added_cond_kwargs=added_cond_kwargs_ref,
                    # **kwargs
                    return_dict=False,
                    cross_attention_kwargs={
                        "mode": "w",
                        "num_in_batch": N_ref,
                        "condition_embed_dict": condition_embed_dict,
                    },
                )
                cached_condition["cache"]["condition_embed_dict"] = condition_embed_dict
        else:
            condition_embed_dict = None

        mva_scale = cached_condition.get("mva_scale", 1.0)
        ref_scale = cached_condition.get("ref_scale", 1.0)

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states_gen,
            *args,
            class_labels=None,
            added_cond_kwargs=added_cond_kwargs_gen,
            down_intrablock_additional_residuals=(
                [sample.to(dtype=self.unet.dtype) for sample in down_intrablock_additional_residuals]
                if down_intrablock_additional_residuals is not None
                else None
            ),
            down_block_additional_residuals=(
                [sample.to(dtype=self.unet.dtype) for sample in down_block_res_samples]
                if down_block_res_samples is not None
                else None
            ),
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=self.unet.dtype) if mid_block_res_sample is not None else None
            ),
            return_dict=False,
            cross_attention_kwargs={
                "mode": "r",
                "num_in_batch": N_gen,
                "dino_hidden_states": dino_hidden_states,
                "condition_embed_dict": condition_embed_dict,
                "mva_scale": mva_scale,
                "ref_scale": ref_scale,
                "position_voxel_indices": position_voxel_indices,
            },
        )
