import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP
from equimo.io import load_model


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out
    
# image_encoder = load_model("vit", "dinov3_vits16_pretrain_lvd1689m")

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from equimo.io import load_model
from utils.networks import MLP


# 🔥 Load DINOv3 once, globally, fully outside Flax/JAX transforms
DINO_V3_ENCODER = load_model("vit", "dinov3_vits16_pretrain_lvd1689m")

class DinoV3Encoder(nn.Module):
    """Frozen Equimo DINOv3 encoder wrapper with frame stacking support."""
    mlp_hidden_dims: Sequence[int] = (512,)
    apply_mlp: bool = True

    def setup(self):
        # Only trainable head lives in Flax params
        if self.apply_mlp:
            self.head = MLP(self.mlp_hidden_dims, activate_final=True)

    def _split_frames(self, x):
        """Split stacked frames along channel dim.

        x: (B, H, W, 3*k) or (H, W, 3*k)
        → list of frames, each (B, H, W, 3) or (H, W, 3)
        """
        k = x.shape[-1] // 3
        return jnp.split(x, k, axis=-1)

    def __call__(self, x, train=True):
        # Normalize pixel values
        x = x.astype(jnp.float32) / 255.0

        # Split stacked frames
        frames = self._split_frames(x)   # list of (B, H, W, 3) or (H, W, 3)

        enc = DINO_V3_ENCODER            # 🔑 global, frozen, never None
        key = jax.random.PRNGKey(0)      # fixed, deterministic

        def encode_single(img_hw3):
            # img_hw3: (H, W, 3)
            img_chw = jnp.transpose(img_hw3, (2, 0, 1))   # (3, H, W)
            feats = enc.features(img_chw, key)            # (tokens, dim)
            cls = enc.norm(feats[0])                      # (dim,)
            return cls

        frame_embs = []
        for f in frames:
            single = (x.ndim == 3)

            if f.ndim == 3:
                f = f[None, ...]

            cls_batch = jax.vmap(encode_single)(f)  # (1, dim)

            if single:
                cls_batch = cls_batch[0]    # --> (dim,)
            frame_embs.append(cls_batch)

        # Concatenate over frames
        out = jnp.concatenate(frame_embs, axis=-1)  # (B, dim * num_frames)

        if self.apply_mlp:
            out = self.head(out)

        return out

class VisionTransformerEncoder(nn.Module):
    """Plain Vision Transformer encoder (trainable from scratch).
    
    This is an ablation of DINOv3 to test whether pretraining or architecture drives performance.
    Implements standard ViT architecture: patch embedding → positional encoding → transformer blocks → CLS token.
    """
    
    patch_size: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = None  # If None, uses 4 * embed_dim (standard ViT)
    dropout_rate: float = 0.0
    mlp_hidden_dims: Sequence[int] = (512,)
    apply_mlp: bool = True
    
    def setup(self):
        # MLP dimension defaults to 4x embed_dim (standard ViT)
        if self.mlp_dim is None:
            self.mlp_dim = 4 * self.embed_dim
        
        # Patch embedding: Conv layer that creates patches
        self.patch_embed = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding='VALID',
            kernel_init=nn.initializers.xavier_uniform(),
        )
        
        # CLS token (learnable)
        self.cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        
        # Positional embeddings: create a large enough embedding for typical image sizes
        # We'll use max_patches = 14*14 = 196 (for 224x224 images with patch_size=16)
        # This covers most common image sizes. For larger images, we can interpolate.
        max_patches = 196  # 14*14 patches
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, max_patches + 1, self.embed_dim)  # +1 for CLS token
        )
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        
        # Layer norm after transformer
        self.ln_post = nn.LayerNorm()
        
        # Optional MLP head
        if self.apply_mlp:
            self.head = MLP(self.mlp_hidden_dims, activate_final=True)
    
    def _split_frames(self, x):
        """Split stacked frames along channel dim.
        
        x: (B, H, W, 3*k) or (H, W, 3*k)
        → list of frames, each (B, H, W, 3) or (H, W, 3)
        """
        k = x.shape[-1] // 3
        return jnp.split(x, k, axis=-1)
    
    def _encode_single_frame(self, img, train=True):
        """Encode a single frame through ViT.
        
        Args:
            img: (B, H, W, 3) or (H, W, 3) image tensor
            train: Whether in training mode
        
        Returns:
            CLS token representation: (B, embed_dim) or (embed_dim,)
        """
        # Normalize pixel values
        x = img.astype(jnp.float32) / 255.0
        
        # Handle single image vs batch
        single = (x.ndim == 3)
        if single:
            x = x[None, ...]  # (1, H, W, 3)
        
        B, H, W, C = x.shape
        
        # Patch embedding: (B, H, W, C) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, H/patch_size, W/patch_size, embed_dim)
        num_patches_h = x.shape[1]
        num_patches_w = x.shape[2]
        num_patches = num_patches_h * num_patches_w
        
        x = x.reshape(B, num_patches, self.embed_dim)
        
        # Add CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        # Use the pre-created positional embeddings, truncating if needed
        # pos_embed shape: (1, max_patches + 1, embed_dim)
        required_len = num_patches + 1  # +1 for CLS token
        if required_len <= self.pos_embed.shape[1]:
            # Use truncated positional embeddings
            pos_embed = self.pos_embed[:, :required_len, :]  # (1, num_patches + 1, embed_dim)
        else:
            # Interpolate if we need more patches (shouldn't happen often)
            # Simple 2D interpolation of positional embeddings
            # For now, just repeat the last embedding (better would be bilinear interpolation)
            pos_embed_base = self.pos_embed  # (1, max_patches + 1, embed_dim)
            # Repeat the last patch embedding
            extra_patches = required_len - pos_embed_base.shape[1]
            last_patch = pos_embed_base[:, -1:, :]  # (1, 1, embed_dim)
            extra_embeds = jnp.repeat(last_patch, extra_patches, axis=1)
            pos_embed = jnp.concatenate([pos_embed_base, extra_embeds], axis=1)
        
        x = x + pos_embed
        
        # Apply dropout
        if self.dropout_rate > 0:
            x = self.dropout(x, deterministic=not train)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, train=train)
        
        # Layer norm
        x = self.ln_post(x)
        
        # Extract CLS token (first token)
        cls_token = x[:, 0]  # (B, embed_dim)
        
        if single:
            cls_token = cls_token[0]  # (embed_dim,)
        
        return cls_token
    
    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        """Forward pass with frame stacking support.
        
        Args:
            x: (B, H, W, 3*k) or (H, W, 3*k) image tensor with stacked frames
            train: Whether in training mode
            cond_var: Unused (for compatibility)
        
        Returns:
            Encoded representation: (B, embed_dim * num_frames) or (embed_dim * num_frames,)
        """
        # Split stacked frames
        frames = self._split_frames(x)  # list of (B, H, W, 3) or (H, W, 3)
        
        # Encode each frame
        frame_embs = []
        for f in frames:
            cls_emb = self._encode_single_frame(f, train=train)
            frame_embs.append(cls_emb)
        
        # Concatenate over frames
        out = jnp.concatenate(frame_embs, axis=-1)  # (B, embed_dim * num_frames) or (embed_dim * num_frames,)
        
        # Apply MLP head if enabled
        if self.apply_mlp:
            # Handle single vs batch
            if out.ndim == 1:
                out = out[None, ...]
                out = self.head(out)
                out = out[0]
            else:
                out = self.head(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP."""
    
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0
    
    def setup(self):
        # Self-attention
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        
        # MLP
        self.mlp = MLP(
            hidden_dims=(self.mlp_dim,),
            activate_final=False,
            activations=nn.gelu,
        )
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout1 = nn.Dropout(rate=self.dropout_rate)
            self.dropout2 = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, x, train=True):
        """Apply transformer block.
        
        Args:
            x: (B, seq_len, embed_dim) input tokens
            train: Whether in training mode
        
        Returns:
            (B, seq_len, embed_dim) output tokens
        """
        # Self-attention with residual
        attn_out = self.attention(x, x, deterministic=not train)
        if self.dropout_rate > 0:
            attn_out = self.dropout1(attn_out, deterministic=not train)
        x = x + attn_out
        x = self.ln1(x)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        if self.dropout_rate > 0:
            mlp_out = self.dropout2(mlp_out, deterministic=not train)
        x = x + mlp_out
        x = self.ln2(x)
        
        return x


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block (1x1 -> 3x3 -> 1x1 convs with residual connection)."""
    
    filters: int
    stride: int = 1
    use_projection: bool = False
    
    @nn.compact
    def __call__(self, x, train=True):
        """Apply bottleneck block.
        
        Args:
            x: (B, H, W, C) input tensor
            train: Whether in training mode
        
        Returns:
            (B, H', W', filters) output tensor
        """
        initializer = nn.initializers.he_normal()
        
        # Shortcut connection
        shortcut = x
        if self.use_projection:
            # 1x1 conv to match dimensions
            shortcut = nn.Conv(
                features=self.filters * 4,  # Bottleneck expands by 4x
                kernel_size=(1, 1),
                strides=self.stride,
                kernel_init=initializer,
                use_bias=False,
            )(shortcut)
            shortcut = nn.BatchNorm(use_running_average=not train)(shortcut)
        
        # Main path: 1x1 -> 3x3 -> 1x1
        # 1x1 conv (reduce channels)
        out = nn.Conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=initializer,
            use_bias=False,
        )(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        
        # 3x3 conv (main conv)
        out = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=self.stride,
            padding='SAME',
            kernel_init=initializer,
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        
        # 1x1 conv (expand channels)
        out = nn.Conv(
            features=self.filters * 4,  # Expand by 4x
            kernel_size=(1, 1),
            strides=1,
            kernel_init=initializer,
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        
        # Add residual and apply ReLU
        out = out + shortcut
        out = nn.relu(out)
        
        return out


class ResNet50Encoder(nn.Module):
    """ResNet50 encoder (trainable from scratch).
    
    Standard ResNet50 architecture with bottleneck blocks:
    - Initial 7x7 conv + max pooling
    - 4 stages with bottleneck blocks (3, 4, 6, 3 blocks)
    - Global average pooling
    - Optional MLP head
    """
    
    mlp_hidden_dims: Sequence[int] = (512,)
    apply_mlp: bool = True
    
    def setup(self):
        # Optional MLP head
        if self.apply_mlp:
            self.head = MLP(self.mlp_hidden_dims, activate_final=True)
    
    def _split_frames(self, x):
        """Split stacked frames along channel dim.
        
        x: (B, H, W, 3*k) or (H, W, 3*k)
        → list of frames, each (B, H, W, 3) or (H, W, 3)
        """
        k = x.shape[-1] // 3
        return jnp.split(x, k, axis=-1)
    
    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        """Forward pass with frame stacking support.
        
        Args:
            x: (B, H, W, 3*k) or (H, W, 3*k) image tensor with stacked frames
            train: Whether in training mode
            cond_var: Unused (for compatibility)
        
        Returns:
            Encoded representation: (B, 2048 * num_frames) or (2048 * num_frames,)
        """
        # Split stacked frames
        frames = self._split_frames(x)  # list of (B, H, W, 3) or (H, W, 3)
        
        # Encode each frame
        frame_embs = []
        for frame_idx, f in enumerate(frames):
            # Normalize pixel values
            img = f.astype(jnp.float32) / 255.0
            
            # Handle single image vs batch
            single = (img.ndim == 3)
            if single:
                img = img[None, ...]  # (1, H, W, 3)
            
            initializer = nn.initializers.he_normal()
            
            # Initial 7x7 conv layer
            x_frame = nn.Conv(
                features=64,
                kernel_size=(7, 7),
                strides=2,
                padding='SAME',
                kernel_init=initializer,
                use_bias=False,
            )(img)
            x_frame = nn.BatchNorm(use_running_average=not train)(x_frame)
            x_frame = nn.relu(x_frame)
            
            # Max pooling
            x_frame = nn.max_pool(
                x_frame,
                window_shape=(3, 3),
                strides=(2, 2),
                padding='SAME',
            )
            
            # Stage 1: 3 bottleneck blocks, 256 filters (64 * 4)
            x_frame = BottleneckBlock(filters=64, stride=1, use_projection=True)(x_frame, train=train)
            for _ in range(2):
                x_frame = BottleneckBlock(filters=64, stride=1, use_projection=False)(x_frame, train=train)
            
            # Stage 2: 4 bottleneck blocks, 512 filters (128 * 4)
            x_frame = BottleneckBlock(filters=128, stride=2, use_projection=True)(x_frame, train=train)
            for _ in range(3):
                x_frame = BottleneckBlock(filters=128, stride=1, use_projection=False)(x_frame, train=train)
            
            # Stage 3: 6 bottleneck blocks, 1024 filters (256 * 4)
            x_frame = BottleneckBlock(filters=256, stride=2, use_projection=True)(x_frame, train=train)
            for _ in range(5):
                x_frame = BottleneckBlock(filters=256, stride=1, use_projection=False)(x_frame, train=train)
            
            # Stage 4: 3 bottleneck blocks, 2048 filters (512 * 4)
            x_frame = BottleneckBlock(filters=512, stride=2, use_projection=True)(x_frame, train=train)
            for _ in range(2):
                x_frame = BottleneckBlock(filters=512, stride=1, use_projection=False)(x_frame, train=train)
            
            # Global average pooling
            # x_frame shape: (B, H', W', 2048)
            x_frame = jnp.mean(x_frame, axis=(1, 2))  # (B, 2048)
            
            if single:
                x_frame = x_frame[0]  # (2048,)
            
            frame_embs.append(x_frame)
        
        # Concatenate over frames
        out = jnp.concatenate(frame_embs, axis=-1)  # (B, 2048 * num_frames) or (2048 * num_frames,)
        
        # Apply MLP head if enabled
        if self.apply_mlp:
            # Handle single vs batch
            if out.ndim == 1:
                out = out[None, ...]
                out = self.head(out)
                out = out[0]
            else:
                out = self.head(out)
        
        return out


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    # 'dinov3_identity': functools.partial(),#TODO: Complete this
    'dinov3_s16': DinoV3Encoder,
    # Vision Transformer encoders (trainable from scratch, ablation of DINOv3)
    'vit_small': functools.partial(
        VisionTransformerEncoder,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
    ),
    'vit_base': functools.partial(
        VisionTransformerEncoder,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
    ),
    # ResNet50 encoder (trainable from scratch)
    'resnet50': ResNet50Encoder,
}
