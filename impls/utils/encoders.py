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
}
