from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import exists


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
    ):
    # def ok234324():
        print("\n[模型初始化] 开始构建网络结构...")
        super().__init__()
        print("[父类初始化] 已完成基类构造")

        # 空间变换器与上下文维度交叉验证
        print("[参数检查] 验证 spatial_transformer 与 context_dim 的依赖关系...")
        if use_spatial_transformer:
            assert_msg = "❌ 未设置 context_dim 但启用了空间变换器（需要交叉注意力机制）"
            assert context_dim is not None, assert_msg
            print("✓ 空间变换器启用 | context_dim =", context_dim)
        
        if context_dim is not None:
            assert_msg = "❌ 配置了 context_dim 但未启用空间变换器"
            assert use_spatial_transformer, assert_msg
            print("✓ 上下文维度验证通过 | 使用空间变换器进行交叉注意力")
            
            # 处理OmegaConf列表类型转换
            from omegaconf.listconfig import ListConfig
            if isinstance(context_dim, ListConfig):
                print("▷ 检测到 context_dim 为 OmegaConf 列表类型，正在转换为 Python list...")
                context_dim = list(context_dim)
                print("✓ 类型转换完成 | context_dim 类型:", type(context_dim).__name__)

        # 注意力头数配置逻辑
        print("\n[注意力机制] 配置多头注意力参数...")
        if num_heads_upsample == -1:
            print(f"▷ num_heads_upsample(-1) 使用默认值 num_heads({num_heads})")
            num_heads_upsample = num_heads
            print(f"✓ 最终值 num_heads_upsample = {num_heads_upsample}")
        
        head_check_flag = False
        if num_heads == -1:
            assert_msg = "❌ 必须设置 num_heads 或 num_head_channels 至少一个"
            assert num_head_channels != -1, assert_msg
            print("✓ 头数验证通过 | 使用 num_head_channels =", num_head_channels)
            head_check_flag = True
            
        if num_head_channels == -1:
            assert_msg = "❌ 必须设置 num_heads 或 num_head_channels 至少一个"
            assert num_heads != -1, assert_msg
            print("✓ 头数验证通过 | 使用 num_heads =", num_heads)
            head_check_flag = True
            
        if not head_check_flag:
            print("⚠️ 注意：同时指定了 num_heads 和 num_head_channels，优先使用 num_head_channels")

        # 核心参数初始化
        print("\n[参数初始化] 设置网络基础参数...")
        param_config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "out_channels": out_channels
        }
        for k, v in param_config.items():
            print(f"▷ 设置 {k.ljust(15)} = {v}")
            setattr(self, k, v)

        # 残差块配置逻辑
        print("\n[残差块] 配置各层残差块数量...")
        if isinstance(num_res_blocks, int):
            print(f"▷ 扩展全局残差块数({num_res_blocks})到各分辨率层级...")
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
            print(f"✓ 生成列表: num_res_blocks = {self.num_res_blocks}")
        else:
            print("▷ 使用自定义层级残差块配置:", num_res_blocks)
            if len(num_res_blocks) != len(channel_mult):
                err_msg = (f"❌ 残差块配置维度不匹配 | "
                         f"num_res_blocks长度({len(num_res_blocks)}) ≠ channel_mult长度({len(channel_mult)})")
                raise ValueError(err_msg)
            self.num_res_blocks = num_res_blocks
            print("✓ 残差块配置验证通过")

    #  def ok2423():
        print("\n" + "="*50)
        print("[UNet初始化] 开始构建UNet模型结构")
        print("="*50)

        # 自注意力禁用配置验证
        if disable_self_attentions is not None:
            print("\n[自注意力配置] 验证禁用层配置...")
            expected_len = len(channel_mult)
            actual_len = len(disable_self_attentions)
            assert actual_len == expected_len, (
                f"❌ 禁用自注意力配置维度不匹配 | "
                f"channel_mult长度({expected_len}) ≠ disable_self_attentions长度({actual_len})"
            )
            print(f"✓ 禁用配置验证通过 | 各层级禁用状态：{disable_self_attentions}")

        # 注意力块数量验证
        if num_attention_blocks is not None:
            print("\n[注意力块] 验证注意力块配置...")
            res_blocks_len = len(self.num_res_blocks)
            attn_blocks_len = len(num_attention_blocks)
            
            # 维度匹配验证
            assert attn_blocks_len == res_blocks_len, (
                f"❌ 注意力块维度不匹配 | "
                f"残差块数({res_blocks_len}) ≠ 注意力块数({attn_blocks_len})"
            )
            
            # 数量合理性验证
            invalid_blocks = [i for i in range(attn_blocks_len) if self.num_res_blocks[i] < num_attention_blocks[i]]
            assert not invalid_blocks, (
                f"❌ 注意力块数量超过残差块 | "
                f"非法层级索引：{invalid_blocks}"
            )
            
            warning_msg = (
                f"▷ 注意：num_attention_blocks({num_attention_blocks})的优先级低于attention_resolutions({attention_resolutions})\n"
                f"▷ 即当num_attention_blocks[i]>0但2**i不在attention_resolutions时，仍不会设置注意力"
            )
            print(warning_msg)
            print("✓ 注意力块配置验证通过")

        # 基础参数初始化
        print("\n[核心参数] 设置模型基础配置：")
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        base_params = {
            "attention_resolutions": attention_resolutions,
            "dropout": dropout,
            "channel_mult": channel_mult,
            "conv_resample": conv_resample,
            "num_classes": num_classes,
            "use_checkpoint": use_checkpoint,
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "num_heads_upsample": num_heads_upsample,
            "predict_codebook_ids": n_embed is not None
        }
        for param, value in base_params.items():
            print(f"│ {param.ljust(25)} = {str(value).ljust(30)}")
        
        # 数据类型配置
        dtype_stack = []
        if use_fp16:
            dtype_stack.append("float16")
        if use_bf16:
            dtype_stack.append("bfloat16")
        final_dtype = th.float32
        if use_fp16:
            final_dtype = th.float16
        if use_bf16:
            final_dtype = th.bfloat16
        print(f"└─ dtype {' ← '.join(dtype_stack)} ⇒ {str(final_dtype)}")
        self.dtype = final_dtype

        # 时间嵌入层构建
        print("\n[时间嵌入] 构建时间编码模块：")
        time_embed_dim = model_channels * 4
        print(f"▷ 输入维度: {model_channels} → 中间维度: {time_embed_dim}")
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        print(f"✓ 时间编码器结构: {self.time_embed}")

        # 类别条件处理
        if self.num_classes is not None:
            print("\n[类别条件] 配置条件嵌入层：")
            if isinstance(self.num_classes, int):
                print(f"▷ 离散类别嵌入 (num_classes={self.num_classes})")
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
                print(f"✓ 嵌入层维度: ({num_classes}, {time_embed_dim})")
            
            elif self.num_classes == "continuous":
                print("▷ 连续值条件处理 → 线性投影层")
                self.label_emb = nn.Linear(1, time_embed_dim)
                print(f"✓ 线性层: 1 → {time_embed_dim}")
            
            elif self.num_classes == "sequential":
                print("▷ 序列输入条件处理 → 多层投影")
                assert adm_in_channels is not None, "❌ 序列条件需要 adm_in_channels 参数"
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
                print(f"✓ 投影网络结构: {self.label_emb}")
            
            else:
                raise ValueError(f"❌ 无效的num_classes类型: {type(self.num_classes)}")



    # def ok432432():
        print("\n" + "="*50)
        print("[UNet编码器初始化] 开始构建输入块结构")
        print("="*50)
        
        # 初始化输入块
        print("\n[输入块] 创建初始卷积层：")
        print(f"▷ 输入维度: {in_channels} → 输出维度: {model_channels}")
        print(f"▷ 卷积类型: {dims}D卷积 | 核尺寸: 3x3 | 填充: 1")
        input_conv = conv_nd(dims, in_channels, model_channels, 3, padding=1)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(input_conv)
        ])
        print(f"✓ 初始输入块构建完成 | 模块结构：{self.input_blocks[-1]}")

        # 初始化特征跟踪参数
        print("\n[参数初始化] 设置跟踪变量：")
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        print(f"│ 当前特征尺寸: {self._feature_size}")
        print(f"│ 通道数序列: {input_block_chans}")
        print(f"│ 当前通道数(ch): {ch}")
        print(f"└─ 初始下采样率(ds): {ds}x")

        print("\n" + "="*50)
        print("[层级构建] 开始逐层构造编码器")
        print("="*50)

    # def ok32432():
        # 遍历每个分辨率层级
        for level, mult in enumerate(channel_mult):
            print("\n" + "="*40)
            print(f"▶ 层级 {level+1}/{len(channel_mult)} [通道倍数: ×{mult}]")
            print(f"▷ 目标通道数: {mult * model_channels}")
            print(f"▷ 本层残差块数: {self.num_res_blocks[level]}")

            # 构建当前层级的残差块
            for nr in range(self.num_res_blocks[level]):
                print("\n" + "-"*30)
                print(f"● 残差块 {nr+1}/{self.num_res_blocks[level]}")
                print(f"│ 输入通道: {ch} → 输出通道: {mult * model_channels}")
                print(f"│ 使用checkpoint: {'✓' if use_checkpoint else '✗'}")
                print(f"└─ scale_shift_norm: {'✓' if use_scale_shift_norm else '✗'}")

                # 构建残差块
                res_block = ResBlock(
                    ch, 
                    time_embed_dim,
                    dropout,
                    out_channels=mult * model_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                layers = [res_block]
                
                # 更新通道数
                prev_ch = ch
                ch = mult * model_channels
                print(f"\n✓ 残差块构建完成")
                print(f"│ 原通道数: {prev_ch}")
                print(f"└─ 新通道数: {ch}")

            #  def ok324324():
                print("\n" + "-"*40)
                print(f"▷ 当前下采样率: {ds}x")
                print(f"▷ 注意力分辨率列表: {attention_resolutions}")

                # 更新通道数
                prev_ch = ch
                ch = mult * model_channels
                print(f"\n● 通道数更新: {prev_ch} → {ch} (倍数: {mult}x)")

                if ds in attention_resolutions:
                    print("\n[注意力机制] 配置注意力头参数：")
                    print(f"✓ 当前下采样率 {ds}x 在注意力分辨率列表中")

                    # 头数计算逻辑
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        print(f"│ 模式: 固定头数 | 头数={num_heads}")
                        print(f"│ 计算得头维度: {dim_head} = {ch} // {num_heads}")
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                        print(f"│ 模式: 固定头维度 | 头维度={num_head_channels}")
                        print(f"│ 计算得头数: {num_heads} = {ch} // {num_head_channels}")

                    # 传统模式处理
                    if legacy:
                        legacy_note = "传统模式覆盖: "
                        if use_spatial_transformer:
                            dim_head = ch // num_heads
                            legacy_note += f"空间变换器 → 头维度={dim_head}"
                        else:
                            dim_head = num_head_channels
                            legacy_note += f"常规注意力 → 头维度={dim_head}"
                        print(f"│ {legacy_note}")

                    # 自注意力禁用判断
                    disabled_sa = disable_self_attentions[level] if exists(disable_self_attentions) else False
                    sa_status = "禁用" if disabled_sa else "启用"
                    print(f"└─ 自注意力状态: {sa_status} (层级 {level+1})")

                    # 添加注意力模块
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        print("\n[模块添加] 正在构建注意力层：")
                        if use_spatial_transformer:
                            print(f"▷ 空间变换器 | 上下文维度={context_dim}")
                            print(f"▷ 线性投影: {'✓' if use_linear_in_transformer else '✗'}")
                            transformer = SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth,
                                context_dim=context_dim, disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint
                            )
                            print(f"✓ 创建 {transformer.__class__.__name__}")
                        else:
                            print("▷ 常规注意力块")
                            print(f"▷ 新注意力顺序: {'✓' if use_new_attention_order else '✗'}")
                            attention = AttentionBlock(
                                ch, use_checkpoint=use_checkpoint,
                                num_heads=num_heads, num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order
                            )
                            print(f"✓ 创建 {attention.__class__.__name__}")

                        layers.append(transformer if use_spatial_transformer else attention)
                        print(f"└─ 已添加注意力模块 (当前层数: {len(layers)})")

                # 添加模块并更新跟踪器
                print("\n[网络结构] 更新输入块：")
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                print(f"▷ 新增模块包含 {len(layers)} 个子层")
                print(f"▷ 当前输入块数: {len(self.input_blocks)}")

                self._feature_size += ch
                print(f"│ 累计特征尺寸: {self._feature_size}")

                input_block_chans.append(ch)
                print(f"└─ 更新通道序列: {input_block_chans}")
        # def ok2342():
            # 下采样条件检查
            print("\n" + "="*40)
            print(f"▷ 检查层级 {level+1}/{len(channel_mult)} 是否需要下采样")
            if level != len(channel_mult) - 1:
                print("✓ 非最后层级，添加下采样模块")
                
                # 下采样类型选择
                down_type = "ResBlock下采样" if resblock_updown else "常规下采样"
                print(f"▷ 下采样类型: {down_type}")
                print(f"▷ 输入通道: {ch} → 输出通道: {ch} (保持相同)")
                
                # 构建下采样模块
                down_module = None
                if resblock_updown:
                    print("│ 使用带有下采样的残差块:")
                    print(f"│ • 时间嵌入维度: {time_embed_dim}")
                    print(f"│ • 使用checkpoint: {'✓' if use_checkpoint else '✗'}")
                    print(f"└─ • scale_shift归一化: {'✓' if use_scale_shift_norm else '✗'}")
                    down_module = ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=ch,  # 保持相同通道
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True
                    )
                else:
                    print("│ 使用常规下采样层:")
                    print(f"│ • 卷积重采样类型: {conv_resample}")
                    print(f"└─ • 输出通道: {ch}")
                    down_module = Downsample(ch, conv_resample, dims=dims, out_channels=ch)
                
                # 添加模块到输入块
                print("\n[模块添加] 正在添加下采样层：")
                self.input_blocks.append(TimestepEmbedSequential(down_module))
                print(f"✓ 已添加 {down_module.__class__.__name__}")
                print(f"│ 当前输入块总数: {len(self.input_blocks)}")
                
                # 更新跟踪变量
                prev_ch = ch
                ch = ch  # 此处保持相同（实际可能根据Downsample实现变化）
                input_block_chans.append(ch)
                print(f"│ 通道数保持: {prev_ch} → {ch}") 
                
                prev_ds = ds
                ds *= 2
                print(f"│ 下采样率更新: {prev_ds}x → {ds}x")
                
                prev_feature_size = self._feature_size
                self._feature_size += ch
                print(f"└─ 累计特征尺寸: {prev_feature_size} + {ch} = {self._feature_size}")
            else:
                print("✗ 最后层级，跳过下采样")

    # def ok23432():
        print("\n" + "="*40)
        print("[注意力头配置] 开始计算注意力头参数")
        print(f"▷ 输入通道数(ch): {ch}")
        print(f"▷ 初始头数(num_heads): {num_heads}")
        print(f"▷ 头通道数(num_head_channels): {num_head_channels}")
        print(f"▷ Legacy模式: {'启用' if legacy else '禁用'}")
        print(f"▷ 空间变换器: {'使用' if use_spatial_transformer else '未使用'}")

        # 模式选择分支
        if num_head_channels == -1:
            print("\n● 模式：固定头数 → 计算头维度")
            dim_head = ch // num_heads
            print(f"│ 计算式：dim_head = {ch} // {num_heads}")
            print(f"└─ 头维度: {dim_head} (头数保持 {num_heads})")
        else:
            print("\n● 模式：固定头维度 → 计算头数")
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
            print(f"│ 计算式：num_heads = {ch} // {num_head_channels}")
            print(f"└─ 头数: {num_heads} (头维度保持 {dim_head})")

        # Legacy模式覆盖
        if legacy:
            print("\n⚠️ Legacy模式参数调整：")
            original_dim_head = dim_head
            if use_spatial_transformer:
                dim_head = ch // num_heads
                print(f"│ 空间变换器模式 → dim_head = {ch} // {num_heads} = {dim_head}")
            else:
                dim_head = num_head_channels
                print(f"│ 常规注意力模式 → 维持头维度: {dim_head}")
            print(f"└─ 维度调整：{original_dim_head} → {dim_head}")

        print("\n最终配置：")
        print(f"num_heads: {num_heads}")
        print(f"dim_head: {dim_head}")
        print("="*40)

    
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
    #  def ok324():
        print("\n" + "="*50)
        print("[前向传播] 开始执行UNet前向计算")
        print("="*50)

        # 条件一致性验证
        print("\n[条件验证] 检查类别条件一致性...")
        condition_check = (y is not None) == (self.num_classes is not None)
        assert condition_check, (
            f"❌ 条件不匹配 | y存在({y is not None}) ⇄ 模型类别条件({self.num_classes is not None})"
        )
        print(f"✓ 条件验证通过 | 使用类别条件: {self.num_classes is not None}")

        # 时间嵌入计算
        print("\n[时间嵌入] 生成时间步编码：")
        print(f"▷ 输入 timesteps 形状: {timesteps.shape}")
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        print(f"│ 基础编码形状: {t_emb.shape}")
        emb = self.time_embed(t_emb)
        print(f"└─ 投影后嵌入形状: {emb.shape}")

        # 类别条件融合
        if self.num_classes is not None:
            print("\n[类别融合] 合并类别条件信息：")
            print(f"▷ 输入标签 y 形状: {y.shape}")
            assert y.shape[0] == x.shape[0], (
                f"❌ Batch大小不匹配 | x({x.shape[0]}) ≠ y({y.shape[0]})"
            )
            print(f"✓ Batch大小验证通过: {x.shape[0]}")
            emb += self.label_emb(y)
            print(f"└─ 融合后嵌入形状: {emb.shape}")

        # 输入处理
        print("\n" + "-"*40)
        print("[编码阶段] 执行输入块处理：")
        h = x.type(self.dtype)
        print(f"▷ 初始输入形状: {h.shape} | dtype: {h.dtype}")
        
        hs = []
        for idx, module in enumerate(self.input_blocks):
            print(f"\n● 输入块 {idx+1}/{len(self.input_blocks)}")
            h = module(h, emb, context)
            print(f"│ 输出形状: {h.shape}")
            hs.append(h)
            print(f"└─ 特征图缓存数: {len(hs)}")

        # 中间处理
        print("\n" + "-"*40)
        print("[中间层] 执行中间块处理：")
        h = self.middle_block(h, emb, context)
        print(f"└─ 中间输出形状: {h.shape}")

        # 解码阶段
        print("\n" + "-"*40)
        print("[解码阶段] 执行输出块处理：")
        for idx, module in enumerate(self.output_blocks):
            print(f"\n● 输出块 {idx+1}/{len(self.output_blocks)}")
            print(f"▷ 当前特征形状: {h.shape}")
            skip_conn = hs.pop()
            print(f"│ 取出跳连特征: {skip_conn.shape} (剩余缓存: {len(hs)})")
            h = th.cat([h, skip_conn], dim=1)
            print(f"│ 拼接后形状: {h.shape}")
            h = module(h, emb, context)
            print(f"└─ 处理输出形状: {h.shape}")

        # 最终输出
        print("\n" + "-"*40)
        print("[输出转换] 生成最终结果：")
        h = h.type(x.dtype)
        print(f"▷ 恢复原始dtype: {h.dtype}")
        
        if self.predict_codebook_ids:
            print("→ 使用codebook预测路径")
            output = self.id_predictor(h)
        else:
            print("→ 使用常规输出路径")
            output = self.out(h)
        
        print(f"\n[最终输出] 形状: {output.shape}")
        print("="*50)
        print("[前向传播] 计算完成\n")
        return output

