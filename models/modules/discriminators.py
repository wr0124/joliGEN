import functools
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .utils import spectral_norm, normal_init

torch.autograd.set_detect_anomaly(True)

from .unet_generator_attn.unet_discriminator_attn import (
    AttentionBlock,
    ResBlock,
    EmbedSequential,
    Downsample,
    Upsample,
    normalization,
    zero_module,
)
from .unet_generator_attn.unet_discriminator_attn import UNet as UNet_discriminator_mha


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        use_spectral=False,
        freq_space=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            use_dropout (bool) -- whether to use dropout layers
            use_spectral (bool) -- whether to use spectral norm
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.freq_space = freq_space
        if self.freq_space:
            from .freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(input_nc)
            self.dwt = HaarTransform(input_nc)
            input_nc *= 4

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                use_spectral,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            if use_dropout:
                sequence += [nn.Dropout(0.5)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                use_spectral,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        sequence += [
            spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                use_spectral,
            )
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        if self.freq_space:
            x = self.dwt(input)
        else:
            x = input
        x = self.model(x)
        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class UnetDiscriminator(nn.Module):
    """Create a Unet-based discriminator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        D_num_downs,
        D_ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            D_num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            D_ngf (int)       -- the number of filters in the last conv layer, here  ngf=64, so inner_nc=64*8=512
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetDiscriminator, self).__init__()
        # construct unet structure
        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 8,
            D_ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        # add intermediate layers with ngf * 8 filters
        for i in range(D_num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                D_ngf * 8,
                D_ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 4,
            D_ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 2,
            D_ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            D_ngf, D_ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )

        # add the outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc,
            D_ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def compute_feats(self, input, extract_layer_ids=[]):
        output, feats, output_encoder_inside = self.model(input, feats=[])
        return_feats = []
        for i, feat in enumerate(feats):
            if i in extract_layer_ids:
                return_feats.append(feat)

        return output, return_feats, output_encoder_inside

    def forward(self, input):
        output, _, output_encoder_inside = self.compute_feats(input)
        return output, output_encoder_inside


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.bottleneck_conv_cor2 = nn.Conv2d(
            inner_nc, outer_nc, kernel_size=2, stride=1, padding=0, bias=True
        )
        self.bottleneck_conv_cor1 = nn.Conv2d(
            inner_nc, outer_nc, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x, feats, output_encoder_inside=None):
        output = self.model[0](x)
        return_feats = feats + [output]

        for layer in self.model[1:]:

            if isinstance(layer, UnetSkipConnectionBlock):
                output, return_feats, output_encoder_inside = layer(
                    output, return_feats, output_encoder_inside=output_encoder_inside
                )
            else:
                output = layer(output)
            if self.innermost and isinstance(layer, nn.ReLU):
                output_encoder = output
                if output_encoder.shape[2] == 2:
                    output_encoder_conv = self.bottleneck_conv_cor2(output_encoder)
                else:
                    output_encoder_conv = self.bottleneck_conv_cor1(output_encoder)

                output_encoder_inside = self.tanh(output_encoder_conv)

        if not self.outermost:  # add skip connections
            output = torch.cat([x, output], 1)

        return output, return_feats, output_encoder_inside


class UNet_discriminator_mha(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
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
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        tanh,
        n_timestep_train,
        n_timestep_test,
        norm,
        group_norm_size,
        cond_embed_dim,
        dropout=0,
        channel_mults=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        efficient=False,
        freq_space=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.freq_space = freq_space

        if self.freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
            in_channel *= 4
            out_channel *= 4

        if norm == "groupnorm":
            norm = norm + str(group_norm_size)

        self.cond_embed_dim = cond_embed_dim

        ch = input_ch = int(channel_mults[0] * self.inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.cond_embed_dim,
                        dropout,
                        out_channel=int(mult * self.inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        efficient=efficient,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(mult * self.inner_channel)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            self.cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm=norm,
                            efficient=efficient,
                            freq_space=self.freq_space,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            freq_space=self.freq_space,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                self.cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm=norm,
                efficient=efficient,
                freq_space=self.freq_space,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm=norm,
                efficient=efficient,
                freq_space=self.freq_space,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.cond_embed_dim,
                        dropout,
                        out_channel=int(self.inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        efficient=efficient,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(self.inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm=norm,
                            efficient=efficient,
                            freq_space=self.freq_space,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            freq_space=self.freq_space,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        if tanh:
            self.out = nn.Sequential(
                normalization(ch, norm),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
                nn.Tanh(),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch, norm),
                torch.nn.SiLU(),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
            )

        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": n_timestep_train,
                "linear_start": 1e-6,
                "linear_end": 0.01,
            },
            "test": {
                "schedule": "linear",
                "n_timestep": n_timestep_test,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        }

    def compute_feats(self, input, embed_gammas):
        if embed_gammas is None:
            # Only for GAN
            b = (input.shape[0], self.cond_embed_dim)
            embed_gammas = torch.ones(b).to(input.device)

        emb = embed_gammas

        hs = []

        h = input.type(torch.float32)

        if self.freq_space:
            h = self.dwt(h)

        for module in self.input_blocks:

            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        outh_encoder = nn.Tanh()(h)
        outs, feats = h, hs
        return outs, feats, emb, outh_encoder

    def forward(self, input, embed_gammas=None):
        h, hs, emb, outh_encoder = self.compute_feats(input, embed_gammas=embed_gammas)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(input.dtype)
        outh = self.out(h)

        if self.freq_space:
            outh = self.iwt(outh)

        return outh, outh_encoder

    def get_feats(self, input, extract_layer_ids):
        _, hs, _ = self.compute_feats(input, embed_gammas=None)
        feats = []

        for i, feat in enumerate(hs):
            if i in extract_layer_ids:
                feats.append(feat)

        return feats

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
