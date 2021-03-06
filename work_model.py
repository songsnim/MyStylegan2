import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import upfirdn2d, conv2d_gradfix


class PixelNorm(nn.Module):
    """
    목적 : Adversarial learning이 날뛰는 것을 막는다.
    작동 : 각 conv block의 feature layer를 pixel 단위로 normalizing
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1: 
        # n x 1 짜리 벡터면 transpose랑 행렬곱해서 n x n 짜리 행렬(conv에서 쓰는 kernel이랑 같은 개념)로 만든다 
        k = k[None, :] * k[:, None]
    # 그 외에는 어차피 2차원 이상이니 그냥 전체 합으로 나눠서 사용한다.
    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input.half(), self.kernel.half(), pad=self.pad)

        return out

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.scale)

def fused_leaky_relu(input, bias=None, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale

class EqualConv2d(nn.Module):
    """
    목적 : 각 모듈이 동일한 학습속도 보장하는 Conv2d layer
    작동 : weight init은 N(0,1), 학습중에 동적으로 weight parameter scaling
    """
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        # print(f'conv2d_gradfix.conv2d input.type() is {input.type()}')
        if input.type() != 'torch.cuda.FloatTensor':
            input = input.to(torch.float32)
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2 # fan {in} = {in}put layer neuron 갯수 (여기선 conv block의 pixel 수(CxHxW))
        self.scale = 1 / math.sqrt(fan_in) # s_i에 해당한다
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1) # affine transformation

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        # Fusion 안 할 때 (My case)
        if not self.fused:
            # @@@@@@@@ Mod 준비@@@@@@@@
            style = self.modulation(style) # affine transformation
            weight = self.scale * self.weight.squeeze(0) # w'_ijk = s_i * w_ijk 구현!
            
            # @@@@@@@@ Demod 준비@@@@@@@@
            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1) # style로 conv weight scaling
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt() 
                # in_channel, kernel_size, kernel_size 방향을 다 sum
                # w''_ijk = w'_ijk / sqrt(i,j,k 방향 sum(w'ijk^2 + epsilon))에서 w'_ijk 뺀 부분(계수)
            
            # Mod (들어온 input에 style을 입힌다 일단.)
            input = input * style.reshape(batch, in_channel, 1, 1) 
            # input(constant거나 이전 bias/noise 결과물+leakyrelu)에다가 learnable style 곱해줌(feature map당 style 하나)
            
            # Style을 입힌 input을 conv에 넣을 때 학습되는 weight는 w'_ijk = s_i * w_ijk 이거임!(Mod)
            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)
            # upsample이든 downsample이든 하고 나면 직접 modulate한 한 weight를 사용하는 conv 사용 
            
            # Demod (style입힌 input을 style로 modulate한 weight를 갖는 conv에 넣은 결과를 demod)
            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)
                
            return out

        #========================아래는 안 돌아가는 부분=============================
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) 
        weight = self.scale * self.weight * style 

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        ) # style로 mod하고 demod를 모두 거친

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            # input image(feature map)과 같은 크기의 Gaussian noise를 생성
            # 모든 feature map에 동일한 weight로 scaled된 noise를 적용해야함
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1) # shape = (batch, channel, size, size)로 반환

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        # modulated conv 이후 noise를 더하고 bias를 왜 안 넣는지는 모르겠다.
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False) # 3은 rgb, 1은 kernel size
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style) # Style이 첨가된(당연히) 해당 resolution의 image를 rgb로 바꿈
        out = out + self.bias # 거기에 bias를 더해주어 학습이 용이하도록 함

        if skip is not None: # Skip을 해야되면(마지막 아니면 해야됨) upsample해서 
            skip = self.upsample(skip)

            out = out + skip

        return out

class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]
        # mapping network
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        
        #mapping network
        self.mapping_network = nn.Sequential(*layers)

        # 해상도와 channel
        if self.size == 1024:
            self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }
        elif self.size == 64:
            self.channels={
                4: 128,
                8: 128,
                16: 64 * channel_multiplier,
                32: 32 * channel_multiplier,
                64: 16 * channel_multiplier,
            }
        

        self.input = ConstantInput(self.channels[4])
        self.conv_init = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb_init = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2)) # size 가 4면 2, 16이면 4, 32면 5 
        self.layer_indices = (self.log_size - 2) * 2 + 1 # size가 4면 layer 수는 1, 8이면 3, 16이면 5, 32면 7

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.layer_indices):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1): # resolution마다 ModuleList(self.conv)에다가 StyleConv를 넣어줄건데
            out_channel = self.channels[2 ** i] 

            # upsample 하는거 안 하는 StyleConv를 번갈아가면서 넣을 예정
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        # 이미지 사이즈가 4,8,16,32면 n_latent는 0,1,2,3
        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device                          
        # 4x4 이미지용 noise를 생성
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        # resolution 하나씩 올려가면서 그에 맞는 해상도의 noise를 추가해줌
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    # truncation trick을 위한 latent의 center of mass(truncation latent) 구하기
    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.mapping_network(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.mapping_network(input)

    def forward(
        self,
        input_noise,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):  
        # input noise z를 latent로 매핑
        if not input_is_latent:
            # print(f"input noise.shape = {input_noise.shape}")
            styles = [self.mapping_network(s) for s in input_noise] # batch 별로 latent w 생성

        if noise is None:
            if randomize_noise:
                noise = [None] * self.layer_indices
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.layer_indices)
                ]

        # truncatation trick으로 raw한 style sampling이 아닌 truncated style sampling
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    #truncation_latent + truncation * (style - truncation_latent)
                    (1 - truncation)*truncation_latent + truncation*style
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent 

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
        
        # skip generator를 만들기 전 가장 초기 Style Conv를 initialize
        out = self.input(latent) # latent와 같은 batch를 같는 constant input 생성
        
        out = self.conv_init(out, latent[:, 0], noise=noise[0]) # constant input(out)과 initial style(latent[:,0])와 noise 넣어주기

        skip = self.to_rgb_init(out, latent[:, 1]) # skip하려는 결과는 이전 conv layer 출력과 두번째 style(latent[:,1])을 넣어줌

        i = 1 # 그 이후 skip generator 구조를 구현한 부분
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ): # self.convs ModuleList에서 짝수번쨰 module은 upsample 하는 conv, 홀수번째 module은 일반 conv
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if size == 1024:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }

        if size == 64:
            channels = {
                4: 128,
                8: 128,
                16: 64 * channel_multiplier,
                32: 32 * channel_multiplier,
                64: 16 * channel_multiplier,
            }
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2)) # 2배씩 channel size가 커지기 때문에 2^(지수)에서 지수를 얻으려고 함

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        """
        기법 : minibatch discriminator
        목적 : variation에 대한 학습 효과 증진
        작동 : 각 이미지와 minibatch에 대한 통계정보를 discriminator에 전달
               => 전체 minibatch에 대해 각 feature map의 spatial stddev를 계산
               => 구한 spatial stddev를 모든 feature에 대해 평균을 구한다
               => 해당 stddev의 평균값을 마지막 conv layer에 넣어준다.
        """
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

