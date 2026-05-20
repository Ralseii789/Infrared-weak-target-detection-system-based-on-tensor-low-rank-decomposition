import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 高效通道注意力
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


#频域细化
class FFParser(nn.Module):
    def __init__(self, dim, h=256, w=129):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spatial_size=None):
        r = x
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size
        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)
        out = x + r

        return out



class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


def eta_l1(r_, lam_):
    B, C, H, W = r_.shape
    lam_ = torch.reshape(lam_, [1, C, 1, 1])
    lam_ = lam_.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R



class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv1_Wdz = self._make_layer(n, c, 2)
        self.conv1_Wdtz = self._make_layer(c, n, 2)
        self.random_lowrank = RandomizedLowRankChannelMix(n, rank=n//4)   # 新增
        self.attenton_1 = ECA(n)

    def _make_layer(self, input_channels, output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x, tensor_z, lam_theta, lam_z):
        convZ1 = self.conv1_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z * tensor_z + self.conv1_Wdtz(midZ)
        tensor_c = self.random_lowrank(tensor_c)   # 随机化低秩处理
        tensor_c = self.attenton_1(tensor_c)
        Z = eta_l1(tensor_c, lam_theta)
        return Z


class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2 * n, s, stride)
        # 小阈值初始化利于 LISTA 式迭代稳定收敛，避免初始过度截断
        self.lamj = nn.Parameter(torch.rand(1, self.n * 2) * 0.04 + 0.02)
        self.lamz = nn.Parameter(torch.rand(1, 1) * 0.15 + 0.35)
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))

    def _make_layer(self, input_channels, output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)
        tensor_z = eta_l1(tensor_l, self.lamj)
        S_history = []
        for i in range(self.num_block):
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_z = lrrblock(x, tensor_z, self.lamj, self.lamz)
            S_current = tensor_z[:, self.n: 2 * self.n, :, :]
            S_history.append(S_current)
        L = tensor_z[:, :self.n // 2, :, :]
        D = tensor_z[:, self.n // 2: 1 * self.n, :, :]
        S = tensor_z[:, self.n: 2 * self.n, :, :]
        return L, S, D, S_history


class PAFM(nn.Module):
    def __init__(self, channels_high, out_channels):
        super(PAFM, self).__init__()
        self.dconv = nn.ConvTranspose2d(channels_high, channels_high, 3, 2, 1)
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_high, channels_high // 4, 1),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, 1),
            nn.BatchNorm2d(channels_high)
        )
        self.AAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels_high, channels_high // 4, 1),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, 1),
            nn.BatchNorm2d(channels_high)
        )
        self.active = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_high, channels_high // 4, 1),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, 1),
            nn.BatchNorm2d(channels_high),
            nn.Sigmoid()
        )

    def forward(self, x_high, x):
        _, _, h, w = x_high.shape
        x_sum = x_high + x
        x_conv = self.conv(x_sum)
        p_x_high = self.active((self.GAP(x_high) + self.AAP(x_high)))
        p_x_high = F.interpolate(p_x_high, scale_factor=h // 4, mode='nearest')
        output = ((x_conv) * x_sum) * p_x_high
        return output


#MFF模块+通道注意力强化
class new_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(new_fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)


        self.fuse = PAFM(out_channels, out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(kernel_size=3)


        #简化版通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )


    def forward(self, x, r, mask=None):

        # 原始路径
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)

        # 通道注意力强化
        attn_out = x_conv
        att_weight = self.channel_att(attn_out)
        attn_out = attn_out * att_weight + attn_out
        fuse_out = self.fuse(x_conv, r)
        fuse_out = self.ca(fuse_out) * fuse_out
        fuse_out = self.sa(fuse_out) * fuse_out
        out_original = fuse_out + x_conv

        out = out_original + attn_out
        out = self.relu(out)
        return out


#多方向对比度增强模块
class FENET(nn.Module):
    def __init__(self):
        super(FENET, self).__init__()
        self.directions = [((0, -2), (0, 2)), ((-2, 0), (2, 0)), ((-2, -2), (2, 2)), ((-2, 2), (2, -2))]
        self.alpha = nn.Parameter(torch.ones(4) * 0.1)
        self.norm = nn.Sigmoid()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

    def get_shifted(self, x, dy, dx):
        pad = 2
        padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        shifted = padded[:, :, pad + dy:pad + dy + x.shape[2], pad + dx:pad + dx + x.shape[3]]
        return shifted

    def forward(self, x):
        B, C, H, W = x.shape
        contrast_maps = []
        for idx, ((dy1, dx1), (dy2, dx2)) in enumerate(self.directions):
            shifted1 = self.get_shifted(x, dy1, dx1)
            shifted2 = self.get_shifted(x, dy2, dx2)
            diff1 = x - shifted1
            diff2 = x - shifted2
            contrast = self.alpha[idx] * (diff1 * diff2)
            contrast_maps.append(contrast)
        R = torch.stack(contrast_maps, dim=0).sum(dim=0)
        R = self.norm(R)
        alpha_global = torch.clamp(self.alpha.mean(), 0, 0.5)
        out = x + alpha_global * x * R
        return out
class RandomizedLowRankChannelMix(nn.Module):
    """随机化低秩通道混合：使用随机投影近似低秩变换，加速并增强 LISTA 迭代"""
    def __init__(self, channels, rank=None, fix_random=False):
        super().__init__()
        r = rank if rank is not None else max(4, channels // 8)
        r = min(max(1, r), channels)
        # 随机投影矩阵（固定随机性可复现，也可不固定以引入多样性）
        if fix_random:
            self.register_buffer('rand_proj', torch.randn(channels, r) / (r ** 0.5))
        else:
            self.rand_proj = nn.Parameter(torch.randn(channels, r) / (r ** 0.5), requires_grad=False)
        self.up_proj = nn.Parameter(torch.randn(r, channels) / (r ** 0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 通道降维：随机投影
        x_flat = x.permute(0,2,3,1) @ self.rand_proj  # (B,H,W,r)
        x_flat = F.relu(x_flat)
        # 升维回原通道
        x_up = x_flat @ self.up_proj  # (B,H,W,C)
        x_up = x_up.permute(0,3,1,2)  # (B,C,H,W)
        return x + self.gamma * x_up

# 主网络
class LTLNET(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, dataset_name='NUDT', mode='test'):
        super(LTLNET, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv0 = self._make_layer(block, input_channels, 16)
        self.conv1 = self._make_layer(block, 16, 32, 2)
        self.convR = self._make_layer(block, 32, 16)
        if dataset_name == 'NUDT':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
        elif dataset_name == 'NUAA':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 4)
            self.new_fusion = new_fusion(112, 16)
        elif dataset_name == 'IRSTD':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
        else:
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)

        self.contrast_enhance = FENET()
        self.FFP = FFParser(16)

        self.attenton_0 = ECA(16)
        self.attenton_1 = ECA(32)
        self.final_z = nn.Conv2d(16, num_classes, kernel_size=1)
        self.final_d = nn.Conv2d(8, num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input, mask=None):
        x0 = self.conv0(input)                        # 浅层             # 细节特征
        L, S, D, S_history = self.get_ls(x0)

        S_FENET = self.contrast_enhance(S)
        FS = self.FFP(S_FENET)

        FS_history = []
        for s in S_history:
            s_FENET = self.contrast_enhance(s)
            FS_history.append(self.FFP(s_FENET))

        x1 = self.conv1(self.pool(self.attenton_0(FS)))
        z1 = self.attenton_1(x1)
        z2 = self.up(z1)
        z3 = self.convR(z2)
        z0 = torch.cat([*FS_history, FS, z2], 1)
        z0 = self.new_fusion(z0, z3)
        if self.mode == 'train':
            output = self.final_z(z0).sigmoid()
            output_S = self.final_z(FS).sigmoid()
            output_L = self.final_d(L).sigmoid()
            output_D = self.final_d(D).sigmoid()
            out_FS_history = [self.final_z(fs).sigmoid() for fs in FS_history]
            return output, output_L, output_S, output_D, out_FS_history
        else:
            output = self.final_z(z0).sigmoid()
            return output