# 1. 介绍

## 1.1 任务

针对现有红外弱小目标检测技术在复杂干扰环境下虚警率高、实时性差的问题，本项目提出一种基于深度张量低秩分解的智能检测系统。传统的低 秩分解算法虽然理论严谨，但计算量大，无法满足实时系统需求。

本项目利用深度神经网络强大的拟合能力来模拟和加速张量分解过程，通过学习背景的低秩表征和目标的稀疏特征，在保留数学模型物理意义的同时，大幅提升检测速度和鲁棒性。

主要任务：设计一个端到端的神经网络，模拟张量恢复过程，输出层直接分离出低秩背景张量和稀疏目标张量；优化网络结构，确保在抑制背景的同时保留目标细节；开发基于Python+QT的检测系统界面，并能直观展示原始图像、分离后的背景图及检测到的目标图。

## 1.2 基础模型LTL-Net

```
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

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
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
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

#低秩分解
class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2*n, s, stride)
        self.lamj = nn.Parameter(torch.rand(1, self.n*2))  # l1-norm
        self.lamz = nn.Parameter(torch.rand(1, 1))
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))
    def _make_layer(self, input_channels,  output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)  # Z_0
        tensor_z = eta_l1(tensor_l, self.lamj)
        S_history = []
        for i in range(self.num_block):
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_z = lrrblock(x, tensor_z, self.lamj, self.lamz)
            # 提取当前稀疏分量并保存
            S_current = tensor_z[:, self.n: 2 * self.n, :, :]
            S_history.append(S_current)
        L = tensor_z[:, :self.n//2, :, :]  # 前n/2个通道 -> L
        D = tensor_z[:, self.n//2: 1 * self.n, :, :]  # n/2到n个通道 -> D
        S = tensor_z[:, self.n: 2 * self.n, :, :]  # 后n个通道 -> S
        return L, S, D, S_history

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

class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv1_Wdz= self._make_layer(n, c, 2)
        self.conv1_Wdtz = self._make_layer(c, n, 2)
        self.attenton_1 = ECA(48)
    def _make_layer(self, input_channels,  output_channels, num_blocks=1):
        block = Res_CBAM_block
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def forward(self, x, tensor_z, lam_theta, lam_z):
        convZ1 = self.conv1_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z*tensor_z + self.conv1_Wdtz(midZ)
        tensor_c = self.attenton_1(tensor_c)
        Z = eta_l1(tensor_c, lam_theta)
        return Z

def eta_l1(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    B, C, H, W = r_.shape
    lam_ = torch.reshape(lam_, [1, C, 1, 1])
    lam_ = lam_.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R


class LTLNET(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, dataset_name=None, mode='test'):
        super(LTLNET, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv0 = self._make_layer(block, input_channels, 16)
        self.conv1 = self._make_layer(block, 16, 32, 2)
        self.conv2 = self._make_layer(block, 128, 16)
        self.convR = self._make_layer(block, 32, 16)
        if dataset_name == 'NUDT':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
            # self.new_fusion = new_fusion(112, 16)
        elif dataset_name == 'NUAA':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 4)
            self.new_fusion = new_fusion(112, 16)
        elif dataset_name == 'IRSTD':
            self.get_ls = GetLS_Net(3, 16, 16, 1, 6)
            self.new_fusion = new_fusion(144, 16)
        else:
            self.get_ls = GetLS_Net(3, 16, 16, 1, 5)
            self.new_fusion = new_fusion(128, 16)
        self.FFP = FFParser(16, 256, 129)
        self.attenton_0 = ECA(16)
        self.attenton_1 = ECA(32)
        self.final_z = nn.Conv2d(16, num_classes, kernel_size=1)
        self.final_d = nn.Conv2d(8, num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0 = self.conv0(input)
        L, S, D, S_history = self.get_ls(x0)
        FS = self.FFP(S)
        FS_history = [self.FFP(s) for s in S_history]

        x1 = self.conv1(self.pool(self.attenton_0(FS)))
        z1 = self.attenton_1(x1)
        z2 = self.up(z1)
        z3 = self.convR(z2)
        z0 = torch.cat([*FS_history, FS, z2], 1)
        z0 = self.new_fusion(z0,z3)

        if self.mode == 'train':
            output = self.final_z(z0).sigmoid()
            output_S = self.final_z(FS).sigmoid()
            output_L = self.final_d(L).sigmoid()
            output_D = self.final_d(D).sigmoid()
            out_FS_history = [self.final_z(fs).sigmoid() for fs in FS_history]
            return output, output_L, output_S, output_D, out_FS_history
        elif self.mode == 'test':
            output = self.final_z(z0).sigmoid()
            return output

class PAFM(nn.Module):
    def __init__(self, channels_high, out_channels ):
        super(PAFM, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels=channels_high, out_channels=channels_high, kernel_size=3, stride=2, padding=1)
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_high, channels_high // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high)
        )
        self.AAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels_high, channels_high // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_high // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_high // 4, channels_high, kernel_size=1, stride=1, padding=0),
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
    def forward(self, x_high,x):
        _, _, h, w = x_high.shape
        x_sum = x_high +  x
        x_conv = self.conv(x_sum)
        p_x_high = self.active((self.GAP(x_high) + self.AAP(x_high)))
        p_x_high =  F.interpolate(p_x_high, scale_factor=h // 4, mode='nearest')
        output = ((x_conv) * x_sum) * p_x_high

        return output
class new_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(new_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        self.fuse = PAFM(16,16)
    def forward(self, x,r):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.fuse(x,r)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += x
        out = self.relu(out)
        return out

```

# 2. 模型改进

## 2.1 多方向对比度增强模块（FENET）

参照EFN中的FENET模块，在细化频域之前，对拆分出来的目标分量Tk做多方向的灰度对比增强。具体来看，在-45°,0°,45°,90°四个方向上测算灰度对比度：

 ![image-20260509122424465](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260509122424465.png)

这里  W1=1,   W2=0,   W3=−1， Pj 代表中心像素的数值。我们把四个方向的计算结果融合起来，就能得到用于增强的权重图R，再和原始的目标特征做乘法运算：

 ![image-20260509122454937](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260509122454937.png)

强化低信噪比下目标的微弱特征。

```
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
```

## 2.2 new_fusion模块改进

### 2.2.1 通道注意力强化

完成融合后，加入简化版CBAM通道注意力，仅保留通道注意力分支，将焦点放在与目标相关的通道上：

![image-20260509122741484](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260509122741484.png)                 





# 3. 其余改进

## 3.1 Loss重构损失函数

## 3.2 数据增强

## 3.3 不同大小图片的自适应

# 4. 实验结果

## 4.1 指标

### 4.1.1 PixAcc-像素准确率

定义：指所有被正确分类的像素所占的比例：
$$
Pixel Accuracy = \sum_{i}p_{ii}/\sum_{i}\sum_{j}p_{ij}
$$
其中，pij表示被标注为类i，但被预测为类j的像素数量。pii是预测正确的像素数量。分母是总像素数。

### 4.1.2 Mean IU-平均交并比

定义：每个类别的交并比（IoU）的平均值，是语义分割中最常用的评价指标之一。
$$
mIoU = 1/k·\sum_{i=1}^{k}[M_{ii}/(\sum_{j}M_{ij}+\sum_{j}M_{ji}-M_{ii})]
$$

$$
mIoU = 1/k·[正确预测为i/(真实为i+预测为i-正确预测为i)]
$$

分母是类i的并集 = 标注为i的像素总数+被预测为i的像素总数-交集。计算每个类别的IoU,然后取平均。

### 4.1.3 PD-检测率

定义：真实目标中被正确检测出来的比例，反映召回能力，即“找得全”：
$$
PD = 正确检测到的目标个数/真实目标总个数
$$


### 4.1.4 FA-虚警率

定义：检测结果中误报为目标的像素占总像素的比例，反映误检程度，即“错的少”：

$$
FA = 误报为前景的像素总数/测试集所有图像的总像素数
$$

## 4.2 结果

原：

|        |   NUAA   |   NUDT   |  IRSTD  |
| :----: | :------: | :------: | :-----: |
| pixAcc |  0.8835  |  0.9721  |  0.787  |
|  mIoU  |  0.7924  |  0.9568  | 0.6894  |
|   PD   |   1.0    |  0.9937  | 0.9349  |
|   FA   | 1.05e-05 | 5.29e-07 | 8.2e-06 |

改：

|        |   NUAA   |   NUDT   |  IRSTD   |
| :----: | :------: | :------: | :------: |
| pixAcc |  0.9039  |  0.9713  |  0.8288  |
|  mIoU  |  0.7867  |  0.9496  |  0.6624  |
|   PD   |   1.0    |  0.9936  |  0.9208  |
|   FA   | 6.94e-06 | 3.26e-06 | 4.33e-06 |

终：

|        |   NUAA   |   NUDT   |  IRSTD   |
| :----: | :------: | :------: | :------: |
| pixAcc |  0.9117  |  0.9713  |  0.8288  |
|  mIoU  |  0.7735  |  0.9493  |  0.6641  |
|   PD   |   1.0    |  0.9936  |  0.9184  |
|   FA   | 8.46e-06 | 1.38e-06 | 4.33e-06 |

# 5. 图形化界面

![Snipaste_2026-04-13_15-43-08](D:\desk\毕设\图片\Snipaste_2026-04-13_15-43-08.png)

![Snipaste_2026-04-13_15-46-14](D:\desk\毕设\图片\Snipaste_2026-04-13_15-46-14.png)

![Snipaste_2026-04-13_15-51-38](D:\desk\毕设\图片\Snipaste_2026-04-13_15-51-38.png)

![Snipaste_2026-04-13_15-44-05](D:\desk\毕设\图片\Snipaste_2026-04-13_15-44-05.png)
