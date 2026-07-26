#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ==========================================
# 实时增强界面的模型注册信息
# ==========================================
# 使用独立名称，避免与标准版 EAFDNet 在模型下拉框和权重路径中冲突。
MODEL_NAME = "EAFDNet_small"

# PyTorch 模式需要加载 .pth；TensorRT 模式根据同一模型名查找 .engine。
REQUIRE_WEIGHTS = True

# 网络末层没有 Sigmoid，输出可能轻微超出 [0, 1]。显示前按 [0, 1]
# 清理和钳制，可避免少量越界像素导致整帧范围被误判而出现黑屏。
OUTPUT_RANGE = "0_1"


# ==========================================
# 工具函数：底层矩阵内存对齐
# ==========================================
def make_divisible(v, divisor=16):
    """
    确保通道数是 16 的倍数，完美触发 Jetson Tensor Core 硬件算力。
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


# ==========================================
# 1. 鲁棒性权值初始化工具
# ==========================================
@torch.no_grad()
def layerwise_init_weights(module, scale: float = 1.0):
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if any(ext in name for ext in ["pwconv2", "conv_out"]):
                init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.weight.data *= 0.1
            else:
                init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu"
                )
                m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


# ==========================================
# 2. AWBCM: 自适应白平衡矫正模块
# ==========================================
class AWBCM(nn.Module):
    """
    Adaptive White Balance Correction Module (AWBCM) - 工业级防溢出版
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

    def forward(self, x):
        identity = x
        
        # 1. 极致降采样：16x16 足够提取全局照明先验，极大降低访存开销
        x_down = F.adaptive_avg_pool2d(x, (16, 16))
        
        # 2. 统计量计算
        mean = x_down.mean(dim=(2, 3), keepdim=True)
        gray_mean = mean.mean(dim=1, keepdim=True)
        
        # 3. 增益计算与安全钳制 (防焦散/光斑过曝)
        gain = gray_mean / (mean + self.eps)
        gain = torch.clamp(gain, min=0.1, max=10.0) 
        
        # 4. 校正与残差加权
        x_wb = x * gain
        alpha = torch.sigmoid(self.alpha)
        
        return alpha * x_wb + (1 - alpha) * identity


# ==========================================
# 3. WDM: 小波分解模块 (回归本源极速版)
# ==========================================
class WDM(nn.Module):
    """
    Wavelet Decomposition Module (WDM)
    💡 回归最稳定的原生架构：不做任何切片或重排，保持底层的纯粹性，
    依靠下游的 index_select 进行极速内存抽取。
    """
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # 保持深度分离卷积的高效性，依然采用分组卷积核格式
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)

    def forward(self, x):
        # 深度分离小波卷积，输出天然交错排列，极速返回，不做任何张量重塑
        return F.conv2d(x, self.haar_kernel, stride=2, groups=self.channels)


# ==========================================
# 4. NCEM: 非线性对比度增强模块
# ==========================================
class NCEM(nn.Module):
    """
    Non-linear Contrast Enhancement Module (NCEM)
    💡 终极进化：使用空间级 Min-Max 归一化取代 Sigmoid，
    完美保留中灰阶线性对比度的同时，利用可学习参数精准提升暗部。
    """

    def __init__(self, channels, bias=False):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )
        
        # 💡 全局曲线拉伸振幅 (初始化为0.5，防初期震荡)
        self.curve_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        # 💡 独立暗部补偿系数 (初始化为0，交由网络自适应学习)
        self.dark_boost = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        m = self.local_conv(x)
        weight = self.att(m)
        
        # 💡 1. 动态 Min-Max 归一化 (防 FP16 溢出，使用 1e-4)
        # 计算空间维度上的极值，线性缩放至 [0, 1]，100% 无损保留相对对比度
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-4)
        
        # 💡 2. 物理对比度抛物线拉伸 
        curve = x_norm * (1.0 - x_norm)
        
        # 💡 3. 暗部先验捕捉 (最暗的地方 x_norm接近0，提供针对性极强的托底)
        dark_prior = (1.0 - x_norm) ** 2
        
        # 参数化多级增强融合，直接作用于原始张量
        enhancement = (self.curve_scale * curve) + (self.dark_boost * dark_prior)
        x_enhanced = x + weight * enhancement
        return x_enhanced


# ==========================================
# 5. 微观子块: IDS_FFN
# ==========================================
class IDS_FFN(nn.Module):
    """
    Isodimensional Separable Feed-Forward Network (IDS-FFN)
    作为微观子块，采用 ConvNeXt 风格的 3x3 深度滑窗 + 逆瓶颈前馈网络。
    """
    def __init__(self, dim, expansion=2.0, kernel_size=3, padding=1, bias=False):
        super().__init__()
        hidden_dim = make_divisible(dim * expansion, 16)

        # 1. 空间信息交互 (Depthwise)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=bias
        )
        self.norm = nn.BatchNorm2d(dim)
        
        # 2. 通道信息交互与非线性 (Pointwise FFN)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return out + identity


# ==========================================
# 6. IHFRB: 等维高频纹理恢复块 (重构为主容器)
# ==========================================
class IHFRB(nn.Module):
    """
    Isodimensional High-Frequency Restoration Block (IHFRB)
    高频轨主容器：内部串联 2 个 IDS-FFN 提取精细纹理。
    """
    def __init__(self, dim, num_blocks=2, expansion=2.0, bias=False):
        super().__init__()
        self.blocks = nn.Sequential(
            *[IDS_FFN(dim, expansion=expansion, kernel_size=3, padding=1, bias=bias) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


# ==========================================
# 7. ILFCB: 等维低频色彩校正块 (重构为主容器)
# ==========================================
class ILFCB(nn.Module):
    """
    Isodimensional Low-Frequency Chromatic Block (ILFCB)
    低频色彩轨主容器：摒弃单层 7x7 大核，改用 3 个 3x3 IDS-FFN 串联。
    💡 优势：在维持等效感受野的同时，利用高度优化的 3x3 算子提升并发速度，并增加网络深度。
    """
    def __init__(self, dim, num_blocks=3, expansion=2.0, bias=False):
        super().__init__()
        self.blocks = nn.Sequential(
            *[IDS_FFN(dim, expansion=expansion, kernel_size=3, padding=1, bias=bias) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


# ==========================================
# 8. CFM: 跨频特征搅拌器
# ==========================================
class CFM(nn.Module):
    """
    Cross-Frequency Mixer (CFM)
    双分支轴向拼接后，使用 1x1 纯矩阵卷积在底层瞬间打通高低频通道的信息壁垒。
    """

    def __init__(self, channels, bias=False):
        super().__init__()
        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.mix_conv(x)


# ==========================================
# 9. 主网络：EAFDNet (流线架构升级版)
# ==========================================
class EAFDNet(nn.Module):
    """
    Efficient Asymmetric Frequency-Decoupled Network (EAFDNet) 高效非对称频域解耦网络

    🛠️ 架构重构：
    采用高度模块化的平行单轨直通设计，上层由 IHFRB 提取高频，下层由 ILFCB 和 NCEM 处理低频色彩。
    """

    def __init__(self, inp_channels=3, out_channels=3, n_feat=32, bias=False):
        super(EAFDNet, self).__init__()

        self.c = make_divisible(n_feat, 16)

        # 💡 【核心配置】：非对称分配比例
        self.c_up = 16
        self.c_low = self.c - self.c_up  

        # 前端双级串联小波降维
        self.awbcm = AWBCM()
        self.wdm_lvl1 = WDM(channels=inp_channels)  # 3ch -> 12ch
        self.wdm_lvl2 = WDM(channels=inp_channels * 4)  # 12ch -> 48ch

        # 静态注册频域索引 
        self.register_buffer("low_idx", torch.tensor([0, 16, 32], dtype=torch.long))
        high_idx_list = [i for i in range(48) if i not in [0, 16, 32]]
        self.register_buffer("high_idx", torch.tensor(high_idx_list, dtype=torch.long))

       
        self.conv_in_low = nn.Conv2d(
            3, self.c_low, kernel_size=1, bias=True
        )  # 3ch -> 48ch
        self.conv_in_high = nn.Conv2d(
            45, self.c_up, kernel_size=1, bias=True
        )  # 45ch -> 16ch

        # 💡 并行独立特征恢复容器 (彻底摒弃复杂的交叉 Stage 级联)
        self.upper_branch = IHFRB(self.c_up, num_blocks=2, expansion=2.0, bias=bias)
        self.lower_branch = ILFCB(self.c_low, num_blocks=2, expansion=2.0, bias=bias)
        self.ncem = NCEM(self.c_low, bias=bias)  # 挂载于 48ch 宽轨上的自适应拉伸
        
        # 唯一的跨频特征搅拌器
        self.cfm = CFM(self.c, bias=bias)

        # 亚像素 4x 大跨度重建
        self.conv_out = nn.Conv2d(
            self.c, out_channels * 16, kernel_size=3, padding=1, bias=True
        )
        self.shuffle_out = nn.PixelShuffle(4)
        

        layerwise_init_weights(self)

    def forward(self, x):
        # 1. 前端物理色偏矫正
        x_corrected = self.awbcm(x)

        # 2. 级联小波空间解算 (3ch -> 12ch -> 48ch)
        x_freq_l1 = self.wdm_lvl1(x_corrected)
        x_freq_l2 = self.wdm_lvl2(x_freq_l1)

        # 3. 静态抽取，源头解耦
        x_low_raw = torch.index_select(x_freq_l2, 1, self.low_idx)
        x_high_raw = torch.index_select(x_freq_l2, 1, self.high_idx)

        # 4. 非对称投影映射
        f_in_low = self.conv_in_low(x_low_raw)
        f_in_high = self.conv_in_high(x_high_raw)
        
        # 5. 💡 并行独立特征恢复：单轨直通流线设计
        u_out = self.upper_branch(f_in_high)
        l_out = self.lower_branch(f_in_low)
        l_out = self.ncem(l_out)
        
        # 6. 唯一的跨频融合搅拌
        fuse = self.cfm(torch.cat([u_out, l_out], dim=1))

        # 7. 4倍亚像素大跨度超速重建 back to 4K
        y = self.conv_out(fuse)
        out = self.shuffle_out(y)
        
        return out


# ==========================================
# 实时增强界面要求的模型工厂函数
# ==========================================
def create_model():
    """使用轻量版默认配置 n_feat=32 创建模型。"""
    return EAFDNet(n_feat=32)


# ==========================================
# 10. 4K极限性能与复杂度统计基准测试
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  EAFDNet (极简流线架构完全体) 基准性能与开销审计 ")
    print("=" * 60)

    CHANNELS_CONFIG = 64

    model = EAFDNet(n_feat=CHANNELS_CONFIG).cuda().half()
    dummy_input_4k = torch.randn(1, 3, 2160, 3840).cuda().half()

    print(f"模型总通道配置 ：n_feat = {CHANNELS_CONFIG}")
    print(f"├─ 上分支 (高频轨) = {model.c_up}ch，内嵌 2 个 IDS-FFN 微块")
    print(f"└─ 下分支 (低频轨) = {model.c_low}ch，内嵌 3 个 IDS-FFN 微块，接 NCEM 物理增强")
    print(
        f"模型总参数量   ：{sum(p.numel() for p in model.parameters()) / 1e6:.4f} M"
    )

    try:
        from thop import profile

        model_f32 = EAFDNet(n_feat=CHANNELS_CONFIG).cuda().float()
        input_f32 = torch.randn(1, 3, 2160, 3840).cuda().float()

        print("\n正在通过 THOP 分析 4K 分辨率下的网络操作拓扑图...")
        macs, params = profile(model_f32, inputs=(input_f32,), verbose=False)
        actual_flops = macs * 2

        print(f"┌── [4K 复杂度审计] ──────────────────────────")
        print(f"├── 乘加数 (MACs) : {macs / 1e9:.2f} G MACs (Giga Operations)")
        print(
            f"├── 计算量 (FLOPs): {actual_flops / 1e9:.2f} G FLOPs (Floating Point Ops)"
        )
        print(
            f"└── 注：1 MACs 包含一次乘法与一次加法，通常约对应 2 FLOPs 运算吞吐量"
        )
        del model_f32, input_f32
    except ImportError:
        print(
            "\n[提示]: 未检测到 thop 库。请执行 `pip install thop` 开启 4K FLOPs 深度耗时审计。"
        )

    print("\n正在启动 GPU 硬件预热线 (10次推理冷启动剔除)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input_4k)

        torch.cuda.synchronize()
        start_time = time.time()

        output_4k = model(dummy_input_4k)

        torch.cuda.synchronize()
        end_time = time.time()

    print("\n┌── [4K 运行帧率审计] ────────────────────────")
    print(f"├── 4K 输入相机尺寸 : {dummy_input_4k.shape[2:]}")
    print(f"├── 4K 增强输出尺寸 : {output_4k.shape[2:]}")
    print(f"├── 纯原生态单帧延时: {(end_time - start_time) * 1000:.2f} ms")
    print(f"└── 未量化极限帧率  : {1.0 / (end_time - start_time):.1f} FPS")
    print("=" * 60)
