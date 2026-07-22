# 水下图像 / 视频实时增强工作台

## 项目简介

这是一个基于 PySide6、PyTorch 和 TensorRT 的水下图像与视频增强桌面程序。程序可以加载图像、视频或摄像头画面，在统一界面中选择增强模型、推理后端和处理分辨率，并实时对比 Original 与 Enhanced 结果。

当前项目以 Python 3.12、PyTorch 2.8.0、CUDA 12.8 为目标环境，主要面向 NVIDIA GPU 推理。TensorRT engine 需要根据模型和输入尺寸提前生成，界面程序负责模型发现、权重匹配、推理、预览和结果保存。

## 项目功能

- 自动扫描 `models/` 中的模型入口文件。
- 支持 PyTorch、TensorRT FP16、TensorRT INT8。
- 支持 3840×2160、2560×1440、1920×1080、1280×720 和自定义分辨率。
- 支持图像、视频文件和摄像头 0。
- Original 与 Enhanced 双窗口同步预览。
- 视频开始播放前预缓存 30 帧，降低短时推理波动带来的卡顿。
- 推理速度较快时按源视频 FPS 播放，较慢时自动按实际处理能力显示。
- 摄像头只保留最新帧，避免实时画面不断累积延迟。
- 预览使用与界面窗口匹配的缩略图，保存仍使用完整分辨率结果。
- 显示 GPU 推理耗时、GPU 推理 FPS 和实际显示 FPS。
- 支持开始、暂停/继续和结束；结束后两个预览窗口恢复为等待界面。
- 支持将增强图像保存为 PNG、增强视频保存为 MP4。

## 项目结构

```text
underwater_realtime_enhancer/
├─ app.py                       # 程序入口
├─ config.py                    # 分辨率、后端和运行配置
├─ check_environment.py         # 环境检查
├─ model_template.py            # 模型入口模板
├─ requirements.txt
├─ README.md
├─ core/
│  ├─ backends.py               # PyTorch和TensorRT后端
│  ├─ model_registry.py         # 模型扫描与注册
│  ├─ weight_resolver.py        # 权重路径解析
│  └─ worker.py                 # 推理、预缓冲、统计和保存
├─ ui/
│  └─ main_window.py            # PySide6界面
├─ models/                      # 模型定义
├─ weights/
│  ├─ pth/                      # PyTorch权重
│  ├─ trt16/                    # TensorRT FP16 engine
│  └─ trt8/                     # TensorRT INT8 engine
└─ outputs/                     # 默认输出目录
```

## 环境要求

| 组件 | 建议或要求 |
| --- | --- |
| 操作系统 | Windows 10/11 |
| Python | 3.12 |
| GPU | 支持 CUDA 的 NVIDIA GPU |
| PyTorch | 2.8.0，CUDA 12.8 构建 |
| PySide6 | `>=6.7,<7` |
| OpenCV | `>=4.10,<5` |
| NumPy | `>=1.26,<3` |
| TensorRT | FP16/INT8模式需要，且必须与engine兼容 |

PyTorch 和 TensorRT 模式目前都需要 CUDA GPU。TensorRT 后端使用 PyTorch CUDA Tensor 作为 engine 缓冲区，因此只使用 TensorRT 时也需要安装 CUDA 版 PyTorch。

`requirements.txt` 不会自动安装 PyTorch 和 TensorRT，以免覆盖已经配置好的 CUDA 环境。

## 安装与启动

### 1. 打开项目

在 VS Code 中打开：

```text
D:\VSCode\underwater_realtime_enhancer
```

按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，切换到 Python 3.12 的 `UIE` 环境。

### 2. 安装界面依赖

```powershell
conda activate UIE
python -m pip install -r requirements.txt
```

也可以直接使用当前环境的解释器：

```powershell
D:\Anaconda\envs\UIE\python.exe -m pip install -r requirements.txt
```

### 3. 检查环境

```powershell
python check_environment.py
```

请确认 `CUDA available` 为 `True`，并且能够识别正确的 NVIDIA GPU。使用 TensorRT 时还应确认：

```powershell
python -c "import tensorrt as trt; print(trt.__version__)"
```

### 4. 启动程序

```powershell
python app.py
```

或者：

```powershell
D:\Anaconda\envs\UIE\python.exe app.py
```

也可以在 VS Code 中按 `F5`，选择“运行水下实时增强界面”。

## 添加模型

程序会扫描 `models/*.py`。模型文件必须提供 `create_model()`，建议同时声明模型名称、是否需要权重和输出范围。

可以复制 `model_template.py` 到 `models/`，例如：

```text
models/PUIENet.py
```

最小接口示例：

```python
from torch import nn


MODEL_NAME = "PUIENet"
REQUIRE_WEIGHTS = True
OUTPUT_RANGE = "0_1"


class PUIENet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 填写与训练权重一致的网络结构

    def forward(self, x):
        return x


def create_model() -> nn.Module:
    return PUIENet()
```

模型输入约定为 RGB、NCHW、范围 `[0, 1]`。`OUTPUT_RANGE` 支持：

- `0_1`：输出范围 `[0, 1]`；
- `minus1_1`：输出范围 `[-1, 1]`；
- `0_255`：输出范围 `[0, 255]`；
- `auto`：由程序自动判断。

如果模型使用特殊归一化、多输入、多输出或其他颜色空间，需要在 `core/backends.py` 中增加专用适配。

## 权重路径与命名

### PyTorch

PyTorch checkpoint 不区分输入分辨率，直接放在：

```text
weights/pth/模型名.pth
```

例如：

```text
weights/pth/EAFDNet.pth
```

程序支持普通 `state_dict`、常见 checkpoint 字典、直接保存的 `nn.Module`，并会自动移除 DataParallel 的 `module.` 前缀。

### TensorRT

TensorRT engine 通常绑定固定输入尺寸，需要按后端和分辨率分类：

```text
weights/trt16/宽x高/模型名_宽x高_trt16.engine
weights/trt8/宽x高/模型名_宽x高_trt8.engine
```

EAFDNet FP16 示例：

```text
weights/trt16/3840x2160/EAFDNet_3840x2160_trt16.engine
weights/trt16/2560x1440/EAFDNet_2560x1440_trt16.engine
weights/trt16/1920x1072/EAFDNet_1920x1072_trt16.engine
weights/trt16/1280x720/EAFDNet_1280x720_trt16.engine
```

文件名统一使用具体的 `宽x高`，不要使用 `4k`、`2k`、`1k` 或 `720p` 等简称。模型名称和大小写必须与 `MODEL_NAME` 一致。

TensorRT engine 应满足：

- 一个图像输入和一个图像输出；
- batch 为 1；
- 输入形状为 `[1, 3, H, W]`；
- 输入输出格式为 TensorRT `LINEAR`；
- 固定输入尺寸与界面选择完全一致；
- 推荐使用浮点输入输出，在 engine 内部完成 FP16 或 INT8 计算。

## 使用说明

1. 点击“选择图像 / 视频”，或者选择“使用摄像头 0”。
2. 选择增强模型和推理模式。
3. 选择处理分辨率。
4. 检查权重提示是否显示“已找到”。
5. 如需保存结果，勾选“保存增强结果”并选择目录。
6. 点击“开始增强”。
7. 运行中可以暂停、继续或结束。

### 使用 1920×1072 engine

1920×1072 不在标准分辨率列表中，需要选择：

```text
处理分辨率：其他 / 自定义
宽度：1920
高度：1072
```

程序会查找：

```text
weights/trt16/1920x1072/EAFDNet_1920x1072_trt16.engine
```

源视频分辨率不必与 engine 相同。例如 2K 或 4K 视频也可以使用 1920×1072 engine，程序会在推理前将源帧缩放到 1920×1072。engine 尺寸必须匹配界面选择的处理分辨率，而不是源视频分辨率。

## 预缓冲与显示策略

### 视频文件

- 点击开始后先缓存 30 帧，界面显示“正在预缓冲 n/30”。
- 缓存完成后，推理速度足够时按照源视频 FPS 显示。
- 推理速度不足时，自动按照实际处理能力降低显示速度。
- 少于 30 帧的短视频会在读取完成后播放已有缓存。
- 预缓冲只能平滑短时波动，不能提高模型的长期推理速度。

### 摄像头

摄像头不等待 30 帧，只保留最新结果。界面来不及显示时会丢弃旧帧，以减少实时延迟。

### 预览和保存

Original 与 Enhanced 预览会缩小到接近界面控件尺寸，并使用 `QImage.Format_BGR888` 直接显示BGR图像。保存路径仍接收完整分辨率结果，因此缩略预览不会降低输出质量。

## 性能指标

| 指标 | 含义 |
| --- | --- |
| GPU 推理耗时 | CUDA Event 测得的模型或engine执行时间 |
| GPU 推理FPS | `1000 / 平均GPU推理耗时(ms)` |
| 实际显示FPS | Enhanced新帧真正进入Qt绘制事件的帧率 |

统计规则：

- TensorRT视频和摄像头的前20帧作为预热，不参与统计，但仍正常显示和保存。
- 使用最近60帧滑动窗口。
- 指标每200 ms更新一次，显示一位小数。
- 单张图像没有连续播放意义，实际显示FPS可能保持为 `--`。

GPU推理FPS通常高于实际显示FPS。视频解码、resize、数据传输、输出转换、视频写入和Qt绘制都会影响最终显示速度。

## 结果保存

- 图像保存为 PNG。
- 视频和摄像头保存为 MP4。
- 文件使用时间戳命名，避免覆盖旧结果。

```text
enhanced_20260722_103000.png
enhanced_20260722_103000.mp4
```

输出使用模型完整分辨率。例如选择 3840×2160 engine 时，界面虽然显示缩略图，保存结果仍为 3840×2160。

当前视频使用 OpenCV `VideoWriter` 和 `mp4v` 编码，只保存画面，不保留原视频音轨。

## 注意事项

1. TensorRT engine 通常与 TensorRT、CUDA、操作系统和GPU架构相关，修改文件名不能解决兼容性问题。
2. 固定尺寸必须精确匹配，例如 1920×1072 和 1920×1080 不能互换。
3. 4K推理仍需要较高显存；缩略预览只降低显示开销，不降低模型推理显存。
4. 开启视频保存会增加编码和写盘负载，实际显示FPS可能下降。
5. 当前TensorRT通用后端只支持单输入、单输出和线性NCHW图像。
6. 当前摄像头编号固定为0，输出视频不包含声音。
7. 替换模型或engine前应先结束当前任务。

## 常见问题

### 模型没有出现在下拉框

确认模型文件位于 `models/`，没有以下划线开头，并且定义了 `create_model()`。模型依赖缺失或导入失败时，错误会显示在窗口底部状态栏。

### 找不到权重

按照界面显示的相对路径检查文件。重点确认模型名大小写、具体宽高、目录层级和 `_trt16/_trt8` 后缀。

### TensorRT engine 无法加载

通常是 TensorRT版本、CUDA、GPU架构、操作系统或插件环境不兼容。建议在目标电脑的当前环境中重新构建 engine。

### Enhanced黑屏、偏色或出现规则纹理

检查以下内容：

- engine 导出前是否加载正确训练权重；
- 训练、ONNX、TensorRT和界面是否统一使用 RGB、NCHW、`[0, 1]`；
- `OUTPUT_RANGE` 是否与模型真实输出一致；
- ONNX和TensorRT输出是否与PyTorch结果接近；
- engine 输入输出是否为 `LINEAR`。

### 点击开始后暂时没有画面

视频正在填充30帧缓存。4K或较慢模型需要更长等待时间；摄像头和单张图像不会执行30帧视频预缓冲。

### GPU FPS很高但画面仍然较慢

GPU FPS只统计模型执行时间。查看“实际显示FPS”，并考虑视频解码、前后处理、保存和Qt绘制开销。可以尝试降低处理分辨率、关闭结果保存或使用TensorRT FP16。

### CUDA显存不足

降低处理分辨率，关闭其他GPU程序，并避免同时启动多个界面实例。部分4K模型可能需要进一步实现分块推理。

---

项目将模型、权重解析、推理后端、媒体工作线程和Qt界面分开组织。新增符合通用输入输出约定的模型时，通常只需增加模型入口文件和对应权重。
