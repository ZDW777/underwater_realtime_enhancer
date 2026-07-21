# 水下图像 / 视频实时增强界面

这是一个面向 NVIDIA GPU 的 PySide6 桌面程序，可在同一界面中切换水下增强模型、PyTorch / TensorRT 推理模式和处理分辨率，并实时对比原始画面与增强画面。

程序按你的环境设计：Python 3.12、PyTorch 2.8.0、CUDA 12.8。项目自带 `Passthrough` 无权重模型，可先用它检查界面、CUDA、图像和视频流程；它不会改变画面。

本版本统一采用具体分辨率命名，并在每种推理后端目录下按分辨率分类权重。项目中的主要模块均配有中文注释，方便在 VS Code 中继续修改。

## 已实现功能

- 自动扫描 `models/` 下的模型定义文件，模型下拉框不需要手工修改界面代码。
- 三种推理模式：PyTorch、TensorRT FP16、TensorRT INT8。
- 四种标准分辨率：3840×2160、2560×1440、1920×1080、1280×720。
- 自定义分辨率，可输入宽和高；TensorRT 会检查 engine 固定输入尺寸是否一致。
- 图像、视频文件和摄像头 0 三类输入源。
- 左右两个大尺寸实时预览窗口，分别标记 `Original` 和 `Enhanced`。
- 实时显示 GPU 纯推理耗时、纯推理 FPS、包含预处理和后处理的端到端 FPS。
- 开始、暂停/继续、结束控制。暂停不会销毁当前模型和视频位置。
- 可选保存增强结果：图像保存为 PNG，视频保存为 MP4。
- 当前模型、推理模式、处理分辨率、预期权重文件名和运行状态均在界面中显示。
- 深海渐变、矢量海浪和气泡组成的海洋主题界面，不依赖外部图片或网络资源。
- 输入源、模型配置、结果保存、性能指标、双画面和运行控制分区排布。
- CUDA 与依赖环境自检脚本；VS Code 一键调试配置。
- 工作线程执行模型与媒体处理，避免推理时冻结界面。

## 项目结构

```text
underwater_realtime_enhancer/
├─ app.py                       # 程序入口
├─ config.py                    # 分辨率、后端和运行配置
├─ check_environment.py         # 环境检查
├─ model_template.py            # 新模型定义模板
├─ requirements.txt
├─ .vscode/launch.json
├─ core/
│  ├─ backends.py               # PyTorch / TensorRT 后端
│  ├─ model_registry.py         # 自动发现模型定义
│  ├─ weight_resolver.py        # 权重命名与路径解析
│  └─ worker.py                 # 图像、视频、摄像头工作线程
├─ ui/main_window.py            # PySide6 主界面
├─ models/
│  └─ Passthrough.py            # 无权重流程测试模型
├─ weights/
│  ├─ pth/                      # PyTorch 权重：模型名.pth
│  ├─ trt16/                    # TensorRT FP16，下一层按分辨率分类
│  └─ trt8/                     # TensorRT INT8，下一层按分辨率分类
└─ outputs/                     # 运行后自动创建
```

## 1. 安装和启动

### 使用现有 Python 环境

在 VS Code 中打开整个 `underwater_realtime_enhancer` 文件夹。确认 VS Code 右下角选择的是你的 Python 3.12 环境，然后在终端执行：

```bash
python -m pip install -r requirements.txt
python check_environment.py
python app.py
```

你已经安装了 `torch==2.8.0` 和对应 CUDA 版本，因此 `requirements.txt` 不会重复安装 PyTorch，以免覆盖现有 CUDA 构建。

也可以打开 VS Code 的“运行和调试”，选择“运行水下实时增强界面”，按 `F5` 启动。

### TensorRT 模式额外要求

只有使用 TensorRT FP16 / INT8 时才需要 Python 环境能导入 `tensorrt`。运行：

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

请确保 engine 是在与运行电脑兼容的 TensorRT、CUDA 和 GPU 环境中生成的。TensorRT engine 通常不能随意跨 TensorRT 版本或不同 GPU 架构复用。

## 2. 添加自己的模型定义

程序依据 `models/` 文件夹里的 `.py` 文件发现模型。复制根目录的 `model_template.py` 到 `models/`，例如：

```text
models/PUIENet.py
```

然后做三件事：

1. 把训练时使用的网络结构放入该文件，网络结构必须与 `.pth` 权重完全一致。
2. 将 `MODEL_NAME` 设为权重文件的模型名前缀，例如 `PUIENet`。
3. 让 `create_model()` 返回一个尚未加载权重的 `torch.nn.Module` 实例。

模型定义最小接口：

```python
from torch import nn

MODEL_NAME = "PUIENet"
REQUIRE_WEIGHTS = True
# 可选：显式声明输出范围，避免线性输出层轻微越界时被自动误判。
# 支持 "0_1"、"minus1_1"、"0_255"、"auto"。
OUTPUT_RANGE = "0_1"

class PUIENet(nn.Module):
    def __init__(self):
        super().__init__()
        # 在这里放入真实网络层

    def forward(self, x):
        # 输入：RGB、NCHW、float32、范围 [0, 1]
        # 输出：RGB、NCHW 或 CHW Tensor
        return x

def create_model():
    return PUIENet()
```

新增或修改文件后，点击界面中的“刷新模型”。模型文件可以导入 `models/` 中的其他 Python 模块，但建议每个可选模型只有一个包含 `create_model()` 的入口文件；以下划线开头的辅助文件不会出现在下拉框。

每个模型会被独立加载。某个模型文件缺少依赖或存在语法错误时，该模型会被跳过，其他有效模型仍可使用，具体错误显示在窗口底部状态栏。

### 模型输入输出约定

- 输入固定为 1 张 RGB 图像：`[1, 3, H, W]`。
- PyTorch 输入是 `float32`，数值范围 `[0, 1]`。
- 输出可以是 Tensor，也可以是 list/tuple 的第一个 Tensor。
- 输出如果是 dict，程序优先读取 `output`、`enhanced`、`image`、`result`，否则读取第一个值。
- 输出支持 `[0, 1]`、`[-1, 1]` 或 `[0, 255]` 范围；建议通过模型文件的
  `OUTPUT_RANGE` 显式声明，未声明时才自动判断。
- 当前 TensorRT 后端要求一个输入和一个图像输出，布局为线性 NCHW。

如果你的模型需要多帧、额外参数、特殊归一化、多个输入或多个输出，需要在 `core/backends.py` 中为它定制预处理和后处理。

## 3. 放置训练好的权重

PyTorch 权重不绑定输入尺寸，统一使用 `weights/pth/模型名.pth`；界面切换分辨率时会继续加载同一个 checkpoint。TensorRT engine 通常与构建尺寸绑定，因此仍由模型名、具体分辨率和推理模式共同决定路径。文件名区分大小写，`MODEL_NAME` 必须与文件名前缀一致。TensorRT 目录和文件名不使用 `4k`、`2k`、`1k` 或 `720p` 等简称，统一使用 `宽x高`。

| 界面分辨率 | TensorRT 目录名 | PyTorch 文件名 | TensorRT FP16 文件名 | TensorRT INT8 文件名 |
| --- | --- | --- | --- | --- |
| 3840×2160 | `3840x2160` | `PUIENet.pth` | `PUIENet_3840x2160_trt16.engine` | `PUIENet_3840x2160_trt8.engine` |
| 2560×1440 | `2560x1440` | `PUIENet.pth` | `PUIENet_2560x1440_trt16.engine` | `PUIENet_2560x1440_trt8.engine` |
| 1920×1080 | `1920x1080` | `PUIENet.pth` | `PUIENet_1920x1080_trt16.engine` | `PUIENet_1920x1080_trt8.engine` |
| 1280×720 | `1280x720` | `PUIENet.pth` | `PUIENet_1280x720_trt16.engine` | `PUIENet_1280x720_trt8.engine` |
| 自定义 960×540 | `960x540` | `PUIENet.pth` | `PUIENet_960x540_trt16.engine` | `PUIENet_960x540_trt8.engine` |

以 PUIENet 和 1920×1080 TensorRT engine 为例，把文件分别放入：

```text
weights/pth/PUIENet.pth
weights/trt16/1920x1080/PUIENet_1920x1080_trt16.engine
weights/trt8/1920x1080/PUIENet_1920x1080_trt8.engine
```

选择自定义分辨率后，程序只为 TensorRT 自动建立 `weights/trt16/宽x高/` 或 `weights/trt8/宽x高/` 目录。界面会实时显示当前设置对应的完整相对路径以及“已找到”或“待放入”。PyTorch 始终直接读取 `weights/pth/模型名.pth`，不会建立分辨率子目录。

### 支持的 `.pth` 内容

程序支持：

- 直接保存的 `state_dict`；
- 含 `state_dict`、`model_state_dict`、`model` 或 `net` 字段的 checkpoint；
- 直接保存的 `nn.Module`；
- 自动去除由 `DataParallel` 产生的 `module.` 前缀。

默认采用 `strict=True` 加载，以便尽早发现模型结构与权重不一致。只加载你自己训练或确认可信的 checkpoint。

## 4. TensorRT engine 要求

你提供的 FP16 / INT8 engine 应满足：

- 一个输入、一个输出；
- batch 固定或运行时可设置为 1；
- 输入形状为 `[1, 3, H, W]`；
- RGB 图像输入，浮点 engine 通常接收 `[0, 1]`；
- 输出为增强后的 NCHW 图像；
- 输入、输出的 TensorRT 对外格式均为 `LINEAR`；
- 固定尺寸 engine 的 H、W 与文件名和界面选项完全一致。

程序在载入 engine 后会读取真实 I/O 形状、数据类型和格式。如果选择 3840×2160，但 engine 实际输入是 1920×1080，或 engine 暴露 CHW4/HWC8 等向量化 I/O，会直接给出错误，不会按错误的内存布局继续推理。动态尺寸 engine 会通过 TensorRT context 设置为界面选择的尺寸。

界面会在 `RGB → NCHW` 后强制调用 `contiguous()`。这是 TensorRT 直接绑定
PyTorch CUDA Tensor 时必须满足的条件：TensorRT 只得到 `data_ptr()`，无法读取
PyTorch 的 stride；非连续的 NCHW 视图会被错误解释并产生规则竖纹、偏色或黑屏。
TensorRT 输出缓冲区会在后端初始化时一次分配并持久绑定，与示例
`test_4k.py` 的推理流程一致。

INT8 指的是 engine 内部的 INT8 量化执行方式，不代表输入张量必须是 INT8。推荐保留浮点输入/输出，由 TensorRT engine 内部完成量化与反量化；如果你的 engine 暴露的是 INT8 输入，需要按量化 scale 定制 `TensorRTBackend.infer()`。

## 5. 界面使用流程

1. 点击“选择图像 / 视频”，或点击“使用摄像头 0”。
2. 选择模型、推理模式和处理分辨率。
3. 如果选择“其他 / 自定义”，输入宽和高。TensorRT 必须准备该尺寸对应的 engine。
4. 检查界面中的权重提示为“已找到”；`Passthrough + PyTorch` 显示“无需权重”。
5. 如需保存结果，勾选“保存增强结果”并选择目录。
6. 点击“开始”。视频和摄像头会持续处理，图像只推理一次。
7. 视频运行时可“暂停”，再点“继续”；点击“结束”释放视频、GPU 后端和保存文件。

保存文件使用时间戳命名：

```text
enhanced_20260714_103000.png
enhanced_20260714_103000.mp4
```

## 6. 性能指标含义

- **GPU 推理耗时**：用 CUDA Event 测量，主要表示模型或 engine 在 GPU 上的执行时间。
- **纯推理帧率**：`1000 / GPU 推理耗时(ms)`，不包含预处理、后处理、CPU/GPU 拷贝和界面绘制。
- **端到端处理帧率**：包含 resize、颜色转换、张量转换、推理和输出图像转换，但不包含 Qt 最终绘制时间。
- 视频数值使用指数滑动平均，避免逐帧剧烈闪动；图像显示单次测量。

首次推理通常包含 CUDA 初始化和内核准备，速度可能明显偏慢。比较模型时建议先预热若干帧，并保持输入源、分辨率、GPU 功耗状态一致。

## 7. 常见问题

### 找不到模型

确认模型 `.py` 文件位于 `models/`、文件名不是以下划线开头，并且定义了可调用的 `create_model()`。查看弹窗中的 Python 导入错误，通常是模型依赖未安装或相对导入路径错误。

### 找不到权重

对照界面“模型与推理”区域底部显示的完整相对路径。PyTorch 请检查 `weights/pth/模型名.pth`；TensorRT 请检查模型名大小写、`宽x高` 目录、文件名中的具体分辨率以及 `_trt16/_trt8` 后缀。

### `load_state_dict` 报 missing/unexpected keys

模型结构与训练时不完全相同，或 checkpoint 使用了其他字段包装。应优先修正 `models/模型名.py` 使结构准确，不建议简单关闭 strict 校验。

### TensorRT engine 反序列化失败

通常是生成 engine 的 TensorRT 版本、CUDA 环境或 GPU 架构与当前运行环境不兼容。请在目标机器的运行环境中重新构建 engine。

### TensorRT 输出有规则竖纹，但 PyTorch 输出正常

这通常不是界面缩放问题，而是 ONNX/TensorRT 图与 PyTorch 的算子语义不一致。请用同一输入比较 PyTorch 与 engine 原始输出；如果差异明显，需要重新导出或替换 engine。界面后端只能正确读取和显示 engine 的输出，不能把数值已经错误的 engine 恢复成原模型结果。

### CUDA out of memory

4K 输入会显著增加显存占用。先尝试 1080P/720P、TensorRT FP16，或在模型结构中使用分块推理。处理过程中不要同时运行其他占用大量显存的程序。

### 输出视频没有声音

OpenCV `VideoWriter` 只写入增强后的视频画面，不复制原视频音轨。如果需要保留音频，可处理完成后使用 FFmpeg 将原音轨封装回输出视频。

### 实际视频播放速度与原视频不同

程序以尽可能快的速度推理并展示，界面 FPS 是处理性能，不是限速后的播放 FPS。输出视频仍使用原视频声明的 FPS；如果需要严格实时节奏，可在工作线程中按源 FPS 加入显示节流。

## 8. 后续适合扩展的功能

- 摄像头编号选择、RTSP/USB 采集卡输入；
- 模型预热次数和 benchmark 统计导出；
- 半屏拖动对比、局部放大和直方图；
- FFmpeg 音轨保留、硬件编码和 RTMP 推流；
- 超大分辨率重叠分块推理；
- UIQM、UCIQE 等水下图像质量指标；
- 针对每个模型独立配置归一化、色彩空间和输出字段。

当前代码将通用界面与模型细节分离。后续新增模型时，通常只需要添加一个模型定义文件和对应权重，无需改动界面。
