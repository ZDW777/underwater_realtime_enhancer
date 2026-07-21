from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.model_registry import ModelSpec


def _to_uint8_bgr(output: Any, output_range: str = "auto") -> np.ndarray:
    """把模型输出统一转换为 OpenCV 可显示的 uint8 BGR 图像。

    ``output_range`` 可由模型文件中的 ``OUTPUT_RANGE`` 指定：
    ``"0_1"``、``"minus1_1"``、``"0_255"`` 或 ``"auto"``。
    显式声明可以避免增强网络少量越界时被自动判断成错误的数值范围。
    """
    import torch

    # 兼容常见训练项目返回的字典、列表或元组。
    if isinstance(output, dict):
        for key in ("output", "enhanced", "image", "result"):
            if key in output:
                output = output[key]
                break
        else:
            output = next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("模型输出必须是 Tensor、list/tuple 或包含 Tensor 的 dict")

    tensor = output.detach().float()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"模型输出应为 CHW/NCHW，实际形状为 {tuple(tensor.shape)}")
    if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor[:3]

    # 先清理 FP16/INT8 engine 可能产生的非有限值，防止后续转换出现黑屏。
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    range_name = str(output_range).strip().lower().replace(" ", "")
    aliases = {
        "[0,1]": "0_1",
        "0-1": "0_1",
        "[-1,1]": "minus1_1",
        "-1_1": "minus1_1",
        "-1-1": "minus1_1",
        "[0,255]": "0_255",
        "0-255": "0_255",
    }
    range_name = aliases.get(range_name, range_name)
    if range_name not in {"auto", "0_1", "minus1_1", "0_255"}:
        raise ValueError(f"不支持的 OUTPUT_RANGE：{output_range}")

    if range_name == "auto":
        min_value = float(tensor.min())
        max_value = float(tensor.max())
        # 增强网络常用线性输出层，结果会轻微越过 0~1；1.5 的容差可避免
        # EAFDNet 一类输出因单个高光像素超过 1.05 而整帧被误判为 0~255。
        if min_value >= -1.5 and max_value <= 1.5:
            range_name = "minus1_1" if min_value < -0.05 else "0_1"
        else:
            range_name = "0_255"

    if range_name == "minus1_1":
        tensor = (tensor + 1.0) * 0.5
    if range_name in {"0_1", "minus1_1"}:
        tensor = tensor.clamp(0.0, 1.0).mul(255.0)
    else:
        tensor = tensor.clamp(0.0, 255.0)
    rgb = tensor.byte().permute(1, 2, 0).contiguous().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class InferenceBackend(ABC):
    """所有推理后端都遵循的最小接口。"""
    @abstractmethod
    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Return enhanced BGR image and GPU inference time in milliseconds."""

    def close(self) -> None:
        pass


class PyTorchBackend(InferenceBackend):
    """加载模型结构和 .pth checkpoint 的 CUDA 推理后端。"""
    def __init__(self, model_spec: ModelSpec, weight_path: Path, width: int, height: int):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 模式需要可用的 NVIDIA CUDA GPU")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.width = width
        self.height = height
        self.output_range = getattr(model_spec.module, "OUTPUT_RANGE", "auto")
        self.model = model_spec.create().to(self.device).eval()

        # Passthrough 可跳过权重；真实增强模型默认必须加载 checkpoint。
        if model_spec.requires_weights:
            if not weight_path.is_file():
                raise FileNotFoundError(f"缺少 PyTorch 权重：{weight_path}")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint.to(self.device).eval()
            else:
                if isinstance(checkpoint, dict):
                    for key in ("state_dict", "model_state_dict", "model", "net"):
                        value = checkpoint.get(key)
                        if isinstance(value, dict):
                            checkpoint = value
                            break
                if not isinstance(checkpoint, dict):
                    raise TypeError(".pth 应包含 state_dict、常见 checkpoint 字典或 nn.Module")
                # 自动兼容 torch.nn.DataParallel 保存的 module. 前缀。
                cleaned = {
                    (key[7:] if key.startswith("module.") else key): value
                    for key, value in checkpoint.items()
                }
                self.model.load_state_dict(cleaned, strict=True)

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self.torch
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        # CUDA Event 只统计 GPU 推理时间，预处理和后处理计入端到端耗时。
        with torch.inference_mode():
            self.start_event.record()
            output = self.model(tensor)
            self.end_event.record()
            self.end_event.synchronize()
        inference_ms = float(self.start_event.elapsed_time(self.end_event))
        return _to_uint8_bgr(output, self.output_range), inference_ms


class TensorRTBackend(InferenceBackend):
    """TensorRT 10.x 后端，直接使用 PyTorch CUDA Tensor 作为 engine 缓冲区。"""

    def __init__(
        self,
        engine_path: Path,
        width: int,
        height: int,
        output_range: str = "auto",
    ):
        if not engine_path.is_file():
            raise FileNotFoundError(f"缺少 TensorRT 引擎：{engine_path}")
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError("TensorRT 模式需要安装 tensorrt，并且需要 CUDA 版 PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT 模式需要可用的 NVIDIA CUDA GPU")

        self.trt = trt
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.width = width
        self.height = height
        self.output_range = output_range
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 初始化 TensorRT 官方插件，兼容包含标准插件层的 engine。
        trt.init_libnvinfer_plugins(self.logger, "")
        with engine_path.open("rb") as file:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT 引擎反序列化失败：{engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("无法创建 TensorRT execution context")

        # 当前通用界面约定一个图像输入和一个增强图像输出。
        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            (input_names if mode == trt.TensorIOMode.INPUT else output_names).append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError("当前程序要求 engine 只有一个图像输入和一个增强图像输出")
        self.input_name = input_names[0]
        self.output_name = output_names[0]

        # 本后端直接把普通 NCHW CUDA Tensor 的地址交给 TensorRT，因此 engine
        # 对外 I/O 必须是线性格式；CHW4/HWC8 等向量化格式需要专门的打包逻辑。
        for name in (self.input_name, self.output_name):
            tensor_format = self.engine.get_tensor_format(name)
            if tensor_format != trt.TensorFormat.LINEAR:
                description = self.engine.get_tensor_format_desc(name)
                raise RuntimeError(
                    f"TensorRT I/O {name!r} 不是 LINEAR 格式：{description}。"
                    "请导出线性 NCHW 输入/输出的 engine。"
                )

        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        wanted = (1, 3, height, width)
        # 动态 engine 在此设置尺寸；静态 engine 则严格核对界面选择。
        if -1 in input_shape:
            if not self.context.set_input_shape(self.input_name, wanted):
                raise RuntimeError(f"无法把动态输入设置为 {wanted}")
        elif input_shape != wanted:
            raise RuntimeError(f"engine 固定输入为 {input_shape}，但界面选择的是 {wanted}")

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"TensorRT 输出形状未确定：{output_shape}")
        if len(output_shape) != 4 or output_shape[0] != 1 or output_shape[1] not in (1, 3, 4):
            raise RuntimeError(
                "当前界面要求 TensorRT 输出为 NCHW 图像张量，"
                f"实际输出是 {output_shape}"
            )

        # 与已验证的 test_4k.py 保持一致：在后端初始化时一次性分配连续输出
        # 缓冲区，并把地址长期绑定给 execution context。这样既避免每帧申请
        # 显存，也排除了临时 Tensor 的生命周期和 stride 对输出造成的影响。
        self.input_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name)
        )
        self.output_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.output_name)
        )
        self.output_buffer = torch.empty(
            output_shape,
            dtype=self.output_dtype,
            device=self.device,
        ).contiguous()
        if not self.context.set_tensor_address(
            self.output_name,
            self.output_buffer.data_ptr(),
        ):
            raise RuntimeError(f"无法绑定 TensorRT 输出：{self.output_name}")

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        # 与 test_4k.py 一样使用独立非默认流，避免 TensorRT 在默认流上
        # 插入额外的全局同步。
        self.stream = torch.cuda.Stream(device=self.device)

    def _torch_dtype(self, trt_dtype):
        trt = self.trt
        torch = self.torch
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        try:
            mapping[trt.bfloat16] = torch.bfloat16
        except AttributeError:
            pass
        try:
            mapping[trt.uint8] = torch.uint8
        except AttributeError:
            pass
        if trt_dtype not in mapping:
            raise TypeError(f"暂不支持 TensorRT 数据类型：{trt_dtype}")
        return mapping[trt_dtype]

    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self.torch
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device,
            non_blocking=True,
        )

        # 关键兼容修复：permute 只改变张量视图，不会重排底层内存。
        # TensorRT 通过 data_ptr() 读取裸显存，不认识 PyTorch stride；如果没有
        # contiguous()，它会把 HWC 内存错误地当作连续 NCHW，表现为规则竖纹、
        # 严重偏色或黑屏。下面的顺序与 test_4k.py 完全一致。
        input_tensor = input_tensor.permute(2, 0, 1).contiguous()
        input_tensor = input_tensor.to(self.input_dtype).unsqueeze(0)
        if self.input_dtype in (torch.float16, torch.float32, torch.bfloat16):
            input_tensor = input_tensor.div_(255.0)
        if not input_tensor.is_contiguous():
            # 理论上不会触发；保留防御检查，避免以后修改预处理时问题复发。
            raise RuntimeError(
                f"TensorRT 输入必须是连续 NCHW，实际 stride={input_tensor.stride()}"
            )

        # 预处理在 PyTorch 当前流完成；TensorRT 独立流先等待该流，再访问缓冲区。
        producer_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer_stream)
        input_tensor.record_stream(self.stream)

        # 输出地址已在初始化时持久绑定；每帧只更新输入地址。
        if not self.context.set_tensor_address(self.input_name, input_tensor.data_ptr()):
            raise RuntimeError(f"无法绑定 TensorRT 输入：{self.input_name}")
        self.start_event.record(self.stream)
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.end_event.record(self.stream)
        self.end_event.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 执行失败")
        inference_ms = float(self.start_event.elapsed_time(self.end_event))
        return _to_uint8_bgr(self.output_buffer, self.output_range), inference_ms


def create_backend(
    backend_name: str,
    model_spec: ModelSpec,
    weight_path: Path,
    width: int,
    height: int,
) -> InferenceBackend:
    """根据界面模式创建相应后端，调用方无需关心具体实现。"""
    if backend_name == "pytorch":
        return PyTorchBackend(model_spec, weight_path, width, height)
    if backend_name in ("trt16", "trt8"):
        output_range = getattr(model_spec.module, "OUTPUT_RANGE", "auto")
        return TensorRTBackend(weight_path, width, height, output_range)
    raise ValueError(f"未知推理模式：{backend_name}")
