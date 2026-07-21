from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import RunConfig
from core.backends import create_backend
from core.model_registry import ModelSpec
from core.weight_resolver import WeightResolver


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class InferenceWorker(QThread):
    """后台完成媒体读取和 GPU 推理，通过 Qt Signal 安全更新主线程界面。"""
    frames_ready = Signal(object, object)
    metrics_ready = Signal(float, float, float)
    state_changed = Signal(str)
    error = Signal(str)
    output_saved = Signal(str)

    def __init__(self, config: RunConfig, model_spec: ModelSpec, parent=None):
        super().__init__(parent)
        self.config = config
        self.model_spec = model_spec
        self._stop_event = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False

    def request_pause(self, paused: bool) -> None:
        """设置暂停状态；Condition 可避免暂停时持续占用 CPU。"""
        with self._pause_condition:
            self._paused = paused
            self._pause_condition.notify_all()

    def request_stop(self) -> None:
        """请求安全结束，并唤醒可能处于暂停状态的线程。"""
        self._stop_event.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def _wait_if_paused(self) -> None:
        with self._pause_condition:
            while self._paused and not self._stop_event.is_set():
                self._pause_condition.wait(timeout=0.2)

    def _output_path(self, suffix: str) -> Path | None:
        """按时间戳创建不会覆盖旧结果的输出路径。"""
        if not self.config.save_output or self.config.output_dir is None:
            return None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return self.config.output_dir / f"enhanced_{stamp}{suffix}"

    def run(self) -> None:
        """在线程内部创建 CUDA 后端，依次处理图像或视频帧。"""
        backend = None
        capture = None
        writer = None
        try:
            resolver = WeightResolver(self.config.root / "weights")
            weight_path = resolver.resolve(
                self.config.model_name,
                self.config.weight_resolution_key,
                self.config.backend,
            )
            backend = create_backend(
                self.config.backend,
                self.model_spec,
                weight_path,
                self.config.width,
                self.config.height,
            )
            self.state_changed.emit("运行中")

            if self.config.source_kind == "image":
                # imdecode + fromfile 可正确读取含中文字符的 Windows 路径。
                raw = np.fromfile(str(self.config.source), dtype=np.uint8)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("无法读取所选图像")
                started = time.perf_counter()
                enhanced, inference_ms = backend.infer(frame)
                total_ms = (time.perf_counter() - started) * 1000.0
                self.frames_ready.emit(frame.copy(), enhanced.copy())
                self.metrics_ready.emit(inference_ms, 1000.0 / max(inference_ms, 1e-6), 1000.0 / max(total_ms, 1e-6))
                output_path = self._output_path(".png")
                if output_path is not None:
                    ok, encoded = cv2.imencode(".png", enhanced)
                    if not ok:
                        raise RuntimeError(f"结果保存失败：{output_path}")
                    encoded.tofile(str(output_path))
                    self.output_saved.emit(str(output_path))
                self.state_changed.emit("图像处理完成")
                return

            capture = cv2.VideoCapture(self.config.source)
            if not capture.isOpened():
                raise RuntimeError("无法打开视频或摄像头")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            output_path = self._output_path(".mp4")
            if output_path is not None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (self.config.width, self.config.height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建输出视频：{output_path}")

            # 指数滑动平均让实时性能读数更稳定，不会逐帧剧烈闪动。
            inference_ema = None
            total_ema = None
            alpha = 0.15
            while not self._stop_event.is_set():
                self._wait_if_paused()
                if self._stop_event.is_set():
                    break
                ok, frame = capture.read()
                if not ok:
                    if self.config.source_kind == "camera":
                        continue
                    break
                started = time.perf_counter()
                enhanced, inference_ms = backend.infer(frame)
                total_ms = (time.perf_counter() - started) * 1000.0
                inference_ema = inference_ms if inference_ema is None else (1 - alpha) * inference_ema + alpha * inference_ms
                total_ema = total_ms if total_ema is None else (1 - alpha) * total_ema + alpha * total_ms
                self.frames_ready.emit(frame.copy(), enhanced.copy())
                self.metrics_ready.emit(
                    inference_ema,
                    1000.0 / max(inference_ema, 1e-6),
                    1000.0 / max(total_ema, 1e-6),
                )
                if writer is not None:
                    writer.write(enhanced)

            if writer is not None:
                writer.release()
                writer = None
            if output_path is not None and output_path.exists():
                self.output_saved.emit(str(output_path))
            self.state_changed.emit("已结束" if self._stop_event.is_set() else "视频处理完成")
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
            self.state_changed.emit("发生错误")
        finally:
            # 无论正常结束还是发生异常，都释放文件句柄和摄像头资源。
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            if backend is not None:
                backend.close()
