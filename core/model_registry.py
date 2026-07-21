from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class ModelSpec:
    """记录一个已成功导入的模型入口及其加载要求。"""
    name: str
    file: Path
    module: ModuleType
    requires_weights: bool

    def create(self):
        """调用用户模型文件约定的工厂函数。"""
        factory = getattr(self.module, "create_model", None)
        if not callable(factory):
            raise RuntimeError(f"{self.file.name} 缺少 create_model() 函数")
        return factory()


class ModelRegistry:
    """扫描 models/*.py，并把有效模型注册到界面下拉框。"""
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: dict[str, ModelSpec] = {}
        self._errors: dict[str, str] = {}

    @property
    def models(self) -> dict[str, ModelSpec]:
        return dict(self._models)

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def scan(self) -> dict[str, ModelSpec]:
        """逐文件扫描；单个模型导入失败不会影响其他模型。"""
        found: dict[str, ModelSpec] = {}
        errors: dict[str, str] = {}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        models_path = str(self.models_dir.resolve())
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        for file in sorted(self.models_dir.glob("*.py")):
            # 下划线开头的文件可作为公共网络层或辅助模块，不显示在界面中。
            if file.name.startswith("_"):
                continue
            safe_stem = re.sub(r"[^0-9a-zA-Z_]", "_", file.stem)
            spec = importlib.util.spec_from_file_location(
                f"underwater_user_model_{safe_stem}", file
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                # 保存详细错误供状态栏显示，同时继续扫描下一个文件。
                errors[file.name] = f"{type(exc).__name__}: {exc}"
                continue
            if not callable(getattr(module, "create_model", None)):
                continue
            name = str(getattr(module, "MODEL_NAME", file.stem))
            found[name] = ModelSpec(
                name=name,
                file=file,
                module=module,
                requires_weights=bool(getattr(module, "REQUIRE_WEIGHTS", True)),
            )
        self._models = found
        self._errors = errors
        return self.models

    def get(self, name: str) -> ModelSpec:
        try:
            return self._models[name]
        except KeyError as exc:
            raise RuntimeError(f"找不到模型定义：{name}") from exc
