from __future__ import annotations

import torch
from torch import nn


MODEL_NAME = "Passthrough"
REQUIRE_WEIGHTS = False


class Passthrough(nn.Module):
    """无权重的流程测试模型，输出与输入完全相同。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def create_model() -> nn.Module:
    """遵循模型注册器要求的统一工厂函数。"""
    return Passthrough()
