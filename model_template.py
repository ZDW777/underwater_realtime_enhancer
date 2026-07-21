"""复制到 models/ 并重命名，例如 models/PUIENet.py。"""
from __future__ import annotations

from torch import nn


MODEL_NAME = "PUIENet"  # 必须与权重文件名的模型名前缀完全一致。
REQUIRE_WEIGHTS = True


class YourNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 替换为训练权重所对应的完整网络结构。
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)


def create_model() -> nn.Module:
    """程序会先调用此工厂函数，再加载 .pth 中的 state_dict。"""
    return YourNetwork()
