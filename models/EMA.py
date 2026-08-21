import torch


class BestCheckpoint:
    """保存/恢复最佳验证指标时的模型参数快照。

    注意：这不是真正的 EMA（指数移动平均）。仅做 checkpoint 用途。
    旧代码中定义了 update/apply_shadow/restore 但从未被调用，已清理。
    """

    def __init__(self, model, **kwargs):
        # kwargs 吸收旧代码传入的 decay 等参数（不再使用）
        self.model = model
        self.last_check = {}

    def register(self):
        """向后兼容：旧代码调用 ema.register()，现在是 no-op"""
        pass

    def save_check(self):
        """保存当前模型参数快照到 CPU（当验证指标提升时调用）"""
        self.last_check = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.last_check[name] = param.data.clone().cpu()

    def load_check(self):
        """恢复上次保存的最佳参数快照"""
        if not self.last_check:
            return  # 从未保存过，跳过
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.last_check, f"Missing key: {name}"
                param.data = self.last_check[name].to(param.device)


# 向后兼容：旧代码 from models.EMA import EMA
EMA = BestCheckpoint
