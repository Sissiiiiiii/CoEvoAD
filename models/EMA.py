import torch


class BestCheckpoint:
    """Snapshot of model parameters taken at the best validation metric.

    Note: this is not a true EMA (exponential moving average); it is only used for checkpointing.
    Legacy update/apply_shadow/restore methods were defined but never called, and have been removed.
    """

    def __init__(self, model, **kwargs):
        # kwargs absorbs legacy arguments such as decay (no longer used)
        self.model = model
        self.last_check = {}

    def register(self):
        """Backward compatibility: legacy code calls ema.register(); now a no-op."""
        pass

    def save_check(self):
        """Save a snapshot of the current model parameters to CPU (called when the validation metric improves)."""
        self.last_check = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.last_check[name] = param.data.clone().cpu()

    def load_check(self):
        """Restore the last saved best-parameter snapshot."""
        if not self.last_check:
            return  # never saved, skip
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.last_check, f"Missing key: {name}"
                param.data = self.last_check[name].to(param.device)


# Backward compatibility: legacy code does 'from models.EMA import EMA'
EMA = BestCheckpoint
