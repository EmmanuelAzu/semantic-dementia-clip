import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class CLIPPruningEngine:
    def __init__(self, model):
        self.base_model = model

    def get_pruned_model(self, amount: float, encoder_type: str = "both", model: nn.Module = None):
        if not (0.0 <= amount <= 1.0):
            raise ValueError("Pruning amount must be a float between 0.0 and 1.0.")

        target_model = model if model is not None else self.base_model
        pruned_model = copy.deepcopy(target_model)
        
        if amount == 0.0:
            return pruned_model

        target_layers = self._gather_target_layers(pruned_model, encoder_type)
        
        if not target_layers:
            print(f"[Warning] No target linear weights matched encoder_type='{encoder_type}'.")
            return pruned_model

        for module, param_name in target_layers:
            prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=1)
            prune.remove(module, name=param_name)

        return pruned_model

    def _gather_target_layers(self, model, encoder_type: str):
        targets = []

        if encoder_type in ["text", "both", "joint"]:
            if hasattr(model, "text_projection") and model.text_projection is not None:
                targets.append((model, "text_projection"))

        if encoder_type in ["vision", "both", "joint"]:
            if hasattr(model.visual, "proj") and model.visual.proj is not None:
                targets.append((model.visual, "proj"))

        return targets

    def verify_sparsity(self, model, encoder_type: str = "both") -> float:
        targets = self._gather_target_layers(model, encoder_type)
        if not targets:
            return 0.0

        total_weights = 0
        zero_weights = 0

        for module, param_name in targets:
            weight = getattr(module, param_name)
            total_weights += weight.numel()
            zero_weights += torch.sum(weight == 0.0).item()

        return zero_weights / total_weights if total_weights > 0 else 0.0