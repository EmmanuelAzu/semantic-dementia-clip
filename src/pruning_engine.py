import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class CLIPPruningEngine:
    def __init__(self, model):
        """
        Initializes the Pruning Engine with an active, pretrained CLIP model.
        """
        self.base_model = model

    def get_pruned_model(self, amount: float, encoder_type: str = "both"):
        """
        Generates a newly pruned copy of the CLIP model targeting the transmodal hub.
        """
        if not (0.0 <= amount <= 1.0):
            raise ValueError("Pruning amount must be a float between 0.0 and 1.0.")

        # Create a deep copy to ensure we preserve the original healthy base model state
        pruned_model = copy.deepcopy(self.base_model)
        
        if amount == 0.0:
            return pruned_model

        # Extract target linear layers (projection heads) for pruning
        target_layers = self._gather_target_layers(pruned_model, encoder_type)
        
        if not target_layers:
            print(f"[Warning] No target linear weights matched encoder_type='{encoder_type}'.")
            return pruned_model

        # Apply STRUCTURED pruning to kill specific dimensions of the embedding space
        for module, param_name in target_layers:
            # n=2 uses L2 magnitude, dim=1 targets the 512-D output space directly
            prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=1)
            # Make pruning permanent
            prune.remove(module, name=param_name)

        return pruned_model

    def _gather_target_layers(self, model, encoder_type: str):
        """
        Locates ONLY the projection parameters that project embeddings 
        into CLIP's 512-dimensional multimodal shared space (hub).
        """
        targets = []

        # Target 1: Text projection matrix (shape: [transformer_width, 512])
        if encoder_type in ["text", "both"]:
            if hasattr(model, "text_projection") and model.text_projection is not None:
                targets.append((model, "text_projection"))

        # Target 2: Vision projection matrix (shape: [vision_width, 512])
        if encoder_type in ["vision", "both"]:
            if hasattr(model.visual, "proj") and model.visual.proj is not None:
                targets.append((model.visual, "proj"))

        return targets

    def verify_sparsity(self, model, encoder_type: str = "both") -> float:
        """Calculates and returns the exact fraction of zeroed weights."""
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