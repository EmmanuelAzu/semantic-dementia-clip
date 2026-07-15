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

    def get_pruned_model(self, amount: float, encoder_type: str = "both", depth: str = "global"):
        """
        Generates a newly pruned copy of the CLIP model based on target parameters.
        """
        if not (0.0 <= amount <= 1.0):
            raise ValueError("Pruning amount must be a float between 0.0 and 1.0.")

        # Create a deep copy to ensure we preserve the original healthy base model state
        pruned_model = copy.deepcopy(self.base_model)
        
        if amount == 0.0:
            return pruned_model

        # Extract target linear layers for pruning
        target_layers = self._gather_target_layers(pruned_model, encoder_type, depth)
        
        if not target_layers:
            print(f"[Warning] No target linear weights matched encoder_type='{encoder_type}' and depth='{depth}'.")
            return pruned_model

        # Apply unstructured L1 magnitude-based pruning
        for module, param_name in target_layers:
            prune.l1_unstructured(module, name=param_name, amount=amount)
            # Make pruning permanent (removes the mask buffers and hard-values the zeroed weights)
            prune.remove(module, name=param_name)

        return pruned_model

    def _gather_target_layers(self, model, encoder_type: str, depth: str):
        """
        Locates the standard linear projections (nn.Linear) in CLIP corresponding
        to specified encoder types and depth criteria.
        """
        targets = []

        text_blocks = list(model.transformer.resblocks) if hasattr(model, "transformer") else []
        vision_blocks = list(model.visual.transformer.resblocks) if (hasattr(model, "visual") and hasattr(model.visual, "transformer")) else []

        def segment_blocks(blocks, depth_query):
            if depth_query == "global":
                return blocks
            elif depth_query == "early":  
                return blocks[0:4]
            elif depth_query == "middle": 
                return blocks[4:8]
            elif depth_query == "deep":   
                return blocks[8:12]
            else:
                return blocks

        active_text = segment_blocks(text_blocks, depth) if encoder_type in ["text", "both"] else []
        active_vision = segment_blocks(vision_blocks, depth) if encoder_type in ["vision", "both"] else []

        for block in active_text:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    targets.append((module, "weight"))

        for block in active_vision:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    targets.append((module, "weight"))

        return targets

    def verify_sparsity(self, model, encoder_type: str = "both", depth: str = "global") -> float:
        """Calculates and returns the exact fraction of zeroed weights."""
        targets = self._gather_target_layers(model, encoder_type, depth)
        if not targets:
            return 0.0

        total_weights = 0
        zero_weights = 0

        for module, param_name in targets:
            weight = getattr(module, param_name)
            total_weights += weight.numel()
            zero_weights += torch.sum(weight == 0.0).item()

        return zero_weights / total_weights if total_weights > 0 else 0.0