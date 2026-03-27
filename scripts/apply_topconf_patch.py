#!/usr/bin/env python3
"""
Apply Top Conference Strategy patches to train_bytetrack.py

This script adds:
1. TripletLoss import and initialization
2. TripletLoss computation in the training loop
3. Support for TP Drop / FP Insert (future)
"""

import re

def apply_patch():
    # Read original file
    with open('/gemini/code/FMtrack-main/FM-Track/train_bytetrack.py', 'r') as f:
        content = f.read()
    
    # Patch 1: Add imports after existing imports
    import_marker = "from models.bytetrack_feature_extractor import ("
    new_imports = """from models.bytetrack_feature_extractor import ("""
    
    if "from models.motip.topconf_losses import" not in content:
        import_addition = """
# Top Conference Strategies (FairMOT, MeMOTR, etc.)
try:
    from models.motip.topconf_losses import TripletLoss, build_triplet_loss
    TOPCONF_AVAILABLE = True
except ImportError:
    TOPCONF_AVAILABLE = False
    TripletLoss = None

"""
        # Find the position after the imports block (before first function def)
        import_insert_pos = content.find("\n\n\ndef build_tracking_modules")
        if import_insert_pos > 0:
            content = content[:import_insert_pos] + import_addition + content[import_insert_pos:]
    
    # Patch 2: Add TripletLoss initialization in build_tracking_modules
    # Find the return statement in build_tracking_modules
    build_modules_pattern = r"(# 构建 ID 损失函数\n    id_criterion = build_id_criterion\(config\))"
    triplet_init = """# 构建 ID 损失函数
    id_criterion = build_id_criterion(config)
    
    # Top Conference: Build TripletLoss if enabled
    triplet_criterion = None
    if TOPCONF_AVAILABLE and config.get("USE_TRIPLET_LOSS", False):
        triplet_criterion = build_triplet_loss(config)"""
    
    if "triplet_criterion = None" not in content:
        content = re.sub(build_modules_pattern, triplet_init, content)
    
    # Patch 3: Modify return statement to include triplet_criterion
    old_return = "return trajectory_modeling, id_decoder, id_criterion"
    new_return = "return trajectory_modeling, id_decoder, id_criterion, triplet_criterion"
    
    if new_return not in content:
        content = content.replace(old_return, new_return)
    
    # Patch 4: Add triplet loss computation before total loss
    # Find the total loss calculation
    total_loss_pattern = r"(        # 总损失\n        loss = id_loss \* id_criterion\.weight)"
    
    triplet_loss_code = """        # Top Conference: TripletLoss (from FairMOT/FastReID)
        triplet_loss = torch.tensor(0.0, device=device)
        triplet_loss_weight = float(config.get("TRIPLET_LOSS_WEIGHT", 0.0))
        if triplet_criterion is not None and triplet_loss_weight > 0:
            # Get embeddings from seq_info (trajectory features)
            traj_features = seq_info.get("trajectory_features", None)
            traj_labels = seq_info.get("trajectory_id_labels", None)
            traj_masks = seq_info.get("trajectory_masks", None)
            if traj_features is not None and traj_labels is not None:
                triplet_loss = triplet_criterion(
                    embeddings=traj_features,
                    labels=traj_labels,
                    masks=traj_masks,
                )
        
        # 总损失
        loss = id_loss * id_criterion.weight"""
    
    if "triplet_criterion is not None" not in content:
        content = re.sub(total_loss_pattern, triplet_loss_code, content)
    
    # Patch 5: Add triplet_loss to total loss
    old_total_loss = """loss = id_loss * id_criterion.weight \\
               + freq_ortho_loss_weight * freq_ortho_loss \\
               + freq_consistency_loss_weight * freq_consistency_loss \\
               + newborn_weight * newborn_penalty"""
    
    new_total_loss = """loss = id_loss * id_criterion.weight \\
               + freq_ortho_loss_weight * freq_ortho_loss \\
               + freq_consistency_loss_weight * freq_consistency_loss \\
               + newborn_weight * newborn_penalty \\
               + triplet_loss_weight * triplet_loss"""
    
    if "triplet_loss_weight * triplet_loss" not in content:
        content = content.replace(old_total_loss, new_total_loss)
    
    # Patch 6: Add triplet_loss to metrics
    old_metrics = """        metrics.update(name="loss", value=loss.item())
        metrics.update(name="id_loss", value=id_loss.item())"""
    
    new_metrics = """        metrics.update(name="loss", value=loss.item())
        metrics.update(name="id_loss", value=id_loss.item())
        if triplet_loss_weight > 0:
            metrics.update(name="triplet_loss", value=triplet_loss.item())"""
    
    if 'metrics.update(name="triplet_loss"' not in content:
        content = content.replace(old_metrics, new_metrics)
    
    # Write patched file
    with open('/gemini/code/FMtrack-main/FM-Track/train_bytetrack.py', 'w') as f:
        f.write(content)
    
    print("Patch applied successfully!")
    return True

if __name__ == "__main__":
    apply_patch()
