################## 1. Download checkpoints and build models
import torch
import numpy as np


def generate_inpainting_mask(patch_nums, target_layer, patch_coord_list, reverse=False):
    """
    Generate a binary mask for latent tokens across scales, allowing for multiple patch coordinates.

    Args:
        patch_nums (tuple): Tuple of patch numbers per layer, e.g. (1, 2, 3, 4, 5, 6, 8, 10, 13, 16).
        target_layer (int): The layer (0-indexed) where the patches are specified.
                            For a "sixth layer", use target_layer=5.
        patch_coord_list (list of tuple): List of coordinates (i, j) of patches in the target_layer grid.
                                          For example, [(2, 3), (4, 1)] in a 6x6 grid.
                                 
    Returns:
        torch.BoolTensor: A 1D Boolean mask with length equal to the total number of tokens
                          across all layers. True means "keep" and False means "mask for inpainting".
    """
    mask_list = []
    
    for s, pn in enumerate(patch_nums):
        tokens_in_layer = pn * pn
        # Start with a mask that keeps all tokens.
        layer_mask = torch.ones(tokens_in_layer, dtype=torch.bool) if not reverse else torch.zeros(tokens_in_layer, dtype=torch.bool)
        
        # For layers before the target layer, no tokens are masked.
        if s < target_layer:
            mask_list.append(layer_mask)
            continue
        
        # For the target layer and subsequent layers, process each patch coordinate.
        for coord in patch_coord_list:
            i_target, j_target = coord
            
            if s == target_layer:
                # For the target layer, mask exactly the token corresponding to each coordinate.
                idx = i_target * pn + j_target
                layer_mask[idx] = False if not reverse else True
            else:
                # For subsequent layers, compute the corresponding region.
                ratio = pn / patch_nums[target_layer]
                x_start = int(np.floor(i_target * ratio))
                x_end = int(np.ceil((i_target + 1) * ratio))
                y_start = int(np.floor(j_target * ratio))
                y_end = int(np.ceil((j_target + 1) * ratio))
                # For every (x, y) in the computed region, mark the token as False.
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        idx = x * pn + y
                        layer_mask[idx] = False if not reverse else True
        
        mask_list.append(layer_mask)
    
    # Concatenate all layer masks into one flat mask.
    full_mask = torch.cat(mask_list)
    return full_mask