################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import logging
import sys

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

import argparse
from utils.data import build_dataset
from torch.utils.data import DataLoader
import tqdm
import json
import matplotlib.pyplot as plt

from utils_clf import generate_inpainting_mask
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import clip
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

MODEL_DEPTH = 16  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
LOG_DIR = "./analysis"

def create_heatmaps_for_classes(probs: torch.Tensor, patch_nums: list, input_img: torch.Tensor, alpha: float = 0.5):
    """
    Given a probability tensor of shape (10, L) (10 classes, L = sum(p^2) patches across 10 layers)
    and an input image tensor normalized to [-1, 1], create a heatmap overlay for each class.
    """
    patch_nums = patch_nums[:len(patch_nums)//2]
    num_classes = probs.shape[0]
    overlaid_images = []
    combined_heatmap_list = []

    # Compute total number of patches L = sum(p^2)
    total_patches = sum([p*p for p in patch_nums])
    
    # Convert input image to numpy [0,255]
    img_np = input_img.clone().detach().cpu()
    img_np = (img_np + 1) / 2  
    if input_img.dim() == 4:
        input_img = input_img.squeeze(0)
    img_np = (input_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    for class_idx in range(num_classes):
        prob_vector = probs[class_idx]
        layer_heatmaps = []
        start = 0
        
        for p in patch_nums:
            num_patches = p * p
            layer_probs = prob_vector[start:start + num_patches]
            start += num_patches
            
            patch_map = layer_probs.view(1, 1, p, p)
            upsampled = torch.nn.functional.interpolate(patch_map, size=(256, 256), mode='bilinear', align_corners=False)
            upsampled = upsampled.squeeze()
            layer_heatmaps.append(upsampled * (num_patches / total_patches))
        
        combined_heatmap = torch.stack(layer_heatmaps, dim=0).sum(dim=0)
        combined_heatmap = combined_heatmap.cpu().numpy()
        combined_heatmap_list.append(combined_heatmap)
    
    combined_heatmap_list = np.stack(combined_heatmap_list)
    for combined_heatmap in combined_heatmap_list:
        combined_heatmap = combined_heatmap - combined_heatmap_list.min()
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / (combined_heatmap_list.max() - combined_heatmap_list.min())
        cmap = plt.get_cmap('jet')
        colored_heatmap = (cmap(combined_heatmap)[..., :3] * 255).astype(np.uint8)
        overlay = np.clip(img_np * (1 - alpha) + colored_heatmap * alpha, 0, 255).astype(np.uint8)
        overlaid_images.append(overlay)

    return overlaid_images


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", type=str, default="imagenet10", choices=["imagenet10", "imagenet", "imagenet-a"], help="Dataset to use")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Name of split")
    parser.add_argument("--extra", type=str, default=None, help="to add to the dataset name")
    parser.add_argument("--partial", type=int, default=200)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=0)
    parser.add_argument("--Clayer", type=int, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--mode", type=str, default="var", choices=["var", "l2_dist"])
    parser.add_argument("--feat", type=str, default="dinov2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--plot_kde", action='store_true')
    parser.add_argument("--top_k", type=int, default=None, help="Number of top probability tokens to consider for L2 distance calculation")
    parser.add_argument("--plot_dist_kde", action='store_true', help="Plot KDE showing token distance vs. probability relation for each scale")
    args = parser.parse_args()
    MODEL_DEPTH = args.depth

    name = f"var"
    extra = args.extra if args.extra is not None else ""
    if args.depth != 16:
        name += f"_d{args.depth}"
    name += f"_cfg[{args.cfg}]"
    if args.top_k is not None:
        name += f"_topk[{args.top_k}]"

    run_folder = osp.join(LOG_DIR, args.dataset,args.mode, name) if len(extra) == 0 else osp.join(LOG_DIR, args.dataset,args.mode, name + f"_{extra}")
    os.makedirs(run_folder, exist_ok=True)
    
    # Setup standard logging instead of the custom PrintLogger
    log_file = osp.join(run_folder, "analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Log file: {log_file}")

    layerwise_folder = osp.join(LOG_DIR, args.dataset, args.mode, name, "layerwise") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "layer_analysis")
    os.makedirs(layerwise_folder, exist_ok=True)

    layer_acc_folder = osp.join(LOG_DIR, args.dataset, args.mode, name, "layer_acc") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "layer_acc_analysis")
    os.makedirs(layer_acc_folder, exist_ok=True)

    layer_cond_folder = osp.join(LOG_DIR, args.dataset, args.mode, name, "layer_cond") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "layer_cond_analysis")
    os.makedirs(layer_cond_folder, exist_ok=True)

    # Build dataset
    data_path = f"./datasets/{args.dataset}"
    if args.dataset == "imagenet-a":
        num_classes, _, dataset_val, class_indices = build_dataset(
            data_path=data_path,
            final_reso=256,
            dataset_type=args.dataset
        )
    else:
        num_classes, _, dataset_val = build_dataset(
            data_path=data_path,
            final_reso=256,
            dataset_type=args.dataset
        )
        class_indices = [i for i in range(num_classes)]
    
    ld_val = DataLoader(dataset_val, num_workers=0, pin_memory=True, batch_size=1, shuffle=False, drop_last=False)
    del dataset_val

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    patch_nums_square_cumsum = np.cumsum(np.array(patch_nums)**2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Only build the model specified by args.depth
    if "vae" not in globals() or "var" not in globals():
        if args.depth == 16:
            logging.info("Building VAE and VAR-d16 model")
            vae, var_model = build_vae_var(
                V=4096, Cvae=32, ch=160, share_quant_resi=4,
                device=device, patch_nums=patch_nums, num_classes=1000, depth=16, shared_aln=False
            )
            model_ckpt = "var_d16.pth"
        elif args.depth == 30:
            logging.info("Building VAE and VAR-d30 model")
            vae, var_model = build_vae_var(
                V=4096, Cvae=32, ch=160, share_quant_resi=4,
                device=device, patch_nums=patch_nums, num_classes=1000, depth=30, shared_aln=False
            )
            model_ckpt = "var_d30.pth"
        else:
            raise ValueError(f"Unsupported model depth: {args.depth}. Must be either 16 or 30.")

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(model_ckpt):
        os.system(f"wget {hf_home}/{model_ckpt}")

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var_model.load_state_dict(torch.load(model_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var_model.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var_model.parameters():
        p.requires_grad_(False)
    logging.info("prepare finished.")
    var_model.cond_drop_rate = 0

    ############################# 2. Sample with classifier-free guidance

    seed = 0
    torch.manual_seed(seed)
    num_sampling_steps = 250
    more_smooth = False

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    correct = 0
    total = 0
    
    # Track accuracies for each scale
    scale_correct = {i: 0 for i in range(len(patch_nums))}
    scale_total = {i: 0 for i in range(len(patch_nums))}
    
    # Track accuracies for accumulated likelihoods
    acc_correct = {scale_idx: 0 for scale_idx in range(len(patch_nums))}  # scale_idx from 0 to 9
    acc_total = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    
    # Track accuracies for conditional likelihoods (excluding first scale_idx scales)
    cond_correct = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    cond_total = {scale_idx: 0 for scale_idx in range(len(patch_nums))}

    # Dictionaries to store probabilities across all samples per class.
    if args.dataset == "imagenet10":
        overall_class_probs = {cls: [] for cls in [i for i in range(num_classes)][:10] + [1000]}
    elif args.dataset == "imagenet-a":
        # For ImageNet-A, use the actual class indices plus the unconditional class (1000)
        overall_class_probs = {cls: [] for cls in class_indices + [1000]}
    else:
        overall_class_probs = {cls: [] for cls in [i for i in range(num_classes)] + [1000]}
    
    # Precompute embedding distances if using l2_dist mode
    if args.mode == "l2_dist":
        # Get embedding weights from vae model
        emb_weight = vae.quantize.embedding.weight  # (V, D)
        # Compute pairwise L2 distances between all embeddings
        dists = torch.cdist(emb_weight, emb_weight, p=2)  # (V, V)
        
        logging.info(f"Precomputed embedding distances with shape: {dists.shape}")
        if args.top_k is not None:
            logging.info(f"Using top {args.top_k} most probable tokens for L2 distance calculation")

    # For plotting token distance vs. probability relation
    if args.plot_dist_kde and args.mode == "l2_dist":
        # For each scale, store distances and probabilities for each sample
        # Key structure: {sample_idx: {scale_idx: {'distances': [], 'probs': []}}}
        sample_scale_distances_probs = {}
        
        # For overall plots across samples
        overall_scale_distances_probs = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        
        # For wrong class condition analysis
        sample_scale_distances_probs_wrong = {}
        overall_scale_distances_probs_wrong = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        
        dist_kde_folder = osp.join(LOG_DIR, args.dataset, args.mode, name, "dist_kde") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "dist_kde_analysis")
        os.makedirs(dist_kde_folder, exist_ok=True)
        
        # Create folder for wrong condition analysis
        dist_kde_folder_wrong = osp.join(LOG_DIR, args.dataset, args.mode, name, "dist_kde_wrong_cond") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "dist_kde_analysis_wrong_cond")
        os.makedirs(dist_kde_folder_wrong, exist_ok=True)
        
        logging.info(f"Will generate token distance vs. probability plots in: {dist_kde_folder}")
        logging.info(f"Will generate token distance vs. probability plots for wrong conditions in: {dist_kde_folder_wrong}")

    pbar = tqdm.tqdm(ld_val)
    for idx, (img, label) in enumerate(pbar):
        if args.partial is not None and idx >= args.partial:
            break
        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}%")
        img = img.to(device)            
        # List of classes to process for this sample.
        if args.dataset == "imagenet10":
            remaining_classes = [i for i in range(num_classes)][:10] + [1000]
        elif args.dataset == "imagenet-a":
            # For ImageNet-A, use the actual class indices plus the unconditional class (1000)
            remaining_classes = class_indices + [1000]
        else:
            remaining_classes = [i for i in range(num_classes)] + [1000]
        log_likelihood_list = []
        # For per-sample plotting if needed.
        prob_list = []
        
        # Store log likelihoods or distances for each scale
        scale_log_likelihoods = {i: [] for i in range(len(patch_nums))}
        
        # Store accumulated log likelihoods or distances
        acc_log_likelihoods = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        
        # Store conditional log likelihoods or distances (excluding first scale_idx scales)
        cond_log_likelihoods = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        
        # Create JSON filename for storing results
        json_fname = osp.join(run_folder, f"{idx}.json")
        with torch.inference_mode():
            gt_idx_list = vae.img_to_idxBl(img)
            gt_tokens = torch.cat(gt_idx_list, dim=1)
            x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_list)

            if args.cfg > 0:
                # Process a batch of classes.
                uncondition_label = 1000
                uncondition_label = torch.tensor(uncondition_label, device=device)

                uncondition_logits = var_model.forward(uncondition_label, x_BLCv_wo_first_l)

                ratio_list = []
                for si, pn in enumerate(patch_nums):
                    ratio = si / (len(patch_nums)-1)  # Scale-dependent ratio
                    ratio_list += [ratio] * (pn * pn)

                ratio_list = torch.tensor(ratio_list, device=device)
                t = args.cfg * ratio_list.unsqueeze(0).unsqueeze(-1)

            while len(remaining_classes) > 0:
                # Process a batch of classes.
                class_labels = remaining_classes[: args.batch_size]
                remaining_classes = remaining_classes[args.batch_size:]
                label_B = torch.tensor(class_labels, device=device)

                logits = var_model.forward(label_B, x_BLCv_wo_first_l)

                if args.cfg > 0:
                    logits = (1 + t) * logits - t * uncondition_logits

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs = torch.nn.functional.softmax(logits, dim=-1)

                gt_probs = probs.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)

                # Collect token distances and probabilities for KDE plotting
                if args.plot_dist_kde and args.mode == "l2_dist":
                    batch_size = probs.shape[0]
                    for b in range(batch_size):
                        start_idx = 0
                        
                        if class_labels[b] == 1000:
                            continue

                        # Process based on whether the class label matches or not
                        elif class_labels[b] == label.item():
                            # Correct class condition - use existing code
                            # Initialize data structure for this sample if needed
                            if idx not in sample_scale_distances_probs:
                                sample_scale_distances_probs[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                
                            for scale_idx, num_patches in enumerate(patch_nums):
                                num_patches_square = num_patches * num_patches
                                end_idx = start_idx + num_patches_square
                                
                                # Get ground truth tokens for this scale
                                scale_gt_tokens = gt_tokens[b, start_idx:end_idx]  # (num_patches_square)
                                
                                # Get distances from gt_tokens to all other tokens
                                scale_gt_distances = dists[scale_gt_tokens]  # (num_patches_square, V)
                                
                                # Get probabilities for this scale
                                scale_probs = probs[b, start_idx:end_idx]  # (num_patches_square, V)
                                
                                # Flatten and collect
                                flat_distances = scale_gt_distances.view(-1).detach().cpu().numpy()
                                flat_probs = scale_probs.view(-1).detach().cpu().numpy()
                                
                                # Store for later plotting
                                sample_scale_distances_probs[idx][scale_idx]['distances'].append(flat_distances)
                                sample_scale_distances_probs[idx][scale_idx]['probs'].append(flat_probs)
                                
                                # Also store for overall plots
                                overall_scale_distances_probs[scale_idx]['distances'].append(flat_distances)
                                overall_scale_distances_probs[scale_idx]['probs'].append(flat_probs)
                                
                                start_idx = end_idx
                                
                        else:
                            # Wrong class condition - store separately
                            # Initialize data structure for this sample if needed
                            if idx not in sample_scale_distances_probs_wrong:
                                sample_scale_distances_probs_wrong[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                
                            for scale_idx, num_patches in enumerate(patch_nums):
                                num_patches_square = num_patches * num_patches
                                end_idx = start_idx + num_patches_square
                                
                                # Get ground truth tokens for this scale
                                scale_gt_tokens = gt_tokens[b, start_idx:end_idx]  # (num_patches_square)
                                
                                # Get distances from gt_tokens to all other tokens
                                scale_gt_distances = dists[scale_gt_tokens]  # (num_patches_square, V)
                                
                                # Get probabilities for this scale
                                scale_probs = probs[b, start_idx:end_idx]  # (num_patches_square, V)
                                
                                # Flatten and collect
                                flat_distances = scale_gt_distances.view(-1).detach().cpu().numpy()
                                flat_probs = scale_probs.view(-1).detach().cpu().numpy()
                                
                                # Store wrong class data for later plotting 
                                sample_scale_distances_probs_wrong[idx][scale_idx]['distances'].append(flat_distances)
                                sample_scale_distances_probs_wrong[idx][scale_idx]['probs'].append(flat_probs)
                                
                                # Also store for overall wrong condition plots
                                overall_scale_distances_probs_wrong[scale_idx]['distances'].append(flat_distances)
                                overall_scale_distances_probs_wrong[scale_idx]['probs'].append(flat_probs)
                                
                                start_idx = end_idx

                # Append per-sample lists.
                prob_list.append(gt_probs)

                # Also accumulate into the overall dictionaries per class.
                # Here, the i-th element in the batch corresponds to class_labels[i].
                for i, cls in enumerate(class_labels):
                    overall_class_probs[cls].append(gt_probs[i].detach().cpu())

                if args.mode == "var":
                    # Calculate log likelihood for each scale
                    gt_log_probs = log_probs.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)
                    
                    start_idx = 0
                    for scale_idx, num_patches in enumerate(patch_nums):
                        num_patches_square = num_patches * num_patches
                        end_idx = start_idx + num_patches_square
                        
                        # Sum over patches in this scale
                        scale_log_likelihood = gt_log_probs[:, start_idx:end_idx].sum(dim=-1)
                        
                        scale_log_likelihoods[scale_idx].append(scale_log_likelihood)
                        
                        # Calculate accumulated likelihoods up to this scale
                        acc_scale_log_likelihood = gt_log_probs[:, :end_idx].sum(dim=-1)
                        acc_log_likelihoods[scale_idx].append(acc_scale_log_likelihood)
                        
                        # Calculate conditional likelihoods (excluding first scale_idx scales)
                        if scale_idx > 0:
                            start_cond_idx = patch_nums_square_cumsum[scale_idx - 1]
                            cond_scale_log_likelihood = gt_log_probs[:, start_cond_idx:].sum(dim=-1)
                            cond_log_likelihoods[scale_idx].append(cond_scale_log_likelihood)
                        else:
                            # When scale_idx = 0, conditional likelihood is the same as overall likelihood
                            cond_log_likelihoods[scale_idx].append(gt_log_probs.sum(dim=-1))
                        
                        start_idx = end_idx
                    
                    # Calculate overall log likelihood
                    log_likelihood = gt_log_probs.sum(dim=-1)
                    log_likelihood_list.append(log_likelihood)
                
                elif args.mode == "l2_dist":
                    # For L2 distance-based classification, we compute:
                    # average distance = sum(dists[gt_token, i] * probs[i]) for all tokens i
                    
                    # Get distances for ground-truth tokens to all other tokens
                    gt_distances = dists[gt_tokens]  # (B, L, V)
                    
                    if args.top_k is not None:
                        # Get top k probabilities and their indices
                        top_k_probs, top_k_indices = torch.topk(probs, k=args.top_k, dim=-1)  # (B, L, k)
                        
                        # Gather distances for top k tokens
                        top_k_distances = torch.gather(gt_distances, dim=-1, index=top_k_indices)  # (B, L, k)
                        
                        # Normalize top k probabilities
                        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                        
                        # Compute weighted average distance using only top k tokens
                        avg_distance = (top_k_distances * top_k_probs).sum(dim=-1)  # (B, L)
                    else:
                        # Original implementation using all tokens
                        avg_distance = (gt_distances * probs).sum(dim=-1)  # (B, L)
                    
                    # For classification, smaller distance is better
                    # We negate the distance so that larger (less negative) values are better
                    # This makes it compatible with the existing max-based classification logic
                    neg_avg_distance = -avg_distance
                    
                    # Calculate per-scale average distances
                    start_idx = 0
                    for scale_idx, num_patches in enumerate(patch_nums):
                        num_patches_square = num_patches * num_patches
                        end_idx = start_idx + num_patches_square
                        
                        # Sum over patches in this scale (smaller sum = better)
                        scale_distance = neg_avg_distance[:, start_idx:end_idx].sum(dim=-1)
                        
                        scale_log_likelihoods[scale_idx].append(scale_distance)
                        
                        # Calculate accumulated distances up to this scale
                        acc_scale_distance = neg_avg_distance[:, :end_idx].sum(dim=-1)
                        acc_log_likelihoods[scale_idx].append(acc_scale_distance)
                        
                        # Calculate conditional distances (excluding first scale_idx scales)
                        if scale_idx > 0:
                            start_cond_idx = patch_nums_square_cumsum[scale_idx - 1]
                            cond_scale_distance = neg_avg_distance[:, start_cond_idx:].sum(dim=-1)
                            cond_log_likelihoods[scale_idx].append(cond_scale_distance)
                        else:
                            # When scale_idx = 0, conditional distance is the same as overall distance
                            cond_log_likelihoods[scale_idx].append(neg_avg_distance.sum(dim=-1))
                        
                        start_idx = end_idx
                    
                    # Calculate overall negated average distance
                    total_distance = neg_avg_distance.sum(dim=-1)
                    log_likelihood_list.append(total_distance)
        
        # Concatenate all log likelihoods or negated distances
        log_likelihood_list = torch.cat(log_likelihood_list, dim=0)
        
        # Calculate predictions for each scale
        for scale_idx in range(len(patch_nums)):
            scale_log_likelihoods[scale_idx] = torch.cat(scale_log_likelihoods[scale_idx], dim=0)
            
            pred_idx_scale = torch.argmax(scale_log_likelihoods[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_scale = torch.tensor(class_indices[pred_idx_scale.item()], device=pred_idx_scale.device)
            else:
                pred_scale = pred_idx_scale
            
            # Update accuracies for this scale
            if pred_scale.item() == label.item():
                scale_correct[scale_idx] += 1
            scale_total[scale_idx] += 1
            
            # Save scale-specific results
            scale_json_fname = osp.join(layerwise_folder, f"{idx}_{scale_idx}-layer.json")
            scale_data = {
                f"pred_d{args.depth}": pred_scale.item(),
                "pred_idx": pred_idx_scale.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                f"target_log_likelihood_d{args.depth}": scale_log_likelihoods[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                f"log_likelihood_d{args.depth}": scale_log_likelihoods[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "scale_idx": scale_idx,
                "patch_size": patch_nums[scale_idx]
            }
            with open(scale_json_fname, "w") as f:
                json.dump(scale_data, f, indent=4)
        
        # Calculate predictions for accumulated likelihoods
        for scale_idx in range(len(patch_nums)):
            acc_log_likelihoods[scale_idx] = torch.cat(acc_log_likelihoods[scale_idx], dim=0)
            
            pred_idx_acc = torch.argmax(acc_log_likelihoods[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_acc = torch.tensor(class_indices[pred_idx_acc.item()], device=pred_idx_acc.device)
            else:
                pred_acc = pred_idx_acc
            
            # Update accuracies for accumulated likelihoods
            if pred_acc.item() == label.item():
                acc_correct[scale_idx] += 1
            acc_total[scale_idx] += 1
            
            # Save accumulated likelihood results
            acc_json_fname = osp.join(layer_acc_folder, f"{idx}_{scale_idx}-layer_acc.json")
            acc_data = {
                f"pred_d{args.depth}": pred_acc.item(),
                "pred_idx": pred_idx_acc.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                f"target_log_likelihood_d{args.depth}": acc_log_likelihoods[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                f"log_likelihood_d{args.depth}": acc_log_likelihoods[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "accumulated_to_scale_idx": scale_idx,
                "accumulated_to_patch_size": patch_nums[scale_idx]
            }
            with open(acc_json_fname, "w") as f:
                json.dump(acc_data, f, indent=4)
        
        # Calculate predictions for conditional likelihoods
        for scale_idx in range(len(patch_nums)):
            # Concatenate conditional log likelihoods
            cond_log_likelihoods[scale_idx] = torch.cat(cond_log_likelihoods[scale_idx], dim=0)
            
            # Make predictions based on conditional likelihoods
            pred_idx_cond = torch.argmax(cond_log_likelihoods[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_cond = torch.tensor(class_indices[pred_idx_cond.item()], device=pred_idx_cond.device)
            else:
                pred_cond = pred_idx_cond
            
            # Update conditional accuracies
            if pred_cond.item() == label.item():
                cond_correct[scale_idx] += 1
            cond_total[scale_idx] += 1
            
            # Save conditional likelihood results
            cond_json_fname = osp.join(layer_cond_folder, f"{idx}_{scale_idx}-layer_cond.json")
            cond_data = {
                f"pred_d{args.depth}": pred_cond.item(),
                "pred_idx": pred_idx_cond.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                f"target_log_likelihood_d{args.depth}": cond_log_likelihoods[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                f"log_likelihood_d{args.depth}": cond_log_likelihoods[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "conditioned_on_scale_idx": scale_idx,
                "conditioned_on_patch_size": patch_nums[scale_idx]
            }
            with open(cond_json_fname, "w") as f:
                json.dump(cond_data, f, indent=4)
        
        # Calculate overall predictions
        pred_idx = torch.argmax(log_likelihood_list[:-1])
        
        # For ImageNet-A, map the prediction index to the actual class index
        if args.dataset == "imagenet-a":
            pred = torch.tensor(class_indices[pred_idx.item()], device=pred_idx.device)
        else:
            pred = pred_idx

        # Update overall accuracies
        if pred.item() == label.item():
            correct += 1
        total += 1
        
        # Keep the original accuracy calculation for backward compatibility
        data = {
            "pred": pred.item(), 
            "label": label.item(),
            f"pred_d{args.depth}": pred.item(),
            "pred_idx": pred_idx.item() if args.dataset == "imagenet-a" else None,  # Store the index for ImageNet-A
            f"target_log_likelihood_d{args.depth}": log_likelihood_list[label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
            f"log_likelihood_d{args.depth}": log_likelihood_list.detach().cpu().tolist(),
            "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
            "explanation": "Lower L2 distance indicates better fit (values are negated for classification)" if args.mode == "l2_dist" else "Higher log likelihood indicates better fit"
        }
        with open(json_fname, "w") as f:
            json.dump(data, f, indent=4)
    
    if args.plot_kde:
        # ----- Overall KDE Plot: Plot one subplot per class -----
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        for cls in range(10):
            # Concatenate the probabilities for the given class.
            data = torch.cat(overall_class_probs[cls], dim=0).view(-1).detach().cpu().numpy()
            kde = gaussian_kde(data)
            xmin, xmax = 0, 0.2
            x_vals = np.linspace(xmin, xmax, 1000)
            axs[cls // 5, cls % 5].plot(x_vals, kde(x_vals), label=f'var_d{args.depth}')
            axs[cls // 5, cls % 5].set_title(f'Class {cls}')
            axs[cls // 5, cls % 5].set_xlabel("Value")
            axs[cls // 5, cls % 5].set_ylabel("Density")
            axs[cls // 5, cls % 5].legend()
        plt.suptitle("Overall KDE Plot Comparison by Class")
        overall_kde_fname = osp.join(run_folder, "overall_kde_by_class.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(overall_kde_fname)
        plt.close()
        # ---------------------------------------------------------

    # Plot token distance vs. probability for each scale and sample
    if args.plot_dist_kde and args.mode == "l2_dist":
        logging.info(f"Generating token distance vs. probability plots for each sample...")
        
        # Helper function to apply smoothing
        def apply_smoothing(y, method='savgol', window=15, polyorder=3, sigma=2):
            if method == 'savgol':
                if len(y) > window:  # Need sufficient points for Savitzky-Golay
                    return savgol_filter(y, window, polyorder)
                else:
                    return gaussian_filter1d(y, sigma)  # Fallback to Gaussian
            elif method == 'gaussian':
                return gaussian_filter1d(y, sigma)
            else:
                return y  # No smoothing
        
        # Process each sample
        for sample_idx in sample_scale_distances_probs.keys():
            # For each scale in this sample
            for scale_idx in range(len(patch_nums)):
                try:
                    # Concatenate all collected data for this scale and sample
                    all_distances = np.concatenate(sample_scale_distances_probs[sample_idx][scale_idx]['distances'])
                    all_probs = np.concatenate(sample_scale_distances_probs[sample_idx][scale_idx]['probs'])
                    
                    # Create a random subsample if there's too much data
                    max_points = 100000  # Limit sample size for efficiency
                    if len(all_distances) > max_points:
                        indices = np.random.choice(len(all_distances), max_points, replace=False)
                        all_distances = all_distances[indices]
                        all_probs = all_probs[indices]
                    
                    # Filter out extreme values
                    mask = (all_probs > 1e-10) & (all_distances < 50)
                    filtered_distances = all_distances[mask]
                    filtered_probs = all_probs[mask]
                    
                    # Create average probability vs. distance plot
                    plt.figure(figsize=(10, 6))
                    
                    # Create distance bins
                    max_dist = min(filtered_distances.max(), 30)  # Cap at 30 to focus on relevant range
                    bins = np.linspace(0, max_dist, 100)  # More bins for smoother curve
                    
                    # Compute average probability for each bin
                    avg_probs = []
                    bin_centers = []
                    
                    for i in range(len(bins)-1):
                        mask_bin = (filtered_distances >= bins[i]) & (filtered_distances < bins[i+1])
                        
                        if np.sum(mask_bin) > 0:
                            avg_probs.append(np.mean(filtered_probs[mask_bin]))
                        else:
                            avg_probs.append(np.nan)
                            
                        bin_centers.append((bins[i] + bins[i+1]) / 2)
                    
                    # Convert to numpy arrays for easier processing
                    bin_centers = np.array(bin_centers)
                    avg_probs = np.array(avg_probs)
                    
                    # Remove NaN values before smoothing
                    valid_indices = ~np.isnan(avg_probs)
                    
                    if np.sum(valid_indices) > 5:  # Need enough points for smoothing
                        # Apply smoothing to average probabilities
                        smooth_probs = np.full_like(avg_probs, np.nan)
                        smooth_probs[valid_indices] = apply_smoothing(
                            avg_probs[valid_indices], 
                            method='savgol' if np.sum(valid_indices) > 15 else 'gaussian'
                        )
                        
                        # Plot raw data as scatter and smoothed curve as line
                        plt.scatter(bin_centers[valid_indices], avg_probs[valid_indices], 
                                   s=10, alpha=0.4, color='blue', label=f'VAR D{args.depth} (raw)')
                        plt.plot(bin_centers[valid_indices], smooth_probs[valid_indices], 
                                'b-', linewidth=2, label=f'VAR D{args.depth} (smoothed)')
                    
                    plt.xlabel('Token Distance', fontsize=12)
                    plt.ylabel('Average Probability', fontsize=12)
                    plt.yscale('log')
                    plt.title(f'Sample {sample_idx}, Scale {scale_idx} (patches: {patch_nums[scale_idx]}x{patch_nums[scale_idx]})\nAvg Prob vs Distance', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(osp.join(dist_kde_folder, f"{sample_idx}_{scale_idx}-layer_prob_vs_dist.png"))
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Failed to create plot for sample {sample_idx}, scale {scale_idx}: {e}")
                    continue
                
        logging.info(f"Token distance vs. probability plots saved to: {dist_kde_folder}")

        # After processing individual samples, create overall plots across all samples
        logging.info(f"Generating overall token distance vs. probability plots across all samples...")
        for scale_idx in range(len(patch_nums)):
            try:
                # Concatenate all collected data for this scale across all samples
                all_distances = np.concatenate(overall_scale_distances_probs[scale_idx]['distances']) if overall_scale_distances_probs[scale_idx]['distances'] else np.array([])
                all_probs = np.concatenate(overall_scale_distances_probs[scale_idx]['probs']) if overall_scale_distances_probs[scale_idx]['probs'] else np.array([])
                
                if len(all_distances) == 0:
                    logging.warning(f"No data collected for scale {scale_idx} overall plot")
                    continue
                
                # Create a random subsample if there's too much data
                max_points = 500000  # Increased for overall plots
                if len(all_distances) > max_points:
                    indices = np.random.choice(len(all_distances), max_points, replace=False)
                    all_distances = all_distances[indices]
                    all_probs = all_probs[indices]
                
                # Filter out extreme values
                mask = (all_probs > 1e-10) & (all_distances < 50)
                filtered_distances = all_distances[mask]
                filtered_probs = all_probs[mask]
                
                # Create average probability vs. distance plot
                plt.figure(figsize=(12, 8))
                
                # Create distance bins
                max_dist = min(filtered_distances.max(), 30)
                bins = np.linspace(0, max_dist, 150)  # More bins for smoother curve in overall plot
                
                # Compute average probability for each bin
                avg_probs = []
                bin_centers = []
                bin_counts = []  # To track number of points in each bin
                
                for i in range(len(bins)-1):
                    mask_bin = (filtered_distances >= bins[i]) & (filtered_distances < bins[i+1])
                    
                    bin_count = np.sum(mask_bin)
                    bin_counts.append(bin_count)
                    
                    if bin_count > 0:
                        avg_probs.append(np.mean(filtered_probs[mask_bin]))
                    else:
                        avg_probs.append(np.nan)
                        
                    bin_centers.append((bins[i] + bins[i+1]) / 2)
                
                # Convert to numpy arrays for easier processing
                bin_centers = np.array(bin_centers)
                avg_probs = np.array(avg_probs)
                bin_counts = np.array(bin_counts)
                
                # Remove NaN values before smoothing
                valid_indices = ~np.isnan(avg_probs)
                
                # Main plot
                if np.sum(valid_indices) > 5:
                    # Apply smoothing to average probabilities with more points for an overall smoother curve
                    window = 25 if np.sum(valid_indices) > 50 else 15
                    smooth_probs = np.full_like(avg_probs, np.nan)
                    smooth_probs[valid_indices] = apply_smoothing(
                        avg_probs[valid_indices], 
                        method='savgol' if np.sum(valid_indices) > window else 'gaussian',
                        window=window
                    )
                    
                    # Plot raw data with alpha based on bin count (more points = more opaque)
                    max_count = np.max(bin_counts[valid_indices]) if np.sum(valid_indices) > 0 else 1
                    alphas = np.minimum(0.7, 0.1 + 0.6 * bin_counts[valid_indices] / max_count)
                    
                    # Plot scatter with varying alpha
                    for i, idx in enumerate(np.where(valid_indices)[0]):
                        plt.scatter(bin_centers[idx], avg_probs[idx], 
                                   s=20, alpha=alphas[i], color='blue', edgecolor='none')
                    
                    # Plot the smoothed curve
                    plt.plot(bin_centers[valid_indices], smooth_probs[valid_indices], 
                            'b-', linewidth=3, label=f'VAR D{args.depth} (smoothed)')
                
                # Add a fit line showing expected exponential relationship for reference
                if np.sum(valid_indices) > 10:
                    # Create reference exponential decay curve for comparison
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(x, a, b):
                        return a * np.exp(-b * x)
                    
                    try:
                        # Get valid data points
                        x_data = bin_centers[valid_indices]
                        y_data = avg_probs[valid_indices]
                        
                        # Fit exponential decay
                        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[y_data[0], 0.5], maxfev=2000)
                        
                        # Plot fitted curve
                        x_fit = np.linspace(0, max_dist, 1000)
                        y_fit = exp_decay(x_fit, *popt)
                        plt.plot(x_fit, y_fit, 'b--', linewidth=1.5, alpha=0.7, 
                                label=f'Exp fit D{args.depth}: {popt[0]:.2e}·exp(-{popt[1]:.2f}·x)')
                    except Exception as e:
                        logging.warning(f"Could not fit exponential curve: {e}")
                
                plt.xlabel('Token Distance', fontsize=14)
                plt.ylabel('Average Probability', fontsize=14)
                plt.yscale('log')
                plt.title(f'Overall Scale {scale_idx} (patches: {patch_nums[scale_idx]}x{patch_nums[scale_idx]})\nAvg Prob vs Distance Across All Samples', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(osp.join(dist_kde_folder, f"overall_{scale_idx}-layer_prob_vs_dist.png"), dpi=300)
                plt.close()
                
                # Create a second plot showing data density heatmap
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Create 2D histograms
                    h = plt.hist2d(filtered_distances, np.log10(filtered_probs), 
                                      bins=[150, 100], cmap='Blues', alpha=0.6, density=True)
                    
                    # Overlay with the smoothed average probability curves
                    if np.sum(valid_indices) > 5:
                        plt.plot(bin_centers[valid_indices], np.log10(smooth_probs[valid_indices]), 
                                'b-', linewidth=3, label=f'VAR D{args.depth} (smoothed)')
                    
                    plt.colorbar(h[3], label='Normalized Density')
                    plt.xlabel('Token Distance', fontsize=14)
                    plt.ylabel('Log10(Probability)', fontsize=14)
                    plt.title(f'Overall Scale {scale_idx} (patches: {patch_nums[scale_idx]}x{patch_nums[scale_idx]})\nData Density Heatmap', fontsize=16)
                    plt.legend(fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(osp.join(dist_kde_folder, f"overall_{scale_idx}-layer_density_heatmap.png"), dpi=300)
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Failed to create density heatmap for scale {scale_idx}: {e}")
                
            except Exception as e:
                logging.error(f"Failed to create overall plot for scale {scale_idx}: {e}")
                continue

    metric_name = "Average L2 Distance" if args.mode == "l2_dist" else "Log Likelihood"
    logging.info(f"\nOverall Accuracies using {metric_name} for Classification:")
    logging.info(f"Overall Accuracy (d{args.depth}): {100 * correct / total:.2f}%")
    
    logging.info(f"\nPer-Scale Accuracies using {metric_name}:")
    for scale_idx in range(len(patch_nums)):
        logging.info(f"\nScale {scale_idx} (patch size: {patch_nums[scale_idx]}):")
        logging.info(f"  d{args.depth} Accuracy: {100 * scale_correct[scale_idx] / scale_total[scale_idx]:.2f}%")
    
    logging.info(f"\nAccumulated {metric_name} Accuracies (first scale_idx layers):")
    for scale_idx in range(len(patch_nums)):
        logging.info(f"\nFirst {scale_idx+1} layers:")
        logging.info(f"  d{args.depth} Accuracy: {100 * acc_correct[scale_idx] / acc_total[scale_idx]:.2f}%")
    
    logging.info(f"\nConditional {metric_name} Accuracies (excluding first scale_idx layers):")
    for scale_idx in range(len(patch_nums)):
        condition_desc = "all" if scale_idx == 0 else f"first {scale_idx}"
        logging.info(f"\nConditioned on {condition_desc} layers:")
        logging.info(f"  d{args.depth} Accuracy: {100 * cond_correct[scale_idx] / cond_total[scale_idx]:.2f}%")
    
    if args.mode == "l2_dist":
        logging.info("\nNote: For L2 distance classification, smaller distances indicate higher confidence in class prediction.")
        logging.info("      We negate the distances so larger values are better, compatible with argmax-based classification.")


if __name__ == "__main__":
    main()