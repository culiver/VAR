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

MODEL_DEPTH = 16  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
LOG_DIR = "./model_size_analysis"

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
            data_path=data_path.replace("imagenet10", "imagenet"),
            final_reso=256,
            dataset_type=args.dataset.replace("imagenet10", "imagenet")
        )
        class_indices = [i for i in range(num_classes)]
    
    ld_val = DataLoader(dataset_val, num_workers=0, pin_memory=True, batch_size=1, shuffle=False, drop_last=False)
    del dataset_val

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    var_d16_ckpt = "var_d16.pth"
    var_d30_ckpt = "var_d30.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_d16_ckpt):
        os.system(f"wget {hf_home}/{var_d16_ckpt}")
    if not osp.exists(var_d30_ckpt):
        os.system(f"wget {hf_home}/{var_d30_ckpt}")

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    patch_nums_square_cumsum = np.cumsum(np.array(patch_nums)**2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "vae" not in globals() or "var" not in globals():
        vae, var_d16 = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums, num_classes=1000, depth=16, shared_aln=False
        )
        vae, var_d30 = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums, num_classes=1000, depth=30, shared_aln=False
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var_d16.load_state_dict(torch.load(var_d16_ckpt, map_location="cpu"), strict=True)
    var_d30.load_state_dict(torch.load(var_d30_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var_d16.eval(), var_d30.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var_d16.parameters():
        p.requires_grad_(False)
    for p in var_d30.parameters():
        p.requires_grad_(False)
    logging.info("prepare finished.")
    var_d16.cond_drop_rate = 0
    var_d30.cond_drop_rate = 0

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
    correct_d16 = 0
    correct_d30 = 0
    total_d16 = 0
    total_d30 = 0
    
    # Track accuracies for each scale
    scale_correct_d16 = {i: 0 for i in range(len(patch_nums))}
    scale_total_d16 = {i: 0 for i in range(len(patch_nums))}
    scale_correct_d30 = {i: 0 for i in range(len(patch_nums))}
    scale_total_d30 = {i: 0 for i in range(len(patch_nums))}
    
    # Track accuracies for accumulated likelihoods
    acc_correct_d16 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}  # scale_idx from 0 to 9
    acc_total_d16 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    acc_correct_d30 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    acc_total_d30 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    
    # Track accuracies for conditional likelihoods (excluding first scale_idx scales)
    cond_correct_d16 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    cond_total_d16 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    cond_correct_d30 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}
    cond_total_d30 = {scale_idx: 0 for scale_idx in range(len(patch_nums))}

    # Dictionaries to store probabilities across all samples per class.
    if args.dataset == "imagenet10":
        overall_class_probs_d16 = {cls: [] for cls in [i for i in range(num_classes)][:10] + [1000]}
        overall_class_probs_d30 = {cls: [] for cls in [i for i in range(num_classes)][:10] + [1000]}
    elif args.dataset == "imagenet-a":
        # For ImageNet-A, use the actual class indices plus the unconditional class (1000)
        overall_class_probs_d16 = {cls: [] for cls in class_indices + [1000]}
        overall_class_probs_d30 = {cls: [] for cls in class_indices + [1000]}
    else:
        overall_class_probs_d16 = {cls: [] for cls in [i for i in range(num_classes)] + [1000]}
        overall_class_probs_d30 = {cls: [] for cls in [i for i in range(num_classes)] + [1000]}
    
    # Precompute embedding distances if using l2_dist mode
    if args.mode == "l2_dist":
        # Get embedding weights from vae model
        emb_weight_d16 = vae.quantize.embedding.weight  # (V, D)
        # Compute pairwise L2 distances between all embeddings
        dists_d16 = torch.cdist(emb_weight_d16, emb_weight_d16, p=2)  # (V, V)
        
        # For d30 model, use the same VAE embeddings since they share the same VAE
        dists_d30 = dists_d16.clone()
        
        logging.info(f"Precomputed embedding distances with shape: {dists_d16.shape}")
        if args.top_k is not None:
            logging.info(f"Using top {args.top_k} most probable tokens for L2 distance calculation")

    # For plotting token distance vs. probability relation
    if args.plot_dist_kde and args.mode == "l2_dist":
        # For each scale, store distances and probabilities for each sample
        # Key structure: {sample_idx: {scale_idx: {'distances': [], 'probs': []}}}
        sample_scale_distances_probs_d16 = {}
        sample_scale_distances_probs_d30 = {}
        
        # For overall plots across samples
        overall_scale_distances_probs_d16 = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        overall_scale_distances_probs_d30 = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        
        # For wrong class condition analysis
        sample_scale_distances_probs_d16_wrong = {}
        sample_scale_distances_probs_d30_wrong = {}
        overall_scale_distances_probs_d16_wrong = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        overall_scale_distances_probs_d30_wrong = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
        
        # Create folder for unified distance analysis that combines correct and wrong conditions
        dist_analysis_folder = osp.join(LOG_DIR, args.dataset, args.mode, name, "dist_analysis") if len(extra) == 0 else osp.join(LOG_DIR, args.dataset, args.mode, name + f"_{extra}", "dist_analysis")
        os.makedirs(dist_analysis_folder, exist_ok=True)
        
        logging.info(f"Will generate unified token distance vs. probability plots in: {dist_analysis_folder}")
        
        if args.plot_dist_kde:
            logging.info("Generating distance vs. probability plots...")
            logging.info(f"Creating UNIFIED comparison plots in: {dist_analysis_folder}")
            logging.info("These plots will combine both correct and wrong class condition results for easier comparison.")

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
        log_likelihood_list_d16 = []
        log_likelihood_list_d30 = []
        # For per-sample plotting if needed.
        prob_list_d16 = []
        prob_list_d30 = []
        
        # Store log likelihoods or distances for each scale
        scale_log_likelihoods_d16 = {i: [] for i in range(len(patch_nums))}
        scale_log_likelihoods_d30 = {i: [] for i in range(len(patch_nums))}
        
        # Store accumulated log likelihoods or distances
        acc_log_likelihoods_d16 = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        acc_log_likelihoods_d30 = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        
        # Store conditional log likelihoods or distances (excluding first scale_idx scales)
        cond_log_likelihoods_d16 = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        cond_log_likelihoods_d30 = {scale_idx: [] for scale_idx in range(len(patch_nums))}
        
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

                uncondition_logits_d16 = var_d16.forward(uncondition_label, x_BLCv_wo_first_l)
                uncondition_logits_d30 = var_d30.forward(uncondition_label, x_BLCv_wo_first_l)

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

                logits_d16 = var_d16.forward(label_B, x_BLCv_wo_first_l)
                logits_d30 = var_d30.forward(label_B, x_BLCv_wo_first_l)

                if args.cfg > 0:
                    logits_d16 = (1 + t) * logits_d16 - t * uncondition_logits_d16
                    logits_d30 = (1 + t) * logits_d30 - t * uncondition_logits_d30

                log_probs_d16 = torch.nn.functional.log_softmax(logits_d16, dim=-1)
                probs_d16 = torch.nn.functional.softmax(logits_d16, dim=-1)
                log_probs_d30 = torch.nn.functional.log_softmax(logits_d30, dim=-1)
                probs_d30 = torch.nn.functional.softmax(logits_d30, dim=-1)

                gt_probs_d16 = probs_d16.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)
                gt_probs_d30 = probs_d30.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)

                # Collect token distances and probabilities for KDE plotting
                if args.plot_dist_kde and args.mode == "l2_dist":
                    batch_size = probs_d16.shape[0]
                    for b in range(batch_size):
                        start_idx = 0
                        
                        if class_labels[b] == 1000:
                            continue

                        # Process based on whether the class label matches or not
                        elif class_labels[b] == label.item():
                            # Correct class condition - use existing code
                            # Initialize data structure for this sample if needed
                            if idx not in sample_scale_distances_probs_d16:
                                sample_scale_distances_probs_d16[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                sample_scale_distances_probs_d30[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                
                            for scale_idx, num_patches in enumerate(patch_nums):
                                num_patches_square = num_patches * num_patches
                                end_idx = start_idx + num_patches_square
                                
                                # Get ground truth tokens for this scale
                                scale_gt_tokens = gt_tokens[b, start_idx:end_idx]  # (num_patches_square)
                                
                                # Get distances from gt_tokens to all other tokens
                                scale_gt_distances_d16 = dists_d16[scale_gt_tokens]  # (num_patches_square, V)
                                scale_gt_distances_d30 = dists_d30[scale_gt_tokens]  # (num_patches_square, V)
                                
                                # Get probabilities for this scale
                                scale_probs_d16 = probs_d16[b, start_idx:end_idx]  # (num_patches_square, V)
                                scale_probs_d30 = probs_d30[b, start_idx:end_idx]  # (num_patches_square, V)
                                
                                # Flatten and collect
                                flat_distances_d16 = scale_gt_distances_d16.view(-1).detach().cpu().numpy()
                                flat_probs_d16 = scale_probs_d16.view(-1).detach().cpu().numpy()
                                flat_distances_d30 = scale_gt_distances_d30.view(-1).detach().cpu().numpy()
                                flat_probs_d30 = scale_probs_d30.view(-1).detach().cpu().numpy()
                                
                                # Store for later plotting
                                sample_scale_distances_probs_d16[idx][scale_idx]['distances'].append(flat_distances_d16)
                                sample_scale_distances_probs_d16[idx][scale_idx]['probs'].append(flat_probs_d16)
                                sample_scale_distances_probs_d30[idx][scale_idx]['distances'].append(flat_distances_d30)
                                sample_scale_distances_probs_d30[idx][scale_idx]['probs'].append(flat_probs_d30)
                                
                                # Also store for overall plots
                                overall_scale_distances_probs_d16[scale_idx]['distances'].append(flat_distances_d16)
                                overall_scale_distances_probs_d16[scale_idx]['probs'].append(flat_probs_d16)
                                overall_scale_distances_probs_d30[scale_idx]['distances'].append(flat_distances_d30)
                                overall_scale_distances_probs_d30[scale_idx]['probs'].append(flat_probs_d30)
                                
                                start_idx = end_idx
                                
                        else:
                            # Wrong class condition - store separately
                            # Initialize data structure for this sample if needed
                            if idx not in sample_scale_distances_probs_d16_wrong:
                                sample_scale_distances_probs_d16_wrong[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                sample_scale_distances_probs_d30_wrong[idx] = {scale_idx: {'distances': [], 'probs': []} for scale_idx in range(len(patch_nums))}
                                
                            for scale_idx, num_patches in enumerate(patch_nums):
                                num_patches_square = num_patches * num_patches
                                end_idx = start_idx + num_patches_square
                                
                                # Get ground truth tokens for this scale
                                scale_gt_tokens = gt_tokens[b, start_idx:end_idx]  # (num_patches_square)
                                
                                # Get distances from gt_tokens to all other tokens
                                scale_gt_distances_d16 = dists_d16[scale_gt_tokens]  # (num_patches_square, V)
                                scale_gt_distances_d30 = dists_d30[scale_gt_tokens]  # (num_patches_square, V)
                                
                                # Get probabilities for this scale
                                scale_probs_d16 = probs_d16[b, start_idx:end_idx]  # (num_patches_square, V)
                                scale_probs_d30 = probs_d30[b, start_idx:end_idx]  # (num_patches_square, V)
                                
                                # Flatten and collect
                                flat_distances_d16 = scale_gt_distances_d16.view(-1).detach().cpu().numpy()
                                flat_probs_d16 = scale_probs_d16.view(-1).detach().cpu().numpy()
                                flat_distances_d30 = scale_gt_distances_d30.view(-1).detach().cpu().numpy()
                                flat_probs_d30 = scale_probs_d30.view(-1).detach().cpu().numpy()
                                
                                # Store wrong class data for later plotting 
                                sample_scale_distances_probs_d16_wrong[idx][scale_idx]['distances'].append(flat_distances_d16)
                                sample_scale_distances_probs_d16_wrong[idx][scale_idx]['probs'].append(flat_probs_d16)
                                sample_scale_distances_probs_d30_wrong[idx][scale_idx]['distances'].append(flat_distances_d30)
                                sample_scale_distances_probs_d30_wrong[idx][scale_idx]['probs'].append(flat_probs_d30)
                                
                                # Also store for overall wrong condition plots
                                overall_scale_distances_probs_d16_wrong[scale_idx]['distances'].append(flat_distances_d16)
                                overall_scale_distances_probs_d16_wrong[scale_idx]['probs'].append(flat_probs_d16)
                                overall_scale_distances_probs_d30_wrong[scale_idx]['distances'].append(flat_distances_d30)
                                overall_scale_distances_probs_d30_wrong[scale_idx]['probs'].append(flat_probs_d30)
                                
                                start_idx = end_idx

                # Append per-sample lists.
                prob_list_d16.append(gt_probs_d16)
                prob_list_d30.append(gt_probs_d30)

                # Also accumulate into the overall dictionaries per class.
                # Here, the i-th element in the batch corresponds to class_labels[i].
                for i, cls in enumerate(class_labels):
                    overall_class_probs_d16[cls].append(gt_probs_d16[i].detach().cpu())
                    overall_class_probs_d30[cls].append(gt_probs_d30[i].detach().cpu())

                if args.mode == "var":
                    # Calculate log likelihood for each scale
                    gt_log_probs_d16 = log_probs_d16.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)
                    gt_log_probs_d30 = log_probs_d30.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)
                    
                    start_idx = 0
                    for scale_idx, num_patches in enumerate(patch_nums):
                        num_patches_square = num_patches * num_patches
                        end_idx = start_idx + num_patches_square
                        
                        # Sum over patches in this scale
                        scale_log_likelihood_d16 = gt_log_probs_d16[:, start_idx:end_idx].sum(dim=-1)
                        scale_log_likelihood_d30 = gt_log_probs_d30[:, start_idx:end_idx].sum(dim=-1)
                        
                        scale_log_likelihoods_d16[scale_idx].append(scale_log_likelihood_d16)
                        scale_log_likelihoods_d30[scale_idx].append(scale_log_likelihood_d30)
                        
                        # Calculate accumulated likelihoods up to this scale
                        acc_scale_log_likelihood_d16 = gt_log_probs_d16[:, :end_idx].sum(dim=-1)
                        acc_scale_log_likelihood_d30 = gt_log_probs_d30[:, :end_idx].sum(dim=-1)
                        acc_log_likelihoods_d16[scale_idx].append(acc_scale_log_likelihood_d16)
                        acc_log_likelihoods_d30[scale_idx].append(acc_scale_log_likelihood_d30)
                        
                        # Calculate conditional likelihoods (excluding first scale_idx scales)
                        if scale_idx > 0:
                            start_cond_idx = patch_nums_square_cumsum[scale_idx - 1]
                            cond_scale_log_likelihood_d16 = gt_log_probs_d16[:, start_cond_idx:].sum(dim=-1)
                            cond_scale_log_likelihood_d30 = gt_log_probs_d30[:, start_cond_idx:].sum(dim=-1)
                            cond_log_likelihoods_d16[scale_idx].append(cond_scale_log_likelihood_d16)
                            cond_log_likelihoods_d30[scale_idx].append(cond_scale_log_likelihood_d30)
                        else:
                            # When scale_idx = 0, conditional likelihood is the same as overall likelihood
                            cond_log_likelihoods_d16[scale_idx].append(gt_log_probs_d16.sum(dim=-1))
                            cond_log_likelihoods_d30[scale_idx].append(gt_log_probs_d30.sum(dim=-1))
                        
                        start_idx = end_idx
                    
                    # Calculate overall log likelihood
                    log_likelihood_d16 = gt_log_probs_d16.sum(dim=-1)
                    log_likelihood_d30 = gt_log_probs_d30.sum(dim=-1)
                    log_likelihood_list_d16.append(log_likelihood_d16)
                    log_likelihood_list_d30.append(log_likelihood_d30)
                
                elif args.mode == "l2_dist":
                    # For L2 distance-based classification, we compute:
                    # average distance = sum(dists[gt_token, i] * probs[i]) for all tokens i
                    
                    # Get distances for ground-truth tokens to all other tokens
                    gt_distances_d16 = dists_d16[gt_tokens]  # (B, L, V)
                    gt_distances_d30 = dists_d30[gt_tokens]  # (B, L, V)
                    
                    if args.top_k is not None:
                        # Get top k probabilities and their indices
                        top_k_probs_d16, top_k_indices_d16 = torch.topk(probs_d16, k=args.top_k, dim=-1)  # (B, L, k)
                        top_k_probs_d30, top_k_indices_d30 = torch.topk(probs_d30, k=args.top_k, dim=-1)  # (B, L, k)
                        
                        # Gather distances for top k tokens
                        top_k_distances_d16 = torch.gather(gt_distances_d16, dim=-1, index=top_k_indices_d16)  # (B, L, k)
                        top_k_distances_d30 = torch.gather(gt_distances_d30, dim=-1, index=top_k_indices_d30)  # (B, L, k)
                        
                        # Normalize top k probabilities
                        top_k_probs_d16 = top_k_probs_d16 / top_k_probs_d16.sum(dim=-1, keepdim=True)
                        top_k_probs_d30 = top_k_probs_d30 / top_k_probs_d30.sum(dim=-1, keepdim=True)
                        
                        # Compute weighted average distance using only top k tokens
                        avg_distance_d16 = (top_k_distances_d16 * top_k_probs_d16).sum(dim=-1)  # (B, L)
                        avg_distance_d30 = (top_k_distances_d30 * top_k_probs_d30).sum(dim=-1)  # (B, L)
                    else:
                        # Original implementation using all tokens
                        avg_distance_d16 = (gt_distances_d16 * probs_d16).sum(dim=-1)  # (B, L)
                        avg_distance_d30 = (gt_distances_d30 * probs_d30).sum(dim=-1)  # (B, L)
                    
                    # For classification, smaller distance is better
                    # We negate the distance so that larger (less negative) values are better
                    # This makes it compatible with the existing max-based classification logic
                    neg_avg_distance_d16 = -avg_distance_d16
                    neg_avg_distance_d30 = -avg_distance_d30
                    
                    # Calculate per-scale average distances
                    start_idx = 0
                    for scale_idx, num_patches in enumerate(patch_nums):
                        num_patches_square = num_patches * num_patches
                        end_idx = start_idx + num_patches_square
                        
                        # Sum over patches in this scale (smaller sum = better)
                        scale_distance_d16 = neg_avg_distance_d16[:, start_idx:end_idx].sum(dim=-1)
                        scale_distance_d30 = neg_avg_distance_d30[:, start_idx:end_idx].sum(dim=-1)
                        
                        scale_log_likelihoods_d16[scale_idx].append(scale_distance_d16)
                        scale_log_likelihoods_d30[scale_idx].append(scale_distance_d30)
                        
                        # Calculate accumulated distances up to this scale
                        acc_scale_distance_d16 = neg_avg_distance_d16[:, :end_idx].sum(dim=-1)
                        acc_scale_distance_d30 = neg_avg_distance_d30[:, :end_idx].sum(dim=-1)
                        acc_log_likelihoods_d16[scale_idx].append(acc_scale_distance_d16)
                        acc_log_likelihoods_d30[scale_idx].append(acc_scale_distance_d30)
                        
                        # Calculate conditional distances (excluding first scale_idx scales)
                        if scale_idx > 0:
                            start_cond_idx = patch_nums_square_cumsum[scale_idx - 1]
                            cond_scale_distance_d16 = neg_avg_distance_d16[:, start_cond_idx:].sum(dim=-1)
                            cond_scale_distance_d30 = neg_avg_distance_d30[:, start_cond_idx:].sum(dim=-1)
                            cond_log_likelihoods_d16[scale_idx].append(cond_scale_distance_d16)
                            cond_log_likelihoods_d30[scale_idx].append(cond_scale_distance_d30)
                        else:
                            # When scale_idx = 0, conditional distance is the same as overall distance
                            cond_log_likelihoods_d16[scale_idx].append(neg_avg_distance_d16.sum(dim=-1))
                            cond_log_likelihoods_d30[scale_idx].append(neg_avg_distance_d30.sum(dim=-1))
                        
                        start_idx = end_idx
                    
                    # Calculate overall negated average distance
                    total_distance_d16 = neg_avg_distance_d16.sum(dim=-1)
                    total_distance_d30 = neg_avg_distance_d30.sum(dim=-1)
                    log_likelihood_list_d16.append(total_distance_d16)
                    log_likelihood_list_d30.append(total_distance_d30)
        
        # Concatenate all log likelihoods or negated distances
        log_likelihood_list_d16 = torch.cat(log_likelihood_list_d16, dim=0)
        log_likelihood_list_d30 = torch.cat(log_likelihood_list_d30, dim=0)
        
        # Calculate predictions for each scale
        for scale_idx in range(len(patch_nums)):
            scale_log_likelihoods_d16[scale_idx] = torch.cat(scale_log_likelihoods_d16[scale_idx], dim=0)
            scale_log_likelihoods_d30[scale_idx] = torch.cat(scale_log_likelihoods_d30[scale_idx], dim=0)
            
            pred_idx_d16_scale = torch.argmax(scale_log_likelihoods_d16[scale_idx][:-1])
            pred_idx_d30_scale = torch.argmax(scale_log_likelihoods_d30[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_d16_scale = torch.tensor(class_indices[pred_idx_d16_scale.item()], device=pred_idx_d16_scale.device)
                pred_d30_scale = torch.tensor(class_indices[pred_idx_d30_scale.item()], device=pred_idx_d30_scale.device)
            else:
                pred_d16_scale = pred_idx_d16_scale
                pred_d30_scale = pred_idx_d30_scale
            
            # Update accuracies for this scale
            if pred_d16_scale.item() == label.item():
                scale_correct_d16[scale_idx] += 1
            scale_total_d16[scale_idx] += 1
            if pred_d30_scale.item() == label.item():
                scale_correct_d30[scale_idx] += 1
            scale_total_d30[scale_idx] += 1
            
            # Save scale-specific results
            scale_json_fname = osp.join(layerwise_folder, f"{idx}_{scale_idx}-layer.json")
            scale_data = {
                "pred_d16": pred_d16_scale.item(),
                "pred_d30": pred_d30_scale.item(),
                "pred_idx_d16": pred_idx_d16_scale.item() if args.dataset == "imagenet-a" else None,
                "pred_idx_d30": pred_idx_d30_scale.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                "target_log_likelihood_d16": scale_log_likelihoods_d16[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "target_log_likelihood_d30": scale_log_likelihoods_d30[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "log_likelihood_d16": scale_log_likelihoods_d16[scale_idx].detach().cpu().tolist(),
                "log_likelihood_d30": scale_log_likelihoods_d30[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "scale_idx": scale_idx,
                "patch_size": patch_nums[scale_idx]
            }
            with open(scale_json_fname, "w") as f:
                json.dump(scale_data, f, indent=4)
        
        # Calculate predictions for accumulated likelihoods
        for scale_idx in range(len(patch_nums)):
            acc_log_likelihoods_d16[scale_idx] = torch.cat(acc_log_likelihoods_d16[scale_idx], dim=0)
            acc_log_likelihoods_d30[scale_idx] = torch.cat(acc_log_likelihoods_d30[scale_idx], dim=0)
            
            pred_idx_d16_acc = torch.argmax(acc_log_likelihoods_d16[scale_idx][:-1])
            pred_idx_d30_acc = torch.argmax(acc_log_likelihoods_d30[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_d16_acc = torch.tensor(class_indices[pred_idx_d16_acc.item()], device=pred_idx_d16_acc.device)
                pred_d30_acc = torch.tensor(class_indices[pred_idx_d30_acc.item()], device=pred_idx_d30_acc.device)
            else:
                pred_d16_acc = pred_idx_d16_acc
                pred_d30_acc = pred_idx_d30_acc
            
            # Update accuracies for accumulated likelihoods
            if pred_d16_acc.item() == label.item():
                acc_correct_d16[scale_idx] += 1
            acc_total_d16[scale_idx] += 1
            if pred_d30_acc.item() == label.item():
                acc_correct_d30[scale_idx] += 1
            acc_total_d30[scale_idx] += 1
            
            # Save accumulated likelihood results
            acc_json_fname = osp.join(layer_acc_folder, f"{idx}_{scale_idx}-layer_acc.json")
            acc_data = {
                "pred_d16": pred_d16_acc.item(),
                "pred_d30": pred_d30_acc.item(),
                "pred_idx_d16": pred_idx_d16_acc.item() if args.dataset == "imagenet-a" else None,
                "pred_idx_d30": pred_idx_d30_acc.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                "target_log_likelihood_d16": acc_log_likelihoods_d16[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "target_log_likelihood_d30": acc_log_likelihoods_d30[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "log_likelihood_d16": acc_log_likelihoods_d16[scale_idx].detach().cpu().tolist(),
                "log_likelihood_d30": acc_log_likelihoods_d30[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "accumulated_to_scale_idx": scale_idx,
                "accumulated_to_patch_size": patch_nums[scale_idx]
            }
            with open(acc_json_fname, "w") as f:
                json.dump(acc_data, f, indent=4)
        
        # Calculate predictions for conditional likelihoods
        for scale_idx in range(len(patch_nums)):
            # Concatenate conditional log likelihoods
            cond_log_likelihoods_d16[scale_idx] = torch.cat(cond_log_likelihoods_d16[scale_idx], dim=0)
            cond_log_likelihoods_d30[scale_idx] = torch.cat(cond_log_likelihoods_d30[scale_idx], dim=0)
            
            # Make predictions based on conditional likelihoods
            pred_idx_d16_cond = torch.argmax(cond_log_likelihoods_d16[scale_idx][:-1])
            pred_idx_d30_cond = torch.argmax(cond_log_likelihoods_d30[scale_idx][:-1])
            
            # For ImageNet-A, map the prediction index to the actual class index
            if args.dataset == "imagenet-a":
                pred_d16_cond = torch.tensor(class_indices[pred_idx_d16_cond.item()], device=pred_idx_d16_cond.device)
                pred_d30_cond = torch.tensor(class_indices[pred_idx_d30_cond.item()], device=pred_idx_d30_cond.device)
            else:
                pred_d16_cond = pred_idx_d16_cond
                pred_d30_cond = pred_idx_d30_cond
            
            # Update conditional accuracies
            if pred_d16_cond.item() == label.item():
                cond_correct_d16[scale_idx] += 1
            cond_total_d16[scale_idx] += 1
            if pred_d30_cond.item() == label.item():
                cond_correct_d30[scale_idx] += 1
            cond_total_d30[scale_idx] += 1
            
            # Save conditional likelihood results
            cond_json_fname = osp.join(layer_cond_folder, f"{idx}_{scale_idx}-layer_cond.json")
            cond_data = {
                "pred_d16": pred_d16_cond.item(),
                "pred_d30": pred_d30_cond.item(),
                "pred_idx_d16": pred_idx_d16_cond.item() if args.dataset == "imagenet-a" else None,
                "pred_idx_d30": pred_idx_d30_cond.item() if args.dataset == "imagenet-a" else None,
                "label": label.item(),
                "target_log_likelihood_d16": cond_log_likelihoods_d16[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "target_log_likelihood_d30": cond_log_likelihoods_d30[scale_idx][label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
                "log_likelihood_d16": cond_log_likelihoods_d16[scale_idx].detach().cpu().tolist(),
                "log_likelihood_d30": cond_log_likelihoods_d30[scale_idx].detach().cpu().tolist(),
                "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
                "conditioned_on_scale_idx": scale_idx,
                "conditioned_on_patch_size": patch_nums[scale_idx]
            }
            with open(cond_json_fname, "w") as f:
                json.dump(cond_data, f, indent=4)
        
        # Calculate overall predictions
        pred_idx_d16 = torch.argmax(log_likelihood_list_d16[:-1])
        pred_idx_d30 = torch.argmax(log_likelihood_list_d30[:-1])
        
        # For ImageNet-A, map the prediction index to the actual class index
        if args.dataset == "imagenet-a":
            pred_d16 = torch.tensor(class_indices[pred_idx_d16.item()], device=pred_idx_d16.device)
            pred_d30 = torch.tensor(class_indices[pred_idx_d30.item()], device=pred_idx_d30.device)
        else:
            pred_d16 = pred_idx_d16
            pred_d30 = pred_idx_d30

        # Update overall accuracies
        if pred_d16.item() == label.item():
            correct_d16 += 1
        total_d16 += 1
        if pred_d30.item() == label.item():
            correct_d30 += 1
        total_d30 += 1
        
        # Keep the original accuracy calculation for backward compatibility
        pred = pred_d16  # Using d16 for the original prediction
        data = {
            "pred": pred.item(), 
            "label": label.item(),
            "pred_d16": pred_d16.item(),
            "pred_d30": pred_d30.item(),
            "pred_idx_d16": pred_idx_d16.item() if args.dataset == "imagenet-a" else None,  # Store the index for ImageNet-A
            "pred_idx_d30": pred_idx_d30.item() if args.dataset == "imagenet-a" else None,  # Store the index for ImageNet-A
            "target_log_likelihood_d16": log_likelihood_list_d16[label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
            "target_log_likelihood_d30": log_likelihood_list_d30[label.item() if args.dataset != "imagenet-a" else class_indices.index(label.item())].item(),
            "log_likelihood_d16": log_likelihood_list_d16.detach().cpu().tolist(),
            "log_likelihood_d30": log_likelihood_list_d30.detach().cpu().tolist(),
            "metric_type": "negative_l2_distance" if args.mode == "l2_dist" else "log_likelihood",
            "explanation": "Lower L2 distance indicates better fit (values are negated for classification)" if args.mode == "l2_dist" else "Higher log likelihood indicates better fit"
        }
        with open(json_fname, "w") as f:
            json.dump(data, f, indent=4)
        if pred.item() == label.item():
            correct += 1
        total += 1
    
    if args.plot_kde:
        # ----- Overall KDE Plot: Plot one subplot per class -----
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        for cls in range(10):
            # Concatenate the probabilities for the given class.
            data_d16 = torch.cat(overall_class_probs_d16[cls], dim=0).view(-1).detach().cpu().numpy()
            data_d30 = torch.cat(overall_class_probs_d30[cls], dim=0).view(-1).detach().cpu().numpy()
            kde_d16 = gaussian_kde(data_d16)
            kde_d30 = gaussian_kde(data_d30)
            xmin, xmax = 0, 0.2
            x_vals = np.linspace(xmin, xmax, 1000)
            axs[cls // 5, cls % 5].plot(x_vals, kde_d16(x_vals), label='var_d16')
            axs[cls // 5, cls % 5].plot(x_vals, kde_d30(x_vals), label='var_d30')
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
        
        # Generate unified plots combining correct and wrong conditions
        logging.info(f"Generating unified token distance vs. probability plots comparing correct and wrong conditions...")
        for scale_idx in range(len(patch_nums)):
            try:
                # Correct condition data
                correct_distances_d16 = np.concatenate(overall_scale_distances_probs_d16[scale_idx]['distances']) if overall_scale_distances_probs_d16[scale_idx]['distances'] else np.array([])
                correct_probs_d16 = np.concatenate(overall_scale_distances_probs_d16[scale_idx]['probs']) if overall_scale_distances_probs_d16[scale_idx]['probs'] else np.array([])
                correct_distances_d30 = np.concatenate(overall_scale_distances_probs_d30[scale_idx]['distances']) if overall_scale_distances_probs_d30[scale_idx]['distances'] else np.array([])
                correct_probs_d30 = np.concatenate(overall_scale_distances_probs_d30[scale_idx]['probs']) if overall_scale_distances_probs_d30[scale_idx]['probs'] else np.array([])
                
                # Wrong condition data
                wrong_distances_d16 = np.concatenate(overall_scale_distances_probs_d16_wrong[scale_idx]['distances']) if overall_scale_distances_probs_d16_wrong[scale_idx]['distances'] else np.array([])
                wrong_probs_d16 = np.concatenate(overall_scale_distances_probs_d16_wrong[scale_idx]['probs']) if overall_scale_distances_probs_d16_wrong[scale_idx]['probs'] else np.array([])
                wrong_distances_d30 = np.concatenate(overall_scale_distances_probs_d30_wrong[scale_idx]['distances']) if overall_scale_distances_probs_d30_wrong[scale_idx]['distances'] else np.array([])
                wrong_probs_d30 = np.concatenate(overall_scale_distances_probs_d30_wrong[scale_idx]['probs']) if overall_scale_distances_probs_d30_wrong[scale_idx]['probs'] else np.array([])
                
                if (len(correct_distances_d16) == 0 or len(correct_distances_d30) == 0 or 
                    len(wrong_distances_d16) == 0 or len(wrong_distances_d30) == 0):
                    logging.warning(f"Insufficient data for unified plot at scale {scale_idx}")
                    continue
                
                # Create a random subsample for efficiency
                max_points = 500000
                for data, name in [(correct_distances_d16, "correct_d16"), (correct_probs_d16, "correct_probs_d16"),
                                   (correct_distances_d30, "correct_d30"), (correct_probs_d30, "correct_probs_d30"),
                                   (wrong_distances_d16, "wrong_d16"), (wrong_probs_d16, "wrong_probs_d16"),
                                   (wrong_distances_d30, "wrong_d30"), (wrong_probs_d30, "wrong_probs_d30")]:
                    if len(data) > max_points:
                        if name.startswith("correct_d16") or name.startswith("correct_probs_d16"):
                            indices = np.random.choice(len(correct_distances_d16), max_points, replace=False)
                            correct_distances_d16 = correct_distances_d16[indices]
                            correct_probs_d16 = correct_probs_d16[indices]
                        elif name.startswith("correct_d30") or name.startswith("correct_probs_d30"):
                            indices = np.random.choice(len(correct_distances_d30), max_points, replace=False)
                            correct_distances_d30 = correct_distances_d30[indices]
                            correct_probs_d30 = correct_probs_d30[indices]
                        elif name.startswith("wrong_d16") or name.startswith("wrong_probs_d16"):
                            indices = np.random.choice(len(wrong_distances_d16), max_points, replace=False)
                            wrong_distances_d16 = wrong_distances_d16[indices]
                            wrong_probs_d16 = wrong_probs_d16[indices]
                        elif name.startswith("wrong_d30") or name.startswith("wrong_probs_d30"):
                            indices = np.random.choice(len(wrong_distances_d30), max_points, replace=False)
                            wrong_distances_d30 = wrong_distances_d30[indices]
                            wrong_probs_d30 = wrong_probs_d30[indices]
                
                # Filter out extreme values
                correct_mask_d16 = (correct_probs_d16 > 1e-10) & (correct_distances_d16 < 50)
                correct_filtered_distances_d16 = correct_distances_d16[correct_mask_d16]
                correct_filtered_probs_d16 = correct_probs_d16[correct_mask_d16]
                
                correct_mask_d30 = (correct_probs_d30 > 1e-10) & (correct_distances_d30 < 50)
                correct_filtered_distances_d30 = correct_distances_d30[correct_mask_d30]
                correct_filtered_probs_d30 = correct_probs_d30[correct_mask_d30]
                
                wrong_mask_d16 = (wrong_probs_d16 > 1e-10) & (wrong_distances_d16 < 50)
                wrong_filtered_distances_d16 = wrong_distances_d16[wrong_mask_d16]
                wrong_filtered_probs_d16 = wrong_probs_d16[wrong_mask_d16]
                
                wrong_mask_d30 = (wrong_probs_d30 > 1e-10) & (wrong_distances_d30 < 50)
                wrong_filtered_distances_d30 = wrong_distances_d30[wrong_mask_d30]
                wrong_filtered_probs_d30 = wrong_probs_d30[wrong_mask_d30]
                
                # Create unified plot
                plt.figure(figsize=(15, 10))
                
                # Create distance bins
                max_correct_dist = max(np.max(correct_filtered_distances_d16), np.max(correct_filtered_distances_d30))
                max_wrong_dist = max(np.max(wrong_filtered_distances_d16), np.max(wrong_filtered_distances_d30))
                max_dist = min(max(max_correct_dist, max_wrong_dist), 30)
                bins = np.linspace(0, max_dist, 150)
                
                # Process each condition and model
                def process_data_for_plot(distances, probs, bins, color, linestyle, marker, label):
                    avg_probs = []
                    bin_centers = []
                    bin_counts = []
                    
                    for i in range(len(bins)-1):
                        mask = (distances >= bins[i]) & (distances < bins[i+1])
                        bin_count = np.sum(mask)
                        bin_counts.append(bin_count)
                        
                        if bin_count > 0:
                            avg_probs.append(np.mean(probs[mask]))
                        else:
                            avg_probs.append(np.nan)
                            
                        bin_centers.append((bins[i] + bins[i+1]) / 2)
                    
                    bin_centers = np.array(bin_centers)
                    avg_probs = np.array(avg_probs)
                    bin_counts = np.array(bin_counts)
                    
                    valid_indices = ~np.isnan(avg_probs)
                    
                    if np.sum(valid_indices) > 5:
                        # Plot scatter with varying alpha based on bin count
                        max_count = np.max(bin_counts[valid_indices]) if np.sum(valid_indices) > 0 else 1
                        alphas = np.minimum(0.4, 0.1 + 0.3 * bin_counts[valid_indices] / max_count)
                        
                        for i, idx in enumerate(np.where(valid_indices)[0]):
                            plt.scatter(bin_centers[idx], avg_probs[idx], 
                                       s=15, alpha=alphas[i], color=color, marker=marker, edgecolor='none')
                        
                        # Plot the curve
                        plt.plot(bin_centers[valid_indices], avg_probs[valid_indices], 
                                linestyle, color=color, linewidth=3, label=label)
                        
                        return bin_centers[valid_indices], avg_probs[valid_indices]
                    
                    return None, None
                
                # Process all four combinations
                correct_centers_d16, correct_values_d16 = process_data_for_plot(
                    correct_filtered_distances_d16, correct_filtered_probs_d16, 
                    bins, 'blue', '-', 'o', 'Correct Class - VAR D16'
                )
                
                correct_centers_d30, correct_values_d30 = process_data_for_plot(
                    correct_filtered_distances_d30, correct_filtered_probs_d30, 
                    bins, 'red', '-', 'o', 'Correct Class - VAR D30'
                )
                
                wrong_centers_d16, wrong_values_d16 = process_data_for_plot(
                    wrong_filtered_distances_d16, wrong_filtered_probs_d16, 
                    bins, 'blue', '--', 'x', 'Wrong Class - VAR D16'
                )
                
                wrong_centers_d30, wrong_values_d30 = process_data_for_plot(
                    wrong_filtered_distances_d30, wrong_filtered_probs_d30, 
                    bins, 'red', '--', 'x', 'Wrong Class - VAR D30'
                )
                
                plt.xlabel('Token Distance', fontsize=14)
                plt.ylabel('Average Probability', fontsize=14)
                plt.yscale('log')
                plt.title(f'Scale {scale_idx} (patches: {patch_nums[scale_idx]}x{patch_nums[scale_idx]})\nCorrect vs. Wrong Class Condition Comparison', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(osp.join(dist_analysis_folder, f"scale_{scale_idx}_comparison.png"), dpi=300)
                plt.close()
                
            except Exception as e:
                logging.error(f"Failed to create unified plot for scale {scale_idx}: {e}")
                continue
                
        logging.info(f"Unified token distance vs. probability plots saved to: {dist_analysis_folder}")

    metric_name = "Average L2 Distance" if args.mode == "l2_dist" else "Log Likelihood"
    logging.info(f"\nOverall Accuracies using {metric_name} for Classification:")
    logging.info(f"Overall Accuracy (d16): {100 * correct_d16 / total_d16:.2f}%")
    logging.info(f"Overall Accuracy (d30): {100 * correct_d30 / total_d30:.2f}%")
    logging.info(f"Overall Accuracy (original): {100 * correct / total:.2f}%")
    
    logging.info(f"\nPer-Scale Accuracies using {metric_name}:")
    for scale_idx in range(len(patch_nums)):
        logging.info(f"\nScale {scale_idx} (patch size: {patch_nums[scale_idx]}):")
        logging.info(f"  d16 Accuracy: {100 * scale_correct_d16[scale_idx] / scale_total_d16[scale_idx]:.2f}%")
        logging.info(f"  d30 Accuracy: {100 * scale_correct_d30[scale_idx] / scale_total_d30[scale_idx]:.2f}%")
    
    logging.info(f"\nAccumulated {metric_name} Accuracies (first scale_idx layers):")
    for scale_idx in range(len(patch_nums)):
        logging.info(f"\nFirst {scale_idx+1} layers:")
        logging.info(f"  d16 Accuracy: {100 * acc_correct_d16[scale_idx] / acc_total_d16[scale_idx]:.2f}%")
        logging.info(f"  d30 Accuracy: {100 * acc_correct_d30[scale_idx] / acc_total_d30[scale_idx]:.2f}%")
    
    logging.info(f"\nConditional {metric_name} Accuracies (excluding first scale_idx layers):")
    for scale_idx in range(len(patch_nums)):
        condition_desc = "all" if scale_idx == 0 else f"first {scale_idx}"
        logging.info(f"\nConditioned on {condition_desc} layers:")
        logging.info(f"  d16 Accuracy: {100 * cond_correct_d16[scale_idx] / cond_total_d16[scale_idx]:.2f}%")
        logging.info(f"  d30 Accuracy: {100 * cond_correct_d30[scale_idx] / cond_total_d30[scale_idx]:.2f}%")
    
    if args.mode == "l2_dist":
        logging.info("\nNote: For L2 distance classification, smaller distances indicate higher confidence in class prediction.")
        logging.info("      We negate the distances so larger values are better, compatible with argmax-based classification.")


if __name__ == "__main__":
    main()