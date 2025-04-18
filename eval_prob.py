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

MODEL_DEPTH = 16  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
LOG_DIR = "./output"


def smooth_log_probs_by_k(log_probs, k):
    """
    Smooth the probability distribution in groups of k tokens.
    
    Args:
        log_probs (torch.Tensor): Log probabilities of shape (B, L, V) where V is vocab size.
        k (int): Group size for smoothing. k=1 returns original distribution; k=V yields uniform.
    
    Returns:
        torch.Tensor: New log probabilities smoothed by grouping k tokens together.
    """
    B, L, V = log_probs.shape
    # Convert to probabilities.
    probs = torch.exp(log_probs)
    # Sort probabilities in descending order along the vocab dimension.
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Determine number of complete groups and remainder.
    num_complete_groups = V // k
    remainder = V % k
    
    # If there is a remainder, pad sorted_probs so that the last group has k elements.
    if remainder > 0:
        pad_size = k - remainder
        padding = torch.zeros(B, L, pad_size, device=log_probs.device, dtype=sorted_probs.dtype)
        sorted_probs_padded = torch.cat([sorted_probs, padding], dim=-1)
        # Create a mask for valid entries (ones for real tokens, zeros for padded).
        mask = torch.ones(B, L, V, device=log_probs.device, dtype=sorted_probs.dtype)
        pad_mask = torch.zeros(B, L, pad_size, device=log_probs.device, dtype=sorted_probs.dtype)
        mask = torch.cat([mask, pad_mask], dim=-1)
        
        # Reshape into groups of size k.
        groups = sorted_probs_padded.view(B, L, -1, k)  # shape (B, L, num_groups, k)
        mask_groups = mask.view(B, L, -1, k)
        # Compute group means using only valid tokens.
        group_sum = (groups * mask_groups).sum(dim=-1)
        group_count = mask_groups.sum(dim=-1)
        group_mean = group_sum / group_count
        # Expand group mean back to group shape.
        expanded_group_mean = group_mean.unsqueeze(-1).expand_as(groups)
        # Flatten back to original (padded) shape.
        new_sorted_probs = expanded_group_mean.reshape(B, L, -1)[:, :, :V]  # remove extra padding
    else:
        # If V is divisible by k, simply reshape and average.
        groups = sorted_probs.view(B, L, -1, k)  # shape (B, L, num_groups, k)
        group_mean = groups.mean(dim=-1)  # shape (B, L, num_groups)
        expanded_group_mean = group_mean.unsqueeze(-1).expand_as(groups)
        new_sorted_probs = expanded_group_mean.reshape(B, L, V)
    
    # Unsort: create a new tensor of the same shape and scatter the smoothed values back.
    new_probs = torch.empty_like(new_sorted_probs)
    new_probs.scatter_(dim=-1, index=sorted_idx, src=new_sorted_probs)
    
    # Convert back to log probabilities.
    new_log_probs = torch.log(new_probs + 1e-10)  # epsilon for numerical stability
    return new_log_probs


def create_heatmaps_for_classes(probs: torch.Tensor, patch_nums: list, input_img: torch.Tensor, alpha: float = 0.5):
    """
    Given a probability tensor of shape (10, L) (10 classes, L = sum(p^2) patches across 10 layers)
    and an input image tensor normalized to [-1, 1], create a heatmap overlay for each class.
    
    Args:
        probs (torch.Tensor): Tensor of shape (10, L), where each row corresponds to one class's patch probabilities.
        patch_nums (list): List of patch counts per side for each layer (length should be 10).
        input_img (torch.Tensor): Input image tensor of shape (3, 256, 256) normalized to [-1, 1].
        alpha (float): Blending factor for overlay (0 = only input image, 1 = only heatmap).
    
    Returns:
        List[np.ndarray]: A list of 10 overlaid images (one per class) as NumPy arrays.
    """
    patch_nums = patch_nums[:len(patch_nums)//2]
    num_classes = probs.shape[0]
    overlaid_images = []
    combined_heatmap_list = []

    # Compute total number of patches L = sum(p^2)
    total_patches = sum([p*p for p in patch_nums])
    
    # Convert the input image from tensor normalized in [-1,1] to numpy image in [0,255]
    # Assuming input_img shape is (3, H, W)
    img_np = input_img.clone().detach().cpu()  # clone to avoid modifying original tensor
    # Convert from [-1, 1] to [0, 1]
    img_np = (img_np + 1) / 2  
    # Convert to numpy and change channel order from (C, H, W) to (H, W, C)
    if input_img.dim() == 4:
        input_img = input_img.squeeze(0)
    img_np = (input_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    for class_idx in range(num_classes):
        # Get the probability vector for this class (shape: (L,))
        prob_vector = probs[class_idx]  
        layer_heatmaps = []
        start = 0
        
        # Process each layer individually.
        for p in patch_nums:
            num_patches = p * p  # Number of patches in this layer.
            # Slice out the probabilities for this layer.
            layer_probs = prob_vector[start:start + num_patches]
            start += num_patches
            
            # Reshape into a 2D grid (1, 1, p, p) for interpolation.
            patch_map = layer_probs.view(1, 1, p, p)
            # Upsample to 256x256 using bilinear interpolation.
            upsampled = torch.nn.functional.interpolate(patch_map, size=(256, 256), mode='bilinear', align_corners=False)
            upsampled = upsampled.squeeze()  # Now shape (256,256)
            layer_heatmaps.append(upsampled * (num_patches / total_patches))
        
        # Combine the per-layer heatmaps (here we take the average).
        combined_heatmap = torch.stack(layer_heatmaps, dim=0).sum(dim=0)
        combined_heatmap = combined_heatmap.cpu().numpy()
        combined_heatmap_list.append(combined_heatmap)
    
    combined_heatmap_list = np.stack(combined_heatmap_list)
    for combined_heatmap in combined_heatmap_list:
        # Normalize the combined heatmap to [0,1]
        combined_heatmap = combined_heatmap - combined_heatmap_list.min()
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / (combined_heatmap_list.max() - combined_heatmap_list.min())
        
        # Create a colored heatmap using a colormap (e.g., 'jet').
        cmap = plt.get_cmap('jet')
        # Get RGB (ignoring alpha), then convert to uint8 scale [0,255]
        colored_heatmap = (cmap(combined_heatmap)[..., :3] * 255).astype(np.uint8)
        
        # Blend the heatmap with the input image.
        # Here, blending is done via weighted addition.
        overlay = np.clip(img_np * (1 - alpha) + colored_heatmap * alpha, 0, 255).astype(np.uint8)
        overlaid_images.append(overlay)

    return overlaid_images


def setup_logging(run_folder):
    """Setup logging configuration"""
    log_file = osp.join(run_folder, "analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def download_checkpoints(hf_home, vae_ckpt, var_ckpt):
    """Download model checkpoints if they don't exist"""
    if not osp.exists(vae_ckpt):
        logging.info(f"Downloading VAE checkpoint from {hf_home}/{vae_ckpt}")
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_ckpt):
        logging.info(f"Downloading VAR checkpoint from {hf_home}/{var_ckpt}")
        os.system(f"wget {hf_home}/{var_ckpt}")

def setup_models(device, patch_nums, num_classes, depth):
    """Setup and load VAE and VAR models"""
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        num_classes=num_classes,
        depth=depth,
        shared_aln=False,
    )
    return vae, var

def load_checkpoints(vae, var, vae_ckpt, var_ckpt):
    """Load model checkpoints and set models to eval mode"""
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    logging.info("Models loaded and set to eval mode")
    var.cond_drop_rate = 0

def setup_seed(seed):
    """Setup random seed for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_tf32(tf32=True):
    """Setup TF32 precision"""
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=[
            "imagenet",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets/imagenet",
        help="Data path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Name of split",
    )
    parser.add_argument(
        "--extra", type=str, default=None, help="to add to the dataset name"
    )
    parser.add_argument("--partial", type=int, default=200)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=4)
    parser.add_argument("--Clayer", type=int, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--mode", type=str, default="bayesian")
    parser.add_argument("--feat", type=str, default="dinov2")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    MODEL_DEPTH = args.depth

    name = f"var"
    extra = args.extra if args.extra is not None else ""
    if args.depth != 16:
        name += f"_d{args.depth}"
    if args.mode != "bayesian":
        name += f"_mode[{args.mode}]"
    if args.feat != "dinov2":
        name += f"_feat[{args.feat}]"
    if args.Clayer:
        name += f"_Clayer[{args.Clayer}]"
    name += f"_cfg[{args.cfg}]"
    if "neighbor_bayesian" in args.mode:
        name += f"_threshold[{args.threshold}]"


    run_folder = (
        osp.join(LOG_DIR, args.dataset, name)
        if len(extra) == 0
        else osp.join(LOG_DIR, args.dataset, name + f"_{extra}")
    )
    os.makedirs(run_folder, exist_ok=True)
    log_file = setup_logging(run_folder)
    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Log file: {log_file}")

    # Build dataset
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path,
        final_reso=256,
        hflip=False,
    )
    ld_val = DataLoader(
        dataset_val,
        num_workers=0,
        pin_memory=True,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    del dataset_val

    # Download and setup models
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt, var_ckpt = "vae_ch160v4096z32.pth", f"var_d{args.depth}.pth"
    download_checkpoints(hf_home, vae_ckpt, var_ckpt)

    # Build and load models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    patch_nums_square_cumsum = np.cumsum(np.array(patch_nums)**2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, var = setup_models(device, patch_nums, num_classes, MODEL_DEPTH)
    load_checkpoints(vae, var, vae_ckpt, var_ckpt)

    # Setup seed and precision
    setup_seed(0)
    setup_tf32(True)

    # Initialize counters
    correct = 0
    total = 0
    pbar = tqdm.tqdm(ld_val)

    if args.mode == "gen":
        old_mean = 0.5
        old_std  = 0.5
        if args.feat == "resnet50":
            new_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
            new_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
            # Load pretrained ResNet50 and remove the final classification layer.
            resnet = models.resnet50(pretrained=True)
            # Remove the last fully-connected layer; use adaptive pooling output as features.
            feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            feature_extractor.eval()  # Set to eval mode
            feature_extractor.to(device)
        elif args.feat == "clip":
            new_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1).to(device)
            new_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1).to(device)
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            feature_extractor = clip_model.encode_image
        else:
            new_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
            new_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
            feature_extractor = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
            feature_extractor.to(device)

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = 0  # @param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = 250  # @param {type:"slider", min:0, max:1000, step:1}
    # cfg = 4  # @param {type:"slider", min:1, max:10, step:0.1}
    # class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
    more_smooth = False  # True for more smooth output

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    correct = 0
    total = 0
    pbar = tqdm.tqdm(ld_val)
    if args.mode == "fast_neighbor_bayesian":
        emb_weight = vae.quantize.embedding.weight  # (V, D)
        dists = torch.cdist(emb_weight, emb_weight, p=2)  # (V, V)
        neighbors = torch.argsort(dists, dim=1)  # (V, V) - tokens sorted by increasing distance.
        top_n_neighbors = neighbors[:, :]  # (V, n)
        
    for idx, (img, label) in enumerate(pbar):
        if args.partial is not None and idx >= args.partial:
            break
        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}%")
        # sample
        img = img.to(device)            
        if args.mode == "gen":
            img_input = img * (old_std / new_std) + ((old_mean - new_mean) / new_std)
            img_input = F.interpolate(img_input, size=(224, 224), mode='bicubic')
        remaining_classes = [i for i in range(num_classes)][:10]
        likelihood_list = []
        log_prob_list = []
        json_fname = osp.join(run_folder, f"{idx}.json")
        if os.path.exists(json_fname):
            # print("Skipping", i)
            with open(json_fname, "r") as f:
                # Reading from json file
                data = json.load(f)
            correct += int(data["pred"] == data["label"])
            total += 1
            continue
        with torch.inference_mode():
            # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            # Convert the image to its latent representation (list of token indices)
            gt_idx_list = vae.img_to_idxBl(img)  # List of tensors for each stage
            # Convert the image to its latent representation (list of token indices)
            gt_tokens = torch.cat(gt_idx_list, dim=1)
            while len(remaining_classes) > 0:
                class_labels = remaining_classes[: args.batch_size]
                remaining_classes = (
                    []
                    if len(remaining_classes) <= args.batch_size
                    else remaining_classes[args.batch_size :]
                )

                label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
                
                if args.mode == "bayesian":
                    # Prepare the teacher forcing input (excluding the first tokens)
                    # Here, we assume the same function is used as during training.
                    x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_list)

                    # Pass through the forward method to get logits for each token position.
                    # The forward method uses teacher forcing, meaning it conditions on the ground truth tokens.
                    assert var.cond_drop_rate == 0
                    logits = var.forward(
                        label_B, x_BLCv_wo_first_l
                    )  # (B, L, V) where V is vocab_size

                    # Compute log probabilities over the vocabulary.
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, L, V)

                    # Gather the log probabilities corresponding to the ground truth tokens.
                    # gt_tokens has shape (B, L), so we unsqueeze to (B, L, 1) for gathering.
                    gt_log_probs = log_probs.gather(
                        dim=-1, index=gt_tokens.unsqueeze(-1)
                    ).squeeze(-1)  # (B, L)
                    
                    log_prob_list.append(gt_log_probs)

                    if args.Clayer:
                        mask = torch.zeros_like(gt_tokens).to(device)
                        mask[:, patch_nums_square_cumsum[args.Clayer]:] = 1
                        mask = mask.bool()
                        log_likelihood = gt_log_probs[mask].sum().unsqueeze(0)
                    else:
                        # Sum the log probabilities along the sequence to get the overall log likelihood.
                        log_likelihood = gt_log_probs.sum(dim=1)  # (B,)

                    likelihood_list.append(log_likelihood)
                elif args.mode == "gen":
                    # Create a mask where a specific region is set to 0.
                    # Here we assume that the last 525 tokens (for example) should be inpainted.
                    # Adjust the mask as needed.
                    mask = torch.ones_like(gt_tokens).to(device)
                    if args.Clayer:
                        mask[:, patch_nums_square_cumsum[args.Clayer]:] = 0

                    # Run inpainting. The inpainting function is expected to take the original image,
                    # the latent tokens (gt_tokens), the mask, and other parameters.
                    inpainted_output = var.inpainting(
                        gt_tokens.repeat(args.batch_size, 1), mask.repeat(args.batch_size, 1),
                        cfg=args.cfg, top_k=1, top_p=0,
                        label=label_B, g_seed=seed
                    )

                    if args.feat == "vae_fhat":
                        # --- VAE fhat feature ---
                        fhat_input = vae.img_to_fhat(img)[-1]
                        fhat_inpaint = vae.img_to_fhat(inpainted_output)[-1]

                        feat_input = fhat_input.view(1, -1)
                        feat_inpaint = fhat_inpaint.view(args.batch_size, -1)

                    elif args.feat == "vae_post":
                        # --- VAE post_quant feature ---
                        fhat_input = vae.img_to_post(img)
                        fhat_inpaint = vae.img_to_post(inpainted_output)

                        feat_input = fhat_input.view(1, -1)
                        feat_inpaint = fhat_inpaint.view(args.batch_size, -1)
                    else:
                        inpainted_img = inpainted_output * (old_std / new_std) + ((old_mean - new_mean) / new_std)
                        inpainted_img = F.interpolate(inpainted_img, size=(224, 224), mode='bicubic')

                        # Extract features from the original image.
                        feat_input = feature_extractor(img_input)
                        feat_input = feat_input.view(1, -1)

                        # Extract features from the inpainted image.
                        feat_inpaint = feature_extractor(inpainted_img)
                        feat_inpaint = feat_inpaint.view(args.batch_size, -1)

                    # Compute the L1 distance between the features.
                    l1_distance = torch.abs(feat_input - feat_inpaint).mean(dim=-1)

                    # Convert the distance to likelihood: lower L1 distance implies higher likelihood.
                    # Here we use the negative L1 distance as the likelihood score.
                    likelihood = -l1_distance

                    likelihood_list.append(likelihood)

                elif args.mode == "smooth_bayesian":
                    # Prepare the teacher forcing input (excluding the first tokens)
                    # Here, we assume the same function is used as during training.
                    x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_list)

                    # Pass through the forward method to get logits for each token position.
                    # The forward method uses teacher forcing, meaning it conditions on the ground truth tokens.
                    logits = var.forward(
                        label_B, x_BLCv_wo_first_l
                    )  # (B, L, V) where V is vocab_size

                    # Compute log probabilities over the vocabulary.
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, L, V)

                    # Smooth the distribution by grouping every k tokens.
                    k = 50  # For example, change k to adjust the smoothing
                    log_probs = smooth_log_probs_by_k(log_probs, k)

                    # Gather the log probabilities corresponding to the ground truth tokens.
                    # gt_tokens has shape (B, L), so we unsqueeze to (B, L, 1) for gathering.
                    gt_log_probs = log_probs.gather(
                        dim=-1, index=gt_tokens.unsqueeze(-1)
                    ).squeeze(-1)  # (B, L)
                    
                    log_prob_list.append(gt_log_probs)

                    if args.Clayer:
                        mask = torch.zeros_like(gt_tokens).to(device)
                        mask[:, patch_nums_square_cumsum[args.Clayer]:] = 1
                        mask = mask.bool()
                        log_likelihood = gt_log_probs[mask].sum().unsqueeze(0)
                    else:
                        # Sum the log probabilities along the sequence to get the overall log likelihood.
                        log_likelihood = gt_log_probs.sum(dim=1)  # (B,)

                    likelihood_list.append(log_likelihood)

                elif args.mode == "neighbor_bayesian":
                    smoothed_output, log_likelihood, _ = var.smooth_sampling(gt_tokens, n=4096, cfg=args.cfg, label=class_labels[0], g_seed=seed, neighbor_threshold=args.threshold)
                    likelihood_list.append(log_likelihood.unsqueeze(0))
                
                elif args.mode == "fast_neighbor_bayesian":
                                        # Prepare the teacher forcing input (excluding the first tokens)
                    # Here, we assume the same function is used as during training.
                    x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_list)

                    # Pass through the forward method to get logits for each token position.
                    # The forward method uses teacher forcing, meaning it conditions on the ground truth tokens.
                    logits = var.forward(
                        label_B, x_BLCv_wo_first_l
                    )  # (B, L, V) where V is vocab_size

                    
                    candidate_neighbors_full = top_n_neighbors[gt_tokens]  # (B, cur_L_segment, n)
                    candidate_dists = torch.gather(
                        dists[gt_tokens], dim=-1, index=candidate_neighbors_full
                    )
                    effective_threshold = args.threshold
                    # Build mask: valid candidates are those with distance below effective_threshold.
                    candidate_mask = candidate_dists <= effective_threshold
                    candidate_log_probs = torch.gather(log_probs, dim=-1, index=candidate_neighbors_full)
                    candidate_log_probs = candidate_log_probs.masked_fill(~candidate_mask, float("-inf"))
                    max_vals, max_idx = candidate_log_probs.max(dim=-1)  # (B, cur_L_segment)
                    
                    log_likelihood = max_vals.sum(dim=1)  # (B,)

                    likelihood_list.append(log_likelihood)

        if args.plot:
            log_prob_list = torch.cat(log_prob_list, dim=0)
            overlays = create_heatmaps_for_classes(log_prob_list, patch_nums, img, alpha=0.5)
            # Display the overlaid images for each class.
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            axs = axs.flatten()
            for i, overlay in enumerate(overlays):
                axs[i].imshow(overlay)
                axs[i].axis('off')
                axs[i].set_title(f"Class {i}")
            plt.tight_layout()
            plt.savefig(osp.join(run_folder, f"{idx}.png"))
            plt.close()

        likelihood_list = torch.cat(likelihood_list, dim=0)
        pred = torch.argmax(likelihood_list)
        data = {"pred": pred.item(), "label": label.item()}
        with open(json_fname, "w") as f:
            json.dump(data, f)
        if pred.item() == label.item():
            correct += 1
        total += 1

    logging.info(f"Final accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
