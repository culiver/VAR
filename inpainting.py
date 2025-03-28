################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

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
from torchvision.transforms import InterpolationMode, transforms

MODEL_DEPTH = 30  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
LOG_DIR = "./output"


# Denormalization function
def denormalize(tensor, mean=[0.5], std=[0.5]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean  # Reverse normalization


# Convert the tensor back to a valid image format
def save_tensor_image(tensor, filename="output_image.png"):
    assert len(tensor) == 1
    tensor = tensor[0].detach().cpu()
    tensor = denormalize(tensor)  # Undo normalization
    tensor = torch.clamp(tensor, 0, 1)  # Ensure valid pixel range
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(tensor)
    pil_image.save(filename)


def img_idx_to_row_idx(idx_list):
    patch_nums_array = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    patch_cumsums = np.cumsum(patch_nums_array**2)
    result_list = []
    for (lidx, hidx, widx) in idx_list:
        assert lidx > 0
        patch_num = patch_nums_array[lidx]
        idx = patch_cumsums[lidx-1] + hidx * patch_num + widx
        result_list.append(idx)
    return result_list


def generate_inpainting_mask(patch_nums, target_layer, patch_coord_list):
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
        layer_mask = torch.ones(tokens_in_layer, dtype=torch.bool)
        
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
                layer_mask[idx] = False
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
                        layer_mask[idx] = False
        
        mask_list.append(layer_mask)
    
    # Concatenate all layer masks into one flat mask.
    full_mask = torch.cat(mask_list)
    return full_mask


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
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args = parser.parse_args()

    args.extra = "inpainting"

    name = f"var"
    extra = args.extra if args.extra is not None else ""

    run_folder = (
        osp.join(LOG_DIR, args.dataset, name)
        if len(extra) == 0
        else osp.join(LOG_DIR, args.dataset, name + f"_{extra}")
    )
    os.makedirs(run_folder, exist_ok=True)
    print(f"Run folder: {run_folder}")

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

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt, var_ckpt = "vae_ch160v4096z32.pth", f"var_d{MODEL_DEPTH}.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_ckpt):
        os.system(f"wget {hf_home}/{var_ckpt}")

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "vae" not in globals() or "var" not in globals():
        vae, var = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            device=device,
            patch_nums=patch_nums,
            num_classes=1000,
            depth=MODEL_DEPTH,
            shared_aln=False,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    print(f"prepare finished.")

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = 0  # @param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = 250  # @param {type:"slider", min:0, max:1000, step:1}
    cfg = 4  # @param {type:"slider", min:1, max:10, step:0.1}
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
    for idx, (img, label) in enumerate(pbar):
        if args.partial is not None and idx >= args.partial:
            break
        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}%")
        # sample
        img = img.to(device)
        save_tensor_image(
            img, os.path.join(run_folder, f"{idx}.png"),
        )
        remaining_classes = [i for i in range(num_classes)][:10]
        likelihood_list = []
        log_prob_list = []
        json_fname = osp.join(run_folder, f"{idx}.json")
        with torch.inference_mode():
            # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            while len(remaining_classes) > 0:
                class_labels = remaining_classes[: args.batch_size]
                remaining_classes = (
                    []
                    if len(remaining_classes) <= args.batch_size
                    else remaining_classes[args.batch_size :]
                )

                # label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

                # Convert the image to its latent representation (list of token indices)
                gt_idx_list = vae.img_to_idxBl(img)  # List of tensors for each stage
                # Convert the image to its latent representation (list of token indices)
                gt_tokens = torch.cat(gt_idx_list, dim=1)

                # Suppose we want to mask a patch at the sixth layer (index 5)
                # For example, choose the patch at row 2, column 3 in the 6x6 grid.
                target_layer = 7
                patch_coord_list = [(4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5), (6, 6), (6, 7), (7, 4), (7, 5), (7, 6), (7, 7)]
                
                # Generate the inpainting mask.
                mask = generate_inpainting_mask(patch_nums, target_layer, patch_coord_list).to(device).unsqueeze(0)

                # Run inpainting.
                inpainted_output = var.inpainting(img, gt_tokens, mask, cfg=cfg, top_k=900, top_p=0.95, label=class_labels[0], g_seed=seed)
                # inpainted_output = var.autoregressive_infer_cfg(B=1, label_B=class_labels[0], cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

                # Convert the output tensor to a PIL image and save.
                # Assuming output image shape is (B, 3, H, W) with values in [0, 1].
                inpainted_img_np = inpainted_output[0].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                output_pil = PImage.fromarray(inpainted_img_np)
                output_pil.save(os.path.join(run_folder, f"{idx}_inpainted_{class_labels[0]}.png"))
                
                print("Inpainting complete. The image has been saved as 'inpainted_demo.png'.")
        if idx >= 10:
            break


if __name__ == "__main__":
    main()
