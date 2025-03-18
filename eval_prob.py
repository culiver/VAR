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

MODEL_DEPTH = 16  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
LOG_DIR = "./output"


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
        remaining_classes = [i for i in range(num_classes)]
        likelihood_list = []
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

                label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

                # Convert the image to its latent representation (list of token indices)
                gt_idx_list = vae.img_to_idxBl(img)  # List of tensors for each stage
                # Convert the image to its latent representation (list of token indices)
                gt_tokens = torch.cat(gt_idx_list, dim=1)

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

                # Gather the log probabilities corresponding to the ground truth tokens.
                # gt_tokens has shape (B, L), so we unsqueeze to (B, L, 1) for gathering.
                gt_log_probs = log_probs.gather(
                    dim=-1, index=gt_tokens.unsqueeze(-1)
                ).squeeze(-1)  # (B, L)

                # Sum the log probabilities along the sequence to get the overall log likelihood.
                log_likelihood = gt_log_probs.sum(dim=1)  # (B,)

                likelihood_list.append(log_likelihood)
        likelihood_list = torch.cat(likelihood_list, dim=0)

        pred = torch.argmax(likelihood_list)
        data = {"pred": pred.item(), "label": label.item()}
        with open(json_fname, "w") as f:
            json.dump(data, f)
        if pred.item() == label.item():
            correct += 1
        total += 1


if __name__ == "__main__":
    main()
