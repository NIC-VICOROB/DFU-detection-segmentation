"""
Fine-tune SAM's image encoder with LoRA, prompted with detection bounding boxes.

This script depends on the LoRA-adapted SAM implementation from
https://github.com/MathieuNlp/Sam_LoRA (see sam/README.md for setup instructions).
That dependency is not vendored in this repository, the same way nnUNet/nnDetection
are used as external Docker-based tools elsewhere in this repo.

Usage (from the repository root):
    python -m sam.train_lora --config sam/lora_config.yaml
"""
import argparse
import os
import sys
from statistics import mean

import torch
import yaml
from torch.optim import Adam
from tqdm import tqdm

# Make the external Sam_LoRA dependency importable. Override with the
# SAM_LORA_REPO environment variable if it is cloned somewhere else.
THIRD_PARTY_DIR = os.environ.get(
    "SAM_LORA_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "Sam_LoRA"),
)
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

try:
    import monai
    from src.lora import LoRA_sam
    from src.processor import Samprocessor
    from src.segment_anything import build_sam_vit_b
except ImportError as e:
    raise ImportError(
        f"Could not import the Sam_LoRA dependency from '{THIRD_PARTY_DIR}'. "
        "Clone https://github.com/MathieuNlp/Sam_LoRA there (or set SAM_LORA_REPO) "
        "and install its requirements (including monai). See sam/README.md."
    ) from e

from sam.src import utils as sam_utils
from src.utils import get_least_used_gpu
from sam.src.dataloader import DatasetSegmentation, collate_fn
from torch.utils.data import DataLoader


def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    gpu_index = get_least_used_gpu()
    device = torch.device(f"cuda:{gpu_index}" if gpu_index >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sam = build_sam_vit_b(checkpoint=config["SAM"]["CHECKPOINT"])
    sam_lora = LoRA_sam(sam, config["SAM"]["RANK"])
    model = sam_lora.sam

    processor = Samprocessor(model)
    train_ds = DatasetSegmentation(config, processor, mode="train")
    train_dataloader = DataLoader(
        train_ds, batch_size=config["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn
    )

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["TRAIN"].get("LR", 1e-4)))
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    model.train()
    model.to(device)

    output_dir = config["TRAIN"].get("OUTPUT_DIR", "weights/sam_lora")
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = config["TRAIN"]["NUM_EPOCHS"]
    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            outputs = model(batched_input=batch, multimask_output=False)

            stk_gt, stk_out = sam_utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)  # [B, C, H, W] from [H, W]
            loss = seg_loss(stk_out, stk_gt.float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs} | mean training loss: {mean(epoch_losses):.4f}")

    rank = config["SAM"]["RANK"]
    weights_path = os.path.join(output_dir, f"lora_rank{rank}.safetensors")
    sam_lora.save_lora_parameters(weights_path)
    print(f"Training complete. LoRA weights saved to {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sam/lora_config.yaml")
    args = parser.parse_args()

    train(config_path=args.config)
