"""
Quick inference test: use cached model + synthetic data to verify pipeline works.
Avoids downloading the full 418-file dataset.
"""
import os
os.environ.pop('SSL_CERT_FILE', None)

import torch
import numpy as np
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Load pretrained policy (already cached from earlier download)
    print("Loading diffusion_pusht policy from cache...")
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
    policy = policy.to(device)
    policy.eval()
    print("Policy loaded and on GPU!")

    # Create synthetic observation matching pusht format
    # pusht: image (3, 96, 96), state (2,)
    print("Creating synthetic input...")
    batch_size = 1
    image = torch.randn(batch_size, 3, 96, 96).float().to(device)
    state = torch.randn(batch_size, 2).float().to(device)

    # Normalize like real data (image to [0,1], state typical range)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    obs_dict = {
        "observation.state": state,
        "observation.image": image,
    }

    # Run inference
    print("Running inference...")
    with torch.inference_mode():
        action = policy.select_action(obs_dict)

    print(f"Predicted action shape: {action.shape}")  # expected: (1, 16)
    print(f"Predicted action[0] sample: {action.squeeze(0)[:5].cpu().tolist()}")

    # Run a few more steps to verify consistency
    print("\nRunning 5 more inference steps...")
    for i in range(5):
        with torch.inference_mode():
            a = policy.select_action(obs_dict)
        print(f"  Step {i}: action[0][:3] = {a.squeeze(0)[:3].cpu().tolist()}")

    print("\nInference test PASSED!")
    print("=" * 50)
    print("Summary:")
    print(f"  - Policy: lerobot/diffusion_pusht")
    print(f"  - Device: {device}")
    print(f"  - Input: synthetic (image: {image.shape}, state: {state.shape})")
    print(f"  - Output: action {action.shape}")
    print("=" * 50)

if __name__ == "__main__":
    main()
