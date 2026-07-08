"""
Test LeRobot inference without gym_pusht (avoids DLL conflict on Windows).
Loads a sample from the lerobot dataset and runs it through the diffusion_pusht policy.
"""
import os
os.environ.pop('SSL_CERT_FILE', None)

import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load a sample from the pusht dataset
    print("Loading pusht dataset sample...")
    dataset = LeRobotDataset("lerobot/pusht")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    # Get observation
    image = sample["observation.image"]          # (3, 96, 96)
    state = sample["observation.state"]          # (2,)
    action = sample["action"]                    # (16,)

    print(f"Image shape: {image.shape}")
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")

    # Load pretrained policy
    print("Loading diffusion_pusht policy...")
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
    policy = policy.to(device)
    policy.eval()
    print("Policy loaded!")

    # Prepare observation for policy
    image_tensor = image.float().unsqueeze(0).to(device) / 255.0
    state_tensor = state.float().unsqueeze(0).to(device)

    obs_dict = {
        "observation.state": state_tensor,
        "observation.image": image_tensor,
    }

    # Run inference
    print("Running inference...")
    with torch.inference_mode():
        predicted_action = policy.select_action(obs_dict)

    print(f"Predicted action shape: {predicted_action.shape}")  # (1, 16)
    print(f"Ground truth action[0] shape: {action.shape}")       # (16,)
    print(f"Ground truth action[0]: {action[:5].tolist()}...")
    print(f"Predicted action[0]: {predicted_action.squeeze(0)[:5].cpu().tolist()}...")

    # Compute simple MSE between predicted and first frame of GT action
    mse = torch.nn.functional.mse_loss(
        predicted_action.squeeze(0).cpu().float(),
        action.float()
    ).item()
    print(f"\nMSE between predicted and GT action: {mse:.4f}")
    print("\nInference test PASSED!")

if __name__ == "__main__":
    main()
