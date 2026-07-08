"""
SmolVLA inference test — loads local model and runs inference with synthetic data.

Usage (from smolvla_new env):
    python test_smolvla.py
"""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'   # 离线模式，不连网
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, 'D:/FTP-1/ftp1-policy-main/smolvla/src')

import torch
from lerobot.policies import factory
from lerobot.configs import PreTrainedConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "D:/FTP-1/ftp1-policy-main/smolvla/models/smolvla_base"

    print("=" * 60)
    print("SmolVLA Inference Test")
    print("=" * 60)

    # 1. Load model
    print("\n[1/3] Loading model...")
    config = PreTrainedConfig.from_pretrained(model_path)
    print(f"  Type: {config.type}")
    print(f"  VLM base: {config.vlm_model_name}")
    print(f"  Input: state(6) + 3× camera(3,256,256)")
    print(f"  Output: action(6), chunk={config.n_action_steps} steps")

    policy_cls = factory.get_policy_class(config.type)
    policy = policy_cls.from_pretrained(model_path).to(device).eval()
    policy.reset()
    print(f"  Model loaded on {device}!")

    # 2. Build input
    print("\n[2/3] Building input...")
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    prompt = "pick up the red cube and place it on the table"
    tokens = tokenizer(prompt, return_tensors='pt', padding='max_length',
                       max_length=config.tokenizer_max_length, truncation=True)

    obs = {
        "observation.state": torch.randn(1, 1, 6).to(device),
        "observation.images.camera1": torch.randn(1, 1, 3, 256, 256).to(device),
        "observation.images.camera2": torch.randn(1, 1, 3, 256, 256).to(device),
        "observation.images.camera3": torch.randn(1, 1, 3, 256, 256).to(device),
        "observation.language.tokens": tokens.input_ids.to(device),
        "observation.language.attention_mask": tokens.attention_mask.to(device),
        "prompt": prompt,
    }
    print(f"  Prompt: '{prompt}'")

    # 3. Run inference
    print("\n[3/3] Running inference...")
    with torch.inference_mode():
        action = policy.select_action(obs)

    print(f"\n{'=' * 60}")
    print(f"Predicted action shape: {action.shape}")
    print(f"Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"Action sample: {action[0].cpu().tolist()}")
    print(f"{'=' * 60}")
    print("\nSmolVLA inference test PASSED!")

if __name__ == "__main__":
    main()
