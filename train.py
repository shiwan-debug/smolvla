"""
SmolVLA Training Entry Point

Usage:
    python train.py                          # Use train_config.json
    python train.py --config my_config.json  # Use custom config
"""
import argparse
import json
import os
import sys

# Local dev: add src to path. In Docker, lerobot is installed via pip.
_src = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Training")
    parser.add_argument("--config", type=str, default="train_config.json",
                        help="Path to training config JSON")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path")
    parser.add_argument("--model", type=str, default=None,
                        help="Override pretrained model path")
    parser.add_argument("--output", type=str, default="/smolvla/outputs",
                        help="Output directory for checkpoints")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Overrides
    if args.dataset:
        config["dataset"]["root"] = args.dataset
    if args.model:
        config["policy"]["pretrained_path"] = args.model
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"SmolVLA Training: {config['job_name']}")
    print("=" * 60)
    print(f"  Model:            {config['policy']['pretrained_path']}")
    print(f"  Dataset:          {config['dataset']['root']}/{config['dataset']['repo_id']}")
    print(f"  Batch size:       {config['batch_size']}")
    print(f"  Steps:            {config['steps']}")
    print(f"  Learning rate:    {config['optimizer']['lr']}")
    print(f"  Output dir:       {args.output}")
    print("=" * 60)

    # Import lerobot training
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train

    # Build TrainPipelineConfig from JSON
    train_config = TrainPipelineConfig.from_dict(config)
    train_config.policy.pretrained_path = config["policy"]["pretrained_path"]

    print("\nStarting training...")
    train(train_config)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
