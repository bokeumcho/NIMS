import os
import json
import argparse

def build_parser():
    p = argparse.ArgumentParser(
        description="Train SimVP variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hardware / run
    p.add_argument("--gpu", type=int, default=1, help="GPU index (0-3)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--workdir", type=str, default=".", help="Logging/checkpoint dir")

    # Data / temporal
    p.add_argument("--C", type=int, default=4, help="Input/Output channels per frame")
    p.add_argument("--C_enc", type=int, default=16, help="Encoder/latent channels")
    p.add_argument("--C_hid", type=int, default=64, help="Translator hidden channels")
    p.add_argument("--T", type=int, default=6, help="Context length (frames)")
    p.add_argument("--T_interval", type=int, default=10, help="Minutes between frames (must divide data cadence)")
    p.add_argument("--steps", type=int, default=5, help="Number of T-sized blocks to predict (horizon = steps*T)")

    # Model
    p.add_argument("--model_name", type=str, default="SimVP_AR_Decoder",
                   choices=["SimVP_AR_Decoder", "SimVP_kT"],
                   help="Which model wrapper to use")
    p.add_argument("--Ns", type=int, default=4, help="Num spatial stages (encoder/decoder)")
    p.add_argument("--Nt", type=int, default=8, help="Num translator stages")
    p.add_argument("--groups", type=int, default=4, help="Inception/Grouped conv groups")
    # p.add_argument("--incep_kernel_sizes", type=int, nargs="+", default=[3,5,7,11],
    #                help="Inception kernel sizes")

    # Optimization
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0, help="0 disables clipping")

    # Scheduled sampling (if you use it)
    p.add_argument("--tf_mode", action="store_true", help="Enable teacher forcing during training")
    p.add_argument("--ss_start", type=float, default=1.0, help="Initial scheduled sampling ratio")
    p.add_argument("--ss_end", type=float, default=0.0, help="Final scheduled sampling ratio")
    p.add_argument("--ss_schedule", type=str, default="linear", choices=["linear","exp","inv_sigmoid"],
                   help="Decay schedule for ss_ratio")

    # Misc
    p.add_argument("--save_config", action="store_true", help="Save parsed args to JSON in workdir")
    return p

def validate_args(args):
    # Basic checks you likely want
    if not (0 <= args.gpu <= 7):  # adjust if you have >4 GPUs
        raise ValueError(f"--gpu must be a valid device index, got {args.gpu}")
    if args.T <= 0 or args.steps <= 0:
        raise ValueError("T and steps must be positive")
    if args.T_interval <= 0:
        raise ValueError("T_interval must be positive")
    # If your raw cadence is 10 minutes, keep it aligned:
    # if args.T_interval % 10 != 0:
    #     raise ValueError("T_interval must be a multiple of 10 for your dataset cadence")
    return args

def setup_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"
    return device

def save_config(args):
    os.makedirs(args.workdir, exist_ok=True)
    path = os.path.join(args.workdir, "config.json")
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[info] Saved config to {path}")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = validate_args(args)
    device = setup_device(args)
    print(f"[info] Using device: {device}")
    if args.save_config:
        save_config(args)
