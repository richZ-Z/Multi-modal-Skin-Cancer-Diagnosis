"""
W&B Bayesian Hyperparameter Sweep
-----------------------------------
Uses Bayesian optimisation to find the best combination of hyperparameters,
guided by val/mae (lower is better).

Usage:
    # Option A — let this script create the sweep AND run 20 trials automatically:
    python sweep.py

    # Option B — create the sweep, then launch agents manually (good for parallelism):
    python sweep.py --create-only
    wandb agent <entity>/<project>/<sweep_id>   # run in as many terminals as you want

What gets searched:
    learning_rate     log-uniform in [1e-5, 1e-3]
    weight_decay      log-uniform in [1e-5, 1e-2]
    dropout           {0.2, 0.3, 0.4, 0.5}
    batch_size        {16, 32, 64}
    unfreeze_epoch    {5, 10, 15}
    clin_hidden_dims  three MLP depth/width options
    fusion_hidden     three fusion block options
"""

import argparse
import config
from train import train

try:
    import wandb
except ImportError:
    raise SystemExit("wandb is required for sweeps. Run: pip install wandb")


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/mae", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "dropout": {
            "values": [0.2, 0.3, 0.4, 0.5],
        },
        "batch_size": {
            "values": [16, 32, 64],
        },
        "unfreeze_epoch": {
            "values": [5, 10, 15],
        },
        "clin_hidden_dims": {
            "values": [
                [64, 128, 64],
                [128, 256, 128],
                [256, 512, 256],
            ],
        },
        "fusion_hidden": {
            "values": [
                [256, 128],
                [512, 256],
                [512, 256, 128],
            ],
        },
    },
}


def sweep_train():
    """
    Called once per trial by the W&B agent.
    Reads hyperparameters from wandb.config and overrides config.py values,
    then delegates to the standard train() function.
    """
    with wandb.init():
        cfg = wandb.config

        # Override config values with the trial's hyperparameters
        config.LEARNING_RATE      = cfg.learning_rate
        config.WEIGHT_DECAY       = cfg.weight_decay
        config.DROPOUT            = cfg.dropout
        config.BATCH_SIZE         = cfg.batch_size
        config.UNFREEZE_EPOCH     = cfg.unfreeze_epoch
        config.CLIN_HIDDEN_DIMS   = list(cfg.clin_hidden_dims)
        config.FUSION_HIDDEN_DIMS = list(cfg.fusion_hidden)

        # use_wandb=True so train() logs to the already-active run;
        # run_is_mine stays False so train() won't call wandb.finish()
        train(mode="fusion", use_wandb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create the sweep but don't start an agent (print the sweep ID and exit).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep trials to run (default: 20).",
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
    )

    entity  = config.WANDB_ENTITY or wandb.api.default_entity
    print(f"\nSweep created: {sweep_id}")
    print(f"Dashboard    : https://wandb.ai/{entity}/{config.WANDB_PROJECT}/sweeps/{sweep_id}")
    print(f"\nTo add more agents (parallel): wandb agent {entity}/{config.WANDB_PROJECT}/{sweep_id}")

    if not args.create_only:
        print(f"\nStarting agent — running {args.count} trials...")
        wandb.agent(sweep_id, function=sweep_train, count=args.count)
