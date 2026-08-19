"""
Ray Tune Hyperparameter Search
--------------------------------
Uses ASHA early stopping + Optuna Bayesian search to find the best
hyperparameters for a given training mode.

Usage:
    python raytune_sweep.py                        # fusion (default), 50 trials
    python raytune_sweep.py --mode clinical        # MLP-only ablation
    python raytune_sweep.py --mode vision          # CNN-only ablation
    python raytune_sweep.py --trials 20 --epochs 15  # quick smoke test
    python raytune_sweep.py --cpus 4 --gpus 1

Search spaces by mode:
    fusion   clin_hidden_dims, clin_feature_dim, dropout, unfreeze_epoch,
             fusion_hidden_dims, learning_rate, weight_decay, batch_size
    clinical clin_hidden_dims, clin_feature_dim, dropout,
             fusion_hidden_dims, learning_rate, weight_decay, batch_size
    vision   unfreeze_epoch, dropout, fusion_hidden_dims,
             learning_rate, weight_decay, batch_size
"""

import argparse
import functools

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import config
from train import train


# ── Per-mode search spaces ────────────────────────────────────────────────────

# Shared optimizer params used by every mode
_OPTIMIZER_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "weight_decay":  tune.loguniform(1e-5, 1e-2),
    "batch_size":    tune.choice([16, 32, 64]),
}

# MLP branch — only relevant when clinical branch is active
_MLP_SPACE = {
    "clin_hidden_dims": tune.choice([
        [64, 128, 64],
        [128, 256, 128],
        [256, 512, 256],
        [128, 128],
        [256, 256],
    ]),
    "clin_feature_dim": tune.choice([64, 128, 256]),
}

# CNN branch — only relevant when vision branch is active
_CNN_SPACE = {
    "unfreeze_epoch": tune.choice([5, 10, 15, 20]),
}

# Trunk — the dense layers between branch output(s) and the classifier head.
# Present in all three modes, so tuned independently for each.
_TRUNK_SPACE = {
    "dropout": tune.choice([0.2, 0.3, 0.4, 0.5]),
    "fusion_hidden_dims": tune.choice([
        [256, 128],
        [512, 256],
        [512, 256, 128],
        [1024, 512, 256],
    ]),
}

SEARCH_SPACES = {
    "fusion":   {**_MLP_SPACE, **_CNN_SPACE, **_TRUNK_SPACE, **_OPTIMIZER_SPACE},
    "clinical": {**_MLP_SPACE,               **_TRUNK_SPACE, **_OPTIMIZER_SPACE},
    "vision":   {              **_CNN_SPACE, **_TRUNK_SPACE, **_OPTIMIZER_SPACE},
}


# ── Trainable ─────────────────────────────────────────────────────────────────

def _make_trainable(mode: str):
    """Return a trainable closure bound to the given mode."""

    def trainable(ray_cfg: dict):
        # MLP params (fusion + clinical only)
        if mode in ("fusion", "clinical"):
            config.CLIN_HIDDEN_DIMS  = list(ray_cfg["clin_hidden_dims"])
            config.CLIN_FEATURE_DIM  = ray_cfg["clin_feature_dim"]

        # CNN params (fusion + vision only)
        if mode in ("fusion", "vision"):
            config.UNFREEZE_EPOCH = ray_cfg["unfreeze_epoch"]

        # Shared params for all modes
        config.DROPOUT            = ray_cfg["dropout"]
        config.FUSION_HIDDEN_DIMS = list(ray_cfg["fusion_hidden_dims"])
        config.LEARNING_RATE      = ray_cfg["learning_rate"]
        config.WEIGHT_DECAY       = ray_cfg["weight_decay"]
        config.BATCH_SIZE         = ray_cfg["batch_size"]

        def _report(epoch, val_loss, val_acc, val_f1):
            tune.report(val_loss=val_loss, val_acc=val_acc, val_f1=val_f1)

        train(mode=mode, use_wandb=False, epoch_callback=_report)

    return trainable


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["fusion", "clinical", "vision"], default="fusion",
        help="Which branch(es) to train and tune (default: fusion)",
    )
    parser.add_argument("--trials", type=int,   default=50,  help="Number of trials")
    parser.add_argument("--cpus",   type=float, default=4.0, help="CPUs per trial")
    parser.add_argument("--gpus",   type=float, default=1.0, help="GPUs per trial")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override NUM_EPOCHS for faster search (e.g. --epochs 15)",
    )
    args = parser.parse_args()

    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=config.NUM_EPOCHS,
        grace_period=max(5, config.NUM_EPOCHS // 6),
        reduction_factor=3,
    )

    searcher = OptunaSearch(metric="val_loss", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(_make_trainable(args.mode), {"cpu": args.cpus, "gpu": args.gpus}),
        param_space=SEARCH_SPACES[args.mode],
        tune_config=tune.TuneConfig(
            search_alg=searcher,
            scheduler=scheduler,
            num_samples=args.trials,
        ),
        run_config=ray.air.RunConfig(
            name=f"crop_disease_{args.mode}",
            local_dir="./ray_results",
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_loss", mode="min")
    print("\n" + "=" * 55)
    print(f"BEST TRIAL  (mode={args.mode})")
    print("=" * 55)
    print(f"  val_loss : {best.metrics['val_loss']:.4f}")
    print(f"  val_acc  : {best.metrics['val_acc']*100:.1f}%")
    print(f"  val_f1   : {best.metrics['val_f1']:.3f}")
    print("\n  Hyperparameters:")
    for k, v in best.config.items():
        print(f"    {k:22s}: {v}")


if __name__ == "__main__":
    main()
