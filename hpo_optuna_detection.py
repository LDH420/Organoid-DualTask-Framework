#!/usr/bin/env python3
"""
HPO for Detection Models: DiffusionDet, RT-DETR, DeepOrga, Faster R-CNN, DINO, EfficientNet, RTMDet, DDQ, Tellu, YOLOv13
"""

import os, argparse, optuna, torch, numpy as np, random
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import your actual detection models here
# from your_detection_lib import YOLO, RTDETR, DiffusionDet, DINO, FasterRCNN, RTMDet, DDQ, DeepOrga, EfficientDet, Tellu

ADAPTERS = {
    "yolov8": None,  # Replace with actual adapter
    "yolov13": None,
    "rt-detr": None,
    "rtdetr": None,
    "dino": None,
    "dino-4scale": None,
    "diffusiondet": None,
    "faster-rcnn": None,
    "fasterrcnn": None,
    "rtmdet": None,
    "ddq": None,
    "deeporga": None,
    "efficientnet": None,
    "efficientdet": None,
    "tellu": None,
}

def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def space_common(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "bs": trial.suggest_categorical("bs", [8, 16, 32]),
        "wd": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "mosaic": trial.suggest_categorical("mosaic", [0.0, 0.5, 1.0]),
        "mixup": trial.suggest_float("mixup", 0.0, 0.5),
        "num_workers": trial.suggest_categorical("num_workers", [2, 4]),
        "use_amp": trial.suggest_categorical("use_amp", [True, False]),
    }

def space_yolo(trial):
    base = space_common(trial)
    base.update({
        "model_size": trial.suggest_categorical("model_size", ["n", "s", "m", "l"]),
        "lrf": trial.suggest_float("lrf", 0.001, 0.1, log=True),
        "momentum": trial.suggest_float("momentum", 0.85, 0.98),
        "box": trial.suggest_float("box", 5.0, 10.0),
        "cls": trial.suggest_float("cls", 0.3, 1.0),
        "dfl": trial.suggest_float("dfl", 1.0, 3.0),
    })
    return base

def space_detr(trial):
    base = space_common(trial)
    base.update({
        "backbone_lr_scale": trial.suggest_float("backbone_lr_scale", 0.01, 0.5),
        "num_queries": trial.suggest_categorical("num_queries", [300, 500, 900]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    })
    return base

def space_diffusion(trial):
    base = space_common(trial)
    base.update({
        "num_proposals": trial.suggest_categorical("num_proposals", [300, 500, 1000]),
        "num_steps": trial.suggest_int("num_steps", 1, 5),
        "sample_method": trial.suggest_categorical("sample_method", ["ddim", "pndm"]),
    })
    return base

def get_search_space(model_name, trial):
    name = model_name.lower()
    if "yolo" in name: return space_yolo(trial)
    if any(x in name for x in ["detr", "dino", "ddq"]): return space_detr(trial)
    if "diffusion" in name: return space_diffusion(trial)
    return space_common(trial)

def build_adapter_and_model(model_name, device, hp):
    # Replace with actual implementation
    # adapter = ADAPTERS[model_name]()
    # model = adapter.build(device, hp)
    # return adapter, model
    raise NotImplementedError(f"Please implement adapter for {model_name}")

def build_dataloaders(data_root, batch_size=16, num_workers=4):
    # Replace with actual implementation
    raise NotImplementedError("Please implement build_dataloaders")

def train_one_epoch(model, train_loader, optimizer, device, scaler=None, **kwargs):
    # Replace with actual implementation
    raise NotImplementedError("Please implement train_one_epoch")

def evaluate(model, val_loader, device, thr=0.5):
    # Replace with actual implementation
    raise NotImplementedError("Please implement evaluate")

def objective(trial, args):
    set_seed(args.seed)
    
    hp = get_search_space(args.model, trial)
    
    train_loader, val_loader, _ = build_dataloaders(
        args.data_root, batch_size=hp["bs"], num_workers=hp["num_workers"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = hp.get("use_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    torch.backends.cudnn.benchmark = True
    
    adapter, model = build_adapter_and_model(args.model, device, hp)
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    
    best_loss = float('inf')
    best_state = None
    
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        va_loss, va_map = evaluate(model, val_loader, device)
        
        trial.report(tr_loss, ep)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if tr_loss < best_loss:
            best_loss = tr_loss
            best_state = model.state_dict()
    
    trial.set_user_attr("best_state", best_state)
    return best_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ADAPTERS.keys())
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default="./detection_hpo")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{args.model}_hpo",
        storage=f"sqlite:///{args.save_dir}/{args.model}_hpo.db",
        load_if_exists=True
    )
    
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, timeout=args.timeout)
    
    print(f"\nBest Loss: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    
    best_state = study.best_trial.user_attrs["best_state"]
    ckpt_path = os.path.join(args.save_dir, f"best_{args.model}_hpo.pt")
    torch.save({"model": best_state, "loss": study.best_value, "params": study.best_params}, ckpt_path)
    print(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()
