import os, argparse, optuna, torch, numpy as np, random
from torch.optim import AdamW
from baseline3.common.runner import train_one_epoch, evaluate
from baseline3.adapters.gscnn_adapter import GSCNNAdapter
from baseline3.adapters.unet_adapter import UNetAdapter
from baseline3.adapters.swinunetr_adapter import SwinUNETRAdapter
from baseline3.adapters.deeplabv3_adapter import DeepLabV3PlusAdapter
from baseline3.adapters.segformer_adapter import SegformerAdapter
from baseline3.adapters.transunet_adapter import TransUNetAdapter
from baseline3.adapters.TransResUnetAdapter import TransResUnetAdapter
from baseline3.train_baseline import build_dataloaders
from baseline3.adapters.unetpp_adapter import UNetPPAdapter
from baseline3.adapters.PSPNetAdapter import PSPNetAdapter
from baseline3.adapters.H2FormerAdapter import H2FormerAdapter
from baseline3.adapters.ResUnetPlusPlus_adapter import ResUnetPlusPlusAdapter
from baseline3.adapters.MEGA_Net_adapter import MEGANetAdapter
from baseline3.adapters.MBSNet_adapter import MBSNetAdapter
from baseline3.adapters.CPCANet_adapter import CPCANetAdapter
from baseline3.adapters.LVNet_adapter import LV_UNetAdapter


ADAPTERS = {
    "gscnn": GSCNNAdapter,
    "unet": UNetAdapter,
    "swinunetr": SwinUNETRAdapter,
    "segformer": SegformerAdapter,
    "deeplabv3plus": DeepLabV3PlusAdapter,
    "transunet": TransUNetAdapter,
    "unetpp":UNetPPAdapter,
    "transresunet": TransResUnetAdapter,
    "pspnet": PSPNetAdapter,
    "h2former": H2FormerAdapter,
    "MEGA": MEGANetAdapter,
    "LVNet": LV_UNetAdapter,


}

def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def space_common(trial):
    return {
        "lr":        trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "bs":        trial.suggest_categorical("bs", [12, 16, 24]),
        "wd":        trial.suggest_float("weight_decay", 0.0, 5e-4, log=False),
        "dice_w":    trial.suggest_float("dice_w", 0.3, 0.8),
        "thr":       trial.suggest_float("thr", 0.35, 0.65),
        "use_amp":   False,
        "num_workers": trial.suggest_categorical("num_workers",  [2]),

    }

def build_adapter_and_model(model_name, device):
    adapter = ADAPTERS[model_name]()
    model   = adapter.build(device)
    return adapter, model

def objective(trial, args):
    set_seed(args.seed)

    hp = space_common(trial)

    # 2) 数据
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root, batch_size=hp["bs"],
        # img_dir=args.img_dir,  # 如果没有 data_root，传入 img_dir 和 mask_dir
        # mask_dir=args.mask_dir,
        num_workers=hp["num_workers"]
    )

    # 3) 设备/AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use_amp = bool(hp.get("use_amp", False) and device.type == "cuda")
    use_amp = False
    scaler  = None
    # scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    torch.backends.cudnn.benchmark = True

    # 4) 模型 & 优化器
    adapter, model = build_adapter_and_model(args.model, device)
    optim = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

    # best = (-1.0, None)
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    best_model = None
    # 5) 训练若干 epoch，并做中途剪枝
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device, scaler=scaler, dice_w=hp["dice_w"])
        va_loss, va_iou, va_dice, va_nsd, va_hd95, va_pix_acc = evaluate(model, val_loader, device, thr=hp["thr"])
        # trial.report(va_dice, ep)
        trial.report(tr_loss, ep)      # 修改


        if trial.should_prune():
            raise optuna.TrialPruned()

        # 记录最佳
        # if va_dice > best[0]:
        #     best = (va_dice, model.state_dict())

        if tr_loss < best_loss:
            best_loss = tr_loss
            best_model = model.state_dict()           # 修改

    # 6) 这个 trial 的最优 ckpt临时存一份（供最终best trial持久化）
    # trial.set_user_attr("best_state", best[1])   #  优化dice
    # return best[0]  # Optuna会最大化这个返回值
    trial.set_user_attr("best_state", best_model)  # 优化loss
    return best_loss  # Optuna会最大化这个返回值

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ADAPTERS.keys())
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None)  # 秒
    ap.add_argument("--save_dir", type=str, default="./ckpts_hpo")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--amp", action='store_true', help="Enable mixed precision training (AMP)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True) # 贝叶斯
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2) #中位数剪枝
    # pruner  = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                study_name=f"{args.model}_hpo")
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, timeout=args.timeout)

    print("\n=== HPO 完成 ===")
    print("Best Value (Loss):", study.best_value)
    print("Best Params:", study.best_params)
    # 持久化最佳 ckpt & 参数
    best_trial = study.best_trial
    best_state = best_trial.user_attrs["best_state"]
    dataset_name = os.path.basename(args.data_root)
    ckpt_path  = os.path.join(args.save_dir, f"best_{args.model}_{dataset_name}_hpo1.pt")
    torch.save({"model": best_state, "Loss": study.best_value, "params": best_trial.params}, ckpt_path)
    print(f"✓ 已保存最佳权重与参数到: {ckpt_path}")

if __name__ == "__main__":
    main()