import argparse, torch
import numpy as np
import random
from torch.utils.data import ConcatDataset, Subset, DataLoader


from jinja2.nodes import Break
from torch.optim import AdamW
from baseline3.common import runner
from baseline3.common.runner import train_one_epoch, evaluate
from baseline3.dataset import build_dataloaders
from baseline3.adapters.gscnn_adapter import GSCNNAdapter
from baseline3.adapters.unet_adapter import UNetAdapter
from baseline3.adapters.segformer_adapter import SegformerAdapter
from baseline3.adapters.deeplabv3_adapter import DeepLabV3PlusAdapter
from baseline3.adapters.transunet_adapter import TransUNetAdapter
from baseline3.adapters.TransResUnetAdapter import TransResUnetAdapter
from baseline3.adapters.unetpp_adapter import UNetPPAdapter
from baseline3.adapters.PSPNetAdapter import PSPNetAdapter
from baseline3.adapters.H2FormerAdapter import H2FormerAdapter
from baseline3.adapters.ResUnetPlusPlus_adapter import ResUnetPlusPlusAdapter
from baseline3.adapters.MEGA_Net_adapter import MEGANetAdapter
from baseline3.adapters.MBSNet_adapter import MBSNetAdapter
from baseline3.adapters.CPCANet_adapter import CPCANetAdapter
from baseline3.adapters.LVNet_adapter import LV_UNetAdapter
from baseline3.adapters.unet_AMP_adapter import UNet_AMP_Adapter
from baseline3.adapters.unet_CAS_adapter import UNet_CAS_Adapter
from baseline3.adapters.unet_AMP_CAS_adapter import UNet_AMP_CAS_Adapter
from baseline3.adapters.CSCA_adapter import CSCAUNetAdapter
from baseline3.adapters.DSCA_adapter import DCSAUAdapter
from baseline3.adapters.sam2_adapter import SAM2Adapter
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline3.adapters.ICSA_adapter import ICSAUNetAdapter

ADAPTERS = {
    "gscnn": GSCNNAdapter,
    "unet": UNetAdapter,
    "segformer": SegformerAdapter,
    "deeplabv3plus": DeepLabV3PlusAdapter,
    "transunet": TransUNetAdapter,
    "unetpp": UNetPPAdapter,
    "transresunet": TransResUnetAdapter,
    "pspnet": PSPNetAdapter,
    "h2former": H2FormerAdapter,
    "ResUnetPlusPlus": ResUnetPlusPlusAdapter,
    "MEGA": MEGANetAdapter,
    "LVNet": LV_UNetAdapter,
    "CPCANet": CPCANetAdapter,
    "MBSNet": MBSNetAdapter,
    "unet_AMP": UNet_AMP_Adapter,
    "unet_CAS": UNet_CAS_Adapter,
    "unet_AMP_CAS": UNet_AMP_CAS_Adapter,
    "CSCA": CSCAUNetAdapter,
    "ICSA": ICSAUNetAdapter,
    "DSCA": DCSAUAdapter,
    "sam2": SAM2Adapter,


}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 如果希望使用确定性算法，可以设置以下参数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭优化以保持结果的一致性

set_seed(42)

class EarlyStopper:
    def __init__(self, patience=15, min_delta=1e-4):  # 默认 patience 为 10
        self.patience = patience  # 设置 patience 为 10
        self.min_delta = min_delta  # 设置最小变化量为 0.0001
        # self.best = float('inf')  # 初始化为无穷大，表示初始最优损失
        self.best = float(0)  # 初始化为无穷大，表示初始最优损失
        self.bad = 0  # 用于计数无改善的轮数

    # def step(self, metric: float) -> bool:
    #     # 如果当前的 metric 小于最优值减去 min_delta，表示有明显改善
    #     if metric < self.best - self.min_delta:
    #         self.best = metric  # 更新最优损失
    #         self.bad = 0  # 重置无改善计数
    #         return False  # 不触发早停
        # 如果损失变化很小，小于 min_delta，认为没有改进
        # self.bad += 1  # 增加无改善计数
        # return self.bad >= self.patience  # 如果无改善超过 patience 轮，触发早停

# 根据dice进行 优化早停
    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:  # Dice 应该是越大越好
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ADAPTERS.keys(), required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_dir", type=str, default="./ckpts")
    ap.add_argument("--num_workers", type=int, default=25)
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    # 加载保存的最佳参数和权重
    model_name = args.model  # 这是模型名称，比如 "unet" 或 "gscnn"
    dataset_name = os.path.basename(args.data_root)
    checkpoint_path = f'/data/liudenghui/baseline/ckpts_hpo/best_{model_name}_{dataset_name}_hpo1.pt'
    checkpoint = torch.load(checkpoint_path)
    best_params = checkpoint['params']
    best_state_dict = checkpoint['model']   # 模型和模型权重一对一加载

    # # 固定使用 unet 的 HPO 超参数
    # dataset_name = os.path.basename(args.data_root)
    # hpo_source_model = 'unet_AMP_CAS'
    # checkpoint_path = f'/home/liudenghui/baseline/baseline3/ckpts_hpo/best_{hpo_source_model}_{dataset_name}_hpo1.pt'
    #
    # checkpoint = torch.load(checkpoint_path)
    # best_params = checkpoint['params']

    # 使用最佳超参数更新训练配置
    args.lr = best_params['lr']
    args.bs = best_params['bs']  # batch_size 12
    # args.num_workers = best_params['num_workers']
    args.thr = best_params['thr']
    args.weight_decay = best_params['weight_decay']
    args.dice_w = best_params['dice_w']
    print("\n===== 使用 HPO 超参数 =====")
    print(
        f"lr={args.lr}, bs={args.bs}, thr={args.thr}, "
        f"weight_decay={args.weight_decay}, dice_w={args.dice_w}, "
        f"num_workers={args.num_workers}"
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 小加速
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root,
        # img_dir=args.img_dir,
        # # 如果没有 data_root，传入 img_dir 和 mask_dir
        # mask_dir=args.mask_dir,
        batch_size=args.bs)

    adapter = ADAPTERS[args.model]()
    # optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"\n===== 开始训练模型 {args.model} =====")
    model = adapter.build(device)
    # model.load_state_dict(best_state_dict)   #  原加载超参数和权重****

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, threshold_mode="abs", threshold=1e-4, verbose=True)

    # 早停器
    early = EarlyStopper(patience=15, min_delta=1e-4)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == "cuda") else None
    best = (-1, None)
    best_loss = float('inf')  # 初始化最佳损失为正无穷
    best_model = None

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, dice_w=args.dice_w)
        vloss, va_dice, viou, vprecision, vacc, vrecall,  vnsd, vhd95 = evaluate( model, val_loader, device, thr=args.thr, dice_w=args.dice_w,)
        # return tot_loss / n, tot_dice / n, tot_iou / n, tot_precision / n, tot_acc / n, tot_recall / n, total_nsd / n, total_hd95 / n

        print(
            f"[[{ep:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f}  val_loss={vloss:.4f}  "
            f"val_dice={va_dice:.4f}      val_iou={viou:.4f}   "
            f"val_precision={vprecision:.4f}      val_acc={vacc:.4f}   "
            f"val_recall={vrecall:.4f}        val_NSD={vnsd:.4f}  "
            f"val_HD95={vhd95:.4f} "
        )

        # print(
        #     f"[{ep:03d}/{args.epochs}] "
        #     f"train_loss={tr_loss:.4f}  val_loss={vloss:.4f}  "
        #     f"val_iou={viou:.4f}  val_dice={va_dice:.4f} val_HD95={vhd95:.4f}  val_pix_acc={vpix_accuracy:.4f} val_nsd={vnsd:.4f}  "
        # )

        # if tr_loss < best_loss:
        #     best_loss = tr_loss
        #     best_model = model.state_dict()
        #     print(f"[VAL {ep}] loss={vloss:.4f} IoU={viou:.4f} Dice={vdice:.4f}")    # 修改
        # if vloss < best_loss:
        #     best_loss = vloss
        #     best_model = model.state_dict()
        #     print(f"[VAL {ep}] loss={vloss:.4f} IoU={viou:.4f} Dice={vdice:.4f}")    # 修改
        if va_dice > best[0]:
            best = (va_dice, model.state_dict())

            save_dir = args.save_dir  # 例如 "C:/python_file/pythonProject/baseline/baseline3/ckpts"
            os.makedirs(save_dir, exist_ok=True)  # ← 确保目录存在

            save_path = os.path.join(save_dir, f"best_{args.model}_{dataset_name}_hpo1.pt")
            print(f"  ✓ 保存最佳权重到: {save_path}  (dice={best[0]:.4f})", flush=True)
            torch.save({"epoch": ep, "model": best[1], "dice": best[0]}, save_path)
            # torch.save({"epoch": ep, "model": best_model, "loss": best_loss}, save_path)
            # print(f"  ✓ 保存最佳权重到: {save_path}  (loss={best_loss:.4f})", flush=True)


        scheduler.step(va_dice)

        # 触发早停则 break
        # if early.step(tr_loss):
        if early.step(va_dice):
            print(f"  ↳ 早停触发（patience={early.patience}），停止于 epoch {ep}")
            break
    # 测试
    if test_loader is not None and best[1] is not None:
    # if test_loader is not None and best_model is not None:
    #     model.load_state_dict(best_model)
        model.load_state_dict(best[1])
        loss, dice, iou, precision, acc, recall, nsd, hd95 = evaluate( model, test_loader, device=device, thr=args.thr)
        # print(f"[TEST] loss={loss:.4f} IoU={iou:.4f} Dice={dice:.4f} NSD={nsd:.4f} HD95={HD95:.4f} pix_acc={pix_accuracy:.4f}")
        print(
            f"[TEST] "
            f"loss={loss:.4f} Dice={dice:.4f} IoU={iou:.4f}  "
            f"precision={precision:.4f} acc={acc:.4f}  "
            f"recall={recall:.4f}    NSD={nsd:.4f} "
            f"HD95={hd95:.4f} "
        )

