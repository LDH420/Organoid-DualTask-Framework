from torch.utils.data import Dataset, DataLoader, random_split
import os, glob, argparse, random
from torchvision import transforms
import numpy as np
import torch
import cv2
import glob
from PIL import Image


def binarize_mask(arr: np.ndarray, thr: int = 128) -> np.ndarray:
    if arr.max() <= 1.0:   # 概率或[0,1]
        arr = (arr >= 0.5).astype(np.uint8)
    else:                  # 灰度0~255
        arr = (arr >= thr).astype(np.uint8)
    return arr.astype(np.float32)


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256,256), mask_suffix="", aug=True):   # 原本（256， 256）
        self.size = size
        self.samples = []

        img_exts = ("*.jpg", "*.jpeg", "*.png")
        img_paths = []
        for e in img_exts:
            img_paths.extend(glob.glob(os.path.join(img_dir, e)))
        img_paths = sorted(img_paths)

        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            mask_candidates = [
                os.path.join(mask_dir, f"{stem}{mask_suffix}.png"),
                os.path.join(mask_dir, f"{stem}{mask_suffix}.jpg"),
                os.path.join(mask_dir, f"{stem}{mask_suffix}.npy"),
            ]
            mp = None
            for c in mask_candidates:
                if os.path.exists(c):
                    mp = c
                    break
            if mp is None:
                raise RuntimeError(f"Mask not found for image: {ip}")
            self.samples.append((ip, mp))

        self.img_tf = transforms.Compose([
            transforms.Resize(self.size, interpolation=Image.BILINEAR),
            transforms.Grayscale(num_output_channels=1),  # 确保图像是灰度图像，且单通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像的归一化
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(self.size, interpolation=Image.NEAREST),  # 使用 NEAREST 保证掩码为整数
            transforms.ToTensor()
        ])
        self.aug = aug

    def __len__(self):
        return len(self.samples)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，flipCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        image = np.array(image)
        flip = cv2.flip(image, flipCode)
        flip = np.uint8(flip)    #  RGB
        flip_image = Image.fromarray(flip)
        return flip_image


    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size)  # 调整图像大小
        img = Image.fromarray(img)  # 转换为 PIL 图像

        if mask_path.endswith(".npy"):
            mask_np = np.load(mask_path)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]  # 取第一通道
        else:
            mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            # mask_np = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # 读取为RGB
            mask_np = cv2.resize(mask_np, self.size, interpolation=cv2.INTER_NEAREST)  # 调整大小

        # 标签二值化
        mask_np = (mask_np > 0).astype(np.float32)  # 二值化掩码，确保是0或1

        # 数据增强
        if self.aug:
            flipCode = random.choice([None, -1, 0, 1])

            if flipCode is not None:
                img = self.augment(img, flipCode)
                mask_np = self.augment(mask_np, flipCode)
        # flipCode = random.choice([-1, 0, 1])  # 随机选择翻转方式
        # if flipCode != 2:
        #     img = self.augment(img, flipCode)
        #     mask_np = self.augment(mask_np, flipCode)

        # 转换为张量
        img = self.img_tf(img)  # [1, H, W]
        mask_np = np.array(mask_np)
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]

        return img, mask, os.path.basename(img_path)

# class SegDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, size=(256,256), mask_suffix="", aug=True):
#         self.size = size
#         self.samples = []
#
#         img_exts = ("*.jpg","*.jpeg","*.png")
#         img_paths = []
#         for e in img_exts:
#             img_paths.extend(glob.glob(os.path.join(img_dir, e)))
#         img_paths = sorted(img_paths)
#
#         for ip in img_paths:
#             stem = os.path.splitext(os.path.basename(ip))[0]
#             mask_candidates = [
#                 os.path.join(mask_dir, f"{stem}{mask_suffix}.png"),
#                 os.path.join(mask_dir, f"{stem}{mask_suffix}.jpg"),
#                 os.path.join(mask_dir, f"{stem}{mask_suffix}.npy"),
#             ]
#             mp = None
#             for c in mask_candidates:
#                 if os.path.exists(c): mp = c; break
#             if mp is None:
#                 raise RuntimeError(f"Mask not found for image: {ip}")
#             self.samples.append((ip, mp))
#
#         self.img_tf = transforms.Compose([
#             #transforms.Lambda(lambda im: im.convert("RGB")),
#             transforms.Resize(self.size, interpolation=Image.BILINEAR),
#             transforms.ToTensor(),
#             #transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # 数据是3维
#             #解决灰度图是1通道问题................
#
#             transforms.Normalize(mean=[0.5], std=[0.5]) # 数据是2维
#         ])
#         self.aug = aug
#
#     def __len__(self): return len(self.samples)
#
#     def __getitem__(self, idx):
#         img_path, mask_path = self.samples[idx]
#         img = Image.open(img_path).convert("L")
#         if mask_path.endswith(".npy"):
#             mask_np = np.load(mask_path)
#             if mask_np.ndim == 3: mask_np = mask_np[...,0]
#         else:
#             mask_np = np.array(Image.open(mask_path).convert("L"))
#         # mask_np = binarize_mask(mask_np)
#
#         # resize
#         # img = transforms.functional.resize(img, self.size, interpolation=transforms.InterpolationMode.BILINEAR)
#         mask_pil = Image.fromarray(mask_np.astype(np.uint8))
#         mask_pil = transforms.functional.resize(mask_pil, self.size, interpolation=transforms.InterpolationMode.NEAREST)
#         mask_np = np.array(mask_pil, dtype=np.uint8)
#         mask_np = binarize_mask(mask_np)
#         # if mask_np.max() <= 1:
#         #     # 概率或0/1掩码 → 用 0.5 做阈值
#         #     mask_np = (mask_np >= 0.5).astype(np.float32)
#         # else:
#         #     # 0~255 灰度掩码 → 用 128 做阈值
#         #     mask_np = (mask_np >= 128).astype(np.float32)
#
#         img = self.img_tf(img)                               # [3,H,W]
#         mask = torch.from_numpy(mask_np).unsqueeze(0).float()# [1,H,W]
#
#
#         return img, mask, os.path.basename(img_path)
#         # --- resize image ---

def build_dataloaders(
    data_root=None,
    img_dir=None, mask_dir=None,
    mask_suffix="",
    size=(256, 256),
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size=16,
    num_workers=2,
):
    """
    返回: train_loader, val_loader, test_loader(可能为 None)
    - 若提供 data_root：默认假设 root 下有 training/validation/testing 三个子目录，各自含 images/ 和 masks/
    - 若未提供 data_root：使用 img_dir / mask_dir（单目录），并做随机划分
    """
    if data_root is not None:
        tr_img = os.path.join(data_root, "training",   "images")
        tr_msk = os.path.join(data_root, "training",   "masks")
        va_img = os.path.join(data_root, "validation", "images")
        va_msk = os.path.join(data_root, "validation", "masks")
        te_img = os.path.join(data_root, "testing",    "images")
        te_msk = os.path.join(data_root, "testing",    "masks")

        train_ds = SegDataset(tr_img, tr_msk, size=tuple(size), mask_suffix=mask_suffix, aug=True)
        val_ds   = SegDataset(va_img, va_msk, size=tuple(size), mask_suffix=mask_suffix, aug=False)
        test_ds  = SegDataset(te_img, te_msk, size=tuple(size), mask_suffix=mask_suffix, aug=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    # else:
    #     if img_dir is None or mask_dir is None:
    #         raise ValueError("未提供 data_root 时必须指定 img_dir 与 mask_dir")
    #     full_ds = SegDataset(img_dir, mask_dir, size=tuple(size), mask_suffix=mask_suffix, aug=True)
    #     val_len   = max(1, int(len(full_ds) * val_ratio))
    #     train_len = len(full_ds) - val_len
    #     train_ds, val_ds = random_split(
    #         full_ds, [train_len, val_len],
    #         generator=torch.Generator().manual_seed(42)
    #     )
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
    #                               num_workers=num_workers, pin_memory=True)
    #     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
    #                               num_workers=num_workers, pin_memory=True)
    #     test_loader  = None
    else:
        if img_dir is None or mask_dir is None:
            raise ValueError("未提供 data_root 时必须指定 img_dir 与 mask_dir")

        full_ds = SegDataset(img_dir, mask_dir, size=tuple(size), mask_suffix=mask_suffix, aug=True)

        # 计算训练集、验证集和测试集的长度
        total_len = len(full_ds)
        val_len = int(total_len * val_ratio)
        test_len = int(total_len * test_ratio)
        train_len = total_len - val_len - test_len

        # 确保划分后的数据集长度是合理的
        val_len = max(1, val_len)
        test_len = max(1, test_len)
        # train_len = total_len - val_len - test_len

        # 使用 random_split 划分数据集为训练集、验证集和测试集
        train_ds, val_ds, test_ds = random_split(
            full_ds, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建 DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader