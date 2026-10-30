import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from baseline3.adapters.segformer_adapter import SegformerAdapter
from baseline3.adapters.unet_adapter import UNetAdapter
from baseline3.adapters.gscnn_adapter import GSCNNAdapter
from baseline3.adapters.deeplabv3_adapter import DeepLabV3PlusAdapter
from baseline3.adapters.transunet_adapter import TransUNetAdapter


# unet  deplabv3+  segformer transunet

def load_model(checkpoint_path, device):
    """
    加载模型权重
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())

    adapter = UNetAdapter()   # 按需切换模型适配器
    model = adapter.build(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def preprocess_image(img_path, input_size=(256, 256), normalize=True):
    """
    图像预处理：
    1. 读灰度图
    2. 记录原图尺寸
    3. resize 到模型输入大小
    4. 转 tensor
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    original_h, original_w = img.shape[:2]

    img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.reshape(1, 1, img_resized.shape[0], img_resized.shape[1])

    img_tensor = torch.from_numpy(img_resized).float()

    if normalize:
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

    return img_tensor, (original_h, original_w)


def postprocess_prediction(pred, threshold=0.5):
    """
    将概率图后处理成二值 mask
    pred: 2D numpy array, range [0,1]
    """
    pred_binary = (pred >= threshold).astype(np.uint8) * 255
    return pred_binary


def inference_single_image(img_path, model, device, save_dir, threshold=0.5, input_size=(256, 256)):
    """
    单张图像推理，并保存预测 mask
    """
    model.eval()

    img_tensor, original_size = preprocess_image(img_path, input_size=input_size)
    original_h, original_w = original_size

    with torch.no_grad():
        pred = model(img_tensor.to(device))

        if isinstance(pred, tuple):
            pred = pred[0]

        pred = torch.sigmoid(pred)

    # 取出单张图像预测结果
    pred = pred.cpu().numpy()[0][0]   # shape: [H, W], 范围 [0,1]

    # 二值化
    pred_mask = postprocess_prediction(pred, threshold=threshold)

    # 关键：恢复到原图大小，保证后续和 GT 对齐
    pred_mask = cv2.resize(
        pred_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, pred_mask)

    return save_path


def batch_inference(image_dir, model, device, save_dir, threshold=0.5, input_size=(256, 256)):
    """
    对整个文件夹进行批量推理
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    image_list = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_exts)
    ])

    if len(image_list) == 0:
        raise ValueError(f"在目录中没有找到图像: {image_dir}")

    print(f"共找到 {len(image_list)} 张测试图像")
    print(f"预测结果将保存到: {save_dir}")

    for img_name in tqdm(image_list, desc="Batch inference"):
        img_path = os.path.join(image_dir, img_name)
        try:
            save_path = inference_single_image(
                img_path=img_path,
                model=model,
                device=device,
                save_dir=save_dir,
                threshold=threshold,
                input_size=input_size
            )
            # 如需调试可打开
            # print(f"Saved: {save_path}")
        except Exception as e:
            print(f"推理失败: {img_path}")
            print(f"原因: {e}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 改这里 =========
    model_path = '/home/liudenghui/baseline/baseline3/ckpts/best_unet_stomach_intance_hpo1.pt'
    test_image_dir = '/home/liudenghui/baseline/dataset/organoID/stomach_intance/testing/images'
    save_dir = '/data/liudenghui/baseline3/predict/stomach_intance/unet_stomach_intance'
    input_size = (256, 256)
    threshold = 0.5
    # =========================

    model = load_model(model_path, device)

    batch_inference(
        image_dir=test_image_dir,
        model=model,
        device=device,
        save_dir=save_dir,
        threshold=threshold,
        input_size=input_size
    )

    print("全部测试图像推理完成。")