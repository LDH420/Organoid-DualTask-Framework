import torch, argparse, glob
import cv2
import os
import numpy as np
from baseline3.common.runner import evaluate
from baseline3.dataset import build_dataloaders
from baseline3.adapters.gscnn_adapter import GSCNNAdapter
from baseline3.adapters.unet_adapter import UNetAdapter
from baseline3.adapters.swinunetr_adapter import SwinUNETRAdapter
from baseline3.adapters.segformer_adapter import SegformerAdapter
from baseline3.adapters.deeplabv3_adapter import DeepLabV3PlusAdapter
from baseline3.adapters.transunet_adapter import TransUNetAdapter
from baseline3.adapters.TransResUnetAdapter import TransResUnetAdapter
from baseline3.adapters.unetpp_adapter import UNetPPAdapter
from baseline3.adapters.PSPNetAdapter import PSPNetAdapter
from baseline3.adapters.H2FormerAdapter import H2FormerAdapter
from baseline3.adapters.ResUnetPlusPlus_adapter import ResUnetPlusPlusAdapter


# 这里可以根据你选择的模型名称替换
model_name = 'unetpp'  # 选择你训练时使用的模型名称
checkpoint_path = '/path/to/your/checkpoint.pt'  # 模型检查点路径
dataset_name = 'dataset_name'  # 根据数据集名称修改

# 加载模型
ADAPTERS = {
    "gscnn": GSCNNAdapter,
    "unet": UNetAdapter,
    "swinunetr": SwinUNETRAdapter,
    "segformer": SegformerAdapter,
    "deeplabv3plus": DeepLabV3PlusAdapter,
    "transunet": TransUNetAdapter,
    "unetpp": UNetPPAdapter,
    "transresunet": TransResUnetAdapter,
    "pspnet": PSPNetAdapter,
    "h2former": H2FormerAdapter,
    "ResUnetPlusPlus": ResUnetPlusPlusAdapter

}


def load_model(model_name, dataset_name, checkpoint_dir, device):
    """
    动态加载指定模型和数据集的预训练权重

    Parameters:
    - model_name (str): 模型名称，例如 'unet', 'gscnn', 等
    - dataset_name (str): 数据集名称，用于生成模型的存储路径
    - checkpoint_dir (str): 权重文件存储的目录路径
    - device (torch.device): 用于推理的设备（CPU或CUDA）

    Returns:
    - model (torch.nn.Module): 加载的模型
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 动态生成检查点路径
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}_{dataset_name}_hpo1.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path)

    # 选择模型适配器
    adapter = ADAPTERS.get(model_name)
    if adapter is None:
        raise ValueError(f"Model name '{model_name}' not found in adapters.")

    # 初始化并加载模型权重
    adapter = adapter()
    model = adapter.build(device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为推理模式

    print(f"Model '{model_name}' for dataset '{dataset_name}' loaded successfully from {checkpoint_path}")

    return model


def preprocess_images(directory_path, target_size=(256, 256)):
    """
    批量读取和预处理目录下的所有图像

    Parameters:
    - directory_path (str): 包含图像的目录路径
    - target_size (tuple): 目标大小，用于调整图像尺寸

    Returns:
    - img_tensors (torch.Tensor): 批量预处理后的图像张量
    - origin_shapes (list): 所有图像的原始尺寸列表
    """
    img_tensors = []
    origin_shapes = []

    # 获取目录下所有的图像文件路径，假设是PNG格式
    image_paths = glob.glob(os.path.join(directory_path, "*.png"))
    # 你也可以根据需要支持其他格式，如jpg:
    # image_paths += glob.glob(os.path.join(directory_path, "*.jpg"))

    for image_path in image_paths:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            continue  # 如果图片无法读取，跳过

        origin_shape = img.shape
        origin_shapes.append(origin_shape)

        # 图像预处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
        img = cv2.resize(img, target_size)  # 调整大小

        # 转换为tensor
        img = img.reshape(1, 1, img.shape[0], img.shape[1])  # 扩展维度，成为1x1xHxW格式
        img_tensor = torch.from_numpy(img).to(torch.float32)
        img_tensors.append(img_tensor)

    # 将所有图像张量堆叠成一个批次（batch）
    img_tensors = torch.cat(img_tensors, dim=0)

    return img_tensors, origin_shapes


def postprocess_prediction(pred, origin_shape):
    """
    对预测结果进行后处理

    Parameters:
    - pred (numpy.array): 预测结果
    - origin_shape (tuple): 原图尺寸

    Returns:
    - pred (numpy.array): 后处理后的预测图像
    """
    # 后处理：二值化
    pred = np.array(pred.data.cpu()[0])[0]
    print(f"Min value in pred: {pred.min()}, Max value in pred: {pred.max()}")  # 打印最小和最大值，检查值范围
    # 使用 sigmoid 激活函数将预测结果限制在 [0, 1] 范围内
    pred = torch.sigmoid(torch.tensor(pred))  # 使用 sigmoid 将其映射到 [0, 1] 范围
    pred = pred.numpy()  # 转回 numpy 数组
    print(f"Min value in pred: {pred.min()}, Max value in pred: {pred.max()}")  # 打印最小和最大值，检查值范围
    threshold = 0.0001
    pred[pred >= threshold] = 255
    pred[pred < threshold] = 0
    # pred = np.uint8(pred)
    processed_preds = []

    for i, origin_shape in enumerate(origin_shape):
        # 从 origin_shape 提取 height 和 width
        if len(origin_shape) == 3:  # 包含通道数 (height, width, channels)
            height, width = origin_shape[0], origin_shape[1]
        elif len(origin_shape) == 2:  # 如果没有通道数信息 (height, width)
            height, width = origin_shape
        else:
            raise ValueError(f"Unexpected shape of origin_shape: {origin_shape}")
    # 恢复原图大小
    pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
    processed_pred = cv2.resize(pred[i], (width, height), interpolation=cv2.INTER_NEAREST)
    processed_preds.append(processed_pred)

    return processed_preds


def main():
    # 设置命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ADAPTERS.keys(), required=True, help="模型名")
    ap.add_argument("--data_root", required=True, help="数据集")
    ap.add_argument("--test_image", required=True, help="测试图像路径")
    # ap.add_argument("--checkpoint_dir", required=True, help="模型检查点存储路径")
    ap.add_argument("--save_dir", required=True, help="预测结果保存路径")
    args = ap.parse_args()

    # '/home/liudenghui/baseline/dataset/organoID/OrganoidBasic_v20211206/testing/'

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    root = '/home/liudenghui/baseline/baseline3/ckpts'
    # 加载模型
    dataset_name = os.path.basename(args.data_root)
    checkpoint_path = os.path.join(root, f"best_{args.model}_{dataset_name}_hpo1.pt")

    # checkpoint = torch.load(checkpoint_path)
    model = load_model(args.model, dataset_name, root, device)

    # 预处理图像
    img_tensor, origin_shape = preprocess_images(args.test_image)

    # 预测
    with torch.no_grad():
        pred = model(img_tensor.to(device))

    # 后处理
    pred = postprocess_prediction(pred, origin_shape)

    # 检查 pred 是否为空或无效
    if not pred:
        raise ValueError("Prediction is empty or invalid.")

    # 如果 pred 是一个列表，逐个打印每个预测图像的形状
    if isinstance(pred, list):
        for i, p in enumerate(pred):
            print(f"Predicted image {i} shape: {p.shape}")
    else:
        # 如果 pred 是一个单一的 NumPy 数组或 Tensor
        print(f"Predicted image shape: {pred.shape}")


    # 保存预测结果
    save_path = os.path.join(args.save_dir, f"{dataset_name}_prediction.png")
    pred = np.array(pred, dtype=np.uint8)
    pred = cv2.resize(pred[0], (512, 512))
    cv2.imwrite(save_path, pred)
    print(f"Prediction saved to {save_path}")


if __name__ == "__main__":
    main()