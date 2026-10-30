import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.measure import label, regionprops
from tqdm import tqdm


# ============================================================
# 1. 提取单个数据集的 organoid 面积
# ============================================================

def extract_areas_from_dataset(root_path, dataset_name):

    areas = []

    sub_folders = [
        "training",
        "validation",
        "testing"
    ]

    print(f"\nProcessing dataset: {dataset_name}")

    for sub in sub_folders:

        mask_dir = os.path.join(
            root_path,
            dataset_name,
            sub,
            "masks"
        )

        if not os.path.exists(mask_dir):
            continue


        mask_files = [
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".bmp")
            )
        ]


        for mask_name in tqdm(
            mask_files,
            desc=f"{dataset_name}-{sub}",
            leave=False
        ):

            mask_path = os.path.join(
                mask_dir,
                mask_name
            )

            try:

                # 读取mask
                mask = cv2.imdecode(
                    np.fromfile(
                        mask_path,
                        dtype=np.uint8
                    ),
                    cv2.IMREAD_UNCHANGED
                )


                if mask is None:
                    continue


                # 多通道转灰度
                if mask.ndim == 3:
                    mask = cv2.cvtColor(
                        mask,
                        cv2.COLOR_BGR2GRAY
                    )


                # 前景
                binary_mask = (
                    mask > 0
                ).astype(np.uint8)


                # 连通区域
                labeled_mask = label(
                    binary_mask,
                    connectivity=2
                )


                regions = regionprops(
                    labeled_mask
                )


                for region in regions:

                    area = region.area

                    # 去除极小噪声
                    if area >= 10:

                        areas.append(area)


            except Exception as e:

                print(
                    f"Failed: {mask_path}, {e}"
                )


    df = pd.DataFrame(
        {
            "area": areas,
            "dataset": dataset_name
        }
    )

    return df



# ============================================================
# 2. 汇总所有数据集
# ============================================================

def build_dataframe(root_path, dataset_names):

    dfs = []

    for name in dataset_names:

        df = extract_areas_from_dataset(
            root_path,
            name
        )

        if len(df) > 0:
            dfs.append(df)


    if len(dfs) == 0:
        raise RuntimeError(
            "No object area extracted!"
        )


    full_df = pd.concat(
        dfs,
        ignore_index=True
    )


    # log10(pixel^2)
    full_df["log_area"] = np.log10(
        full_df["area"]
    )


    return full_df



# ============================================================
# 3. 输出统计表
# ============================================================

def save_statistics(full_df):

    stats = (
        full_df
        .groupby("dataset")["area"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
        .reset_index()
    )


    print("\n===== Object size statistics =====")

    print(stats)


    stats.to_csv(
        "organoid_object_size_statistics.csv",
        index=False
    )



# ============================================================
# 4. 绘制 violin plot
# ============================================================

def plot_violin(full_df):


    sns.set_theme(
        style="whitegrid"
    )


    plt.figure(
        figsize=(12,7)
    )


    # 按median排序
    order = (
        full_df
        .groupby("dataset")["area"]
        .median()
        .sort_values()
        .index
    )


    ax = sns.violinplot(
        data=full_df,
        x="dataset",
        y="log_area",
        order=order,
        inner=None,
        cut=0,
        bw_adjust=1.0
    )


    sns.boxplot(
        data=full_df,
        x="dataset",
        y="log_area",
        order=order,
        width=0.15,
        showfliers=True,
        boxprops={
            "facecolor":"white",
            "zorder":3
        },
        flierprops={
            "marker":"o",
            "markersize":3,
            "alpha":0.4
        },
        ax=ax
    )


    ax.set_xlabel(
        "Dataset",
        fontsize=12
    )

    ax.set_ylabel(
        "Organoid area (log10 pixel²)",
        fontsize=12
    )


    ax.set_title(
        "Distribution of Organoid Object Sizes Across Datasets",
        fontsize=15,
        fontweight="bold"
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        "/data/liudenghui/dataset/organoID/Organoid_Object_Size_Distribution.png",
        dpi=600,
        bbox_inches="tight"
    )


    plt.show()



# ============================================================
# main
# ============================================================

if __name__ == "__main__":


    root_path = r"/data/liudenghui/dataset/organoID/"


    dataset_names = [

        "OriginalData",

        "Dataset_organoids",

        "MouseOrganoids",

        "OrganoidBasic",

        "stomach_intance"

    ]

    df = build_dataframe(
        root_path,
        dataset_names
    )

    save_statistics(df)
    plot_violin(df)