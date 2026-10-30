import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 1. 数据准备 (保持原逻辑)
# file_path_new = r'D:\个人文件\文献\分割\实验数据.xlsx'
# file_path_new = r'/home/liudenghui/baseline/baseline3/实验数据.xlsx'
# df_experiment = pd.read_excel(file_path_new)
# filtered_df = df_experiment[['baseline', 'dataset', 'test/dice']].dropna()
#
# dataset_order = ['OriginalData', 'Dataset_organoids', 'MouseOrganoids', 'OrganoidBasic', 'stomach_intance']
# model_order = ['GSCNN', 'U-Net', 'SegFormer', 'UNet++', 'TransUNet', 'DeepLabv3+', 'PSPNet', 'TransResUnet', 'MEGANet',
#                'LVNet']
#
# pivot_df = filtered_df.pivot(index='baseline', columns='dataset', values='test/dice')
# plot_df = pivot_df.loc[model_order, dataset_order].T
#
# # 2. 动态计算颜色尺度
# data_min, data_max = plot_df.values.min(), plot_df.values.max()
# vmin = np.floor(data_min * 20) / 20
# vmax = np.ceil(data_max * 20) / 20
#
# # 精确配色：深蓝 -> 白 -> 深粉红
# colors = ["#6BAED6", "#FFFFFF", "#D6619C"]      # colors = ["#6BAED6", "#FFFFFF", "#D6619C"]
# custom_cmap = LinearSegmentedColormap.from_list("pink_blue", colors)
#
# # colors = ["#F57C00", "#FFD54F", "#F8BBD0", "#F06292"]
# # custom_cmap = LinearSegmentedColormap.from_list("line_chart_match", colors)
#
# # 3. 创建画布
# fig, ax = plt.subplots(figsize=(15, 8))
# rows, cols = plot_df.shape
#
# # --- 关键修改：间隔参数 ---
# gap = 0.08  # 间隔大小，越大方块越小，间隔越宽
# box_width = 1 - gap
# box_height = 1 - gap
#
# for i in range(rows):
#     for j in range(cols):
#         val = plot_df.iloc[i, j]
#         color = custom_cmap((val - vmin) / (vmax - vmin))
#
#         # 创建圆角矩形
#         # boxstyle="round,pad=0.2" 会自动把直角磨圆
#         # 我们把起点稍微偏置 (j + gap/2) 来实现居中
#         rect = patches.FancyBboxPatch(
#             (j + gap / 2, i + gap / 2),
#             box_width, box_height,
#             boxstyle=f"round,pad=0,rounding_size=0.2",  # pad设为0，通过box_width控制大小
#             linewidth=0,
#             edgecolor='none',
#             facecolor=color,
#             zorder=2
#         )
#         ax.add_patch(rect)
#
#         # 添加数值标注 (根据颜色深浅自动反色)
#         # 这里的 0.3 和 0.7 是经验值，用于判断深色还是浅色背景
#         norm_val = (val - vmin) / (vmax - vmin)
#         text_col = 'white' if norm_val > 0.8 or norm_val < 0.2 else 'black'
#
#         ax.text(j + 0.5, i + 0.5, f'{val:.4f}',
#                 ha='center', va='center',
#                 weight='bold', fontsize=10, color=text_col, zorder=3)
#
# # 4. 设置坐标轴
# ax.set_xlim(0, cols)
# ax.set_ylim(0, rows)
# ax.set_aspect('equal')
#
# ax.set_xticks(np.arange(cols) + 0.5)
# ax.set_yticks(np.arange(rows) + 0.5)
# ax.set_xticklabels(plot_df.columns, rotation=45, ha='right', fontsize=11)
# ax.set_yticklabels(plot_df.index, fontsize=11)
#
# # 隐藏边框和刻度线
# ax.spines[:].set_visible(False)
# ax.tick_params(left=False, bottom=False)
#
# # 5. 颜色条设置
# sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.03, aspect=20)
# cbar.set_label('Dice Score', weight='bold', labelpad=10)
# cbar.outline.set_visible(False)
#
# # plt.title('Segmentation Model Performance (Rounded with Gaps)', fontsize=16, pad=35, weight='bold')
# plt.tight_layout()
#
# # 保存
# heatmap_file_path = r'D:\个人文件\文献\分割\segmentation_rounded_gap_heatmap.png'
# plt.savefig(heatmap_file_path, dpi=600, bbox_inches='tight')
# plt.show()

def prepare_heatmap_dataframe(file_path_new):
    """
    从实验数据 Excel 中提取 test/dice，并按
    baseline × dataset 计算平均 Dice 后绘制热图。

    如果 Excel 中同一模型、同一数据集存在多行记录（例如：
    五折交叉验证结果、重复实验结果），这里不会直接 pivot，
    而是先求算术平均值，再生成唯一的热图单元格。

    热图最终显示：
        行 = dataset
        列 = model
        值 = mean(test/dice)
    """
    df_experiment = pd.read_excel(file_path_new)

    # 只保留热图需要的三列
    filtered_df = df_experiment[['baseline', 'dataset', 'test/dice']].copy()

    # =========================================================
    # 关键修改：
    # Excel 中的 Dice 现在是：
    #     0.8880±0.0084
    #     0.9502 ± 0.0052
    #     0.8880
    #
    # 热图只截取“平均值”，完全忽略 ± 后面的标准差。
    # 因此：
    #     0.8880±0.0084 -> 0.8880
    #     0.9502±0.0052 -> 0.9502
    #
    # 行列顺序、颜色、热图布局等其余逻辑全部保持不变。
    # =========================================================
    def extract_mean_dice(value):
        if pd.isna(value):
            return np.nan

        # 已经是数值
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) if np.isfinite(value) else np.nan

        s = str(value).strip()

        # 优先截取 ± 前面的平均值
        # 同时兼容：
        #   0.8880±0.0084
        #   0.8880 ± 0.0084
        #   0.8880+/-0.0084
        if '±' in s:
            s = s.split('±', 1)[0].strip()
        elif '+/-' in s:
            s = s.split('+/-', 1)[0].strip()

        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    filtered_df['test/dice'] = filtered_df['test/dice'].apply(
        extract_mean_dice
    )

    filtered_df = filtered_df.dropna(
        subset=['baseline', 'dataset', 'test/dice']
    )

    dataset_order = [
        'OriginalData',
        'Dataset_organoids',
        'MouseOrganoids',
        'OrganoidBasic',
        'stomach_intance'
    ]

    model_order = [
        'GSCNN',
        'U-Net',
        'SegFormer',
        'UNet++',
        'TransUNet',
        'DeepLabv3+',
        'MEGANet',
        'SAM2',
        'Cellpose-SAM',
        'DINOV2',

    ]

    # =========================================================
    # 关键修改：
    # 同一个 baseline + dataset 如果有多条记录，
    # 先计算 test/dice 的平均值。
    # =========================================================
    mean_df = (
        filtered_df
        .groupby(['baseline', 'dataset'], as_index=False)['test/dice']
        .mean()
    )

    # pivot_table 使用 mean 也可以处理重复记录；
    # 这里 mean_df 已经提前聚合，因此结果更加明确。
    pivot_df = mean_df.pivot(
        index='baseline',
        columns='dataset',
        values='test/dice'
    )

    # 按论文表格固定顺序排列
    pivot_df = pivot_df.reindex(
        index=model_order,
        columns=dataset_order
    )

    # 热图：dataset 为行，model 为列
    plot_df = pivot_df.T

    # 检查是否存在完全没有数据的情况
    if plot_df.dropna(how='all').empty:
        raise ValueError(
            "没有可用于绘制热图的 test/dice 数据，请检查 Excel 中 "
            "'baseline'、'dataset'、'test/dice' 三列。"
        )

    # =========================================================
    # 颜色范围只根据实际存在的平均 Dice 值计算，
    # 不受 NaN 影响。
    # =========================================================
    valid_values = plot_df.to_numpy(dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]

    if valid_values.size == 0:
        raise ValueError("test/dice 中没有有效的数值。")

    data_min = valid_values.min()
    data_max = valid_values.max()

    vmin = np.floor(data_min * 20) / 20
    vmax = np.ceil(data_max * 20) / 20

    # 防止所有平均值完全相同导致 Normalize 范围为 0
    if vmax <= vmin:
        vmin = data_min - 0.01
        vmax = data_max + 0.01

    colors = ["#6BAED6", "#FFFFFF", "#D6619C"]
    custom_cmap = LinearSegmentedColormap.from_list(
        "pink_blue", colors
    )

    print("\n========== Heatmap data (mean Dice) ==========")
    print(mean_df.to_string(index=False))
    print("\n========== Heatmap matrix ==========")
    print(plot_df)

    return plot_df, vmin, vmax, custom_cmap


def plot_heatmap_on_ax(ax, plot_df, vmin, vmax, custom_cmap, show_colorbar=True, shrink=0.5):
    rows, cols = plot_df.shape
    gap = 0.08
    box_width = 1 - gap
    box_height = 1 - gap
    # if show_colorbar:
    #     sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #     # 这里的 shrink=shrink 是关键，aspect=30 可以让颜色条变细长，更协调
    #     cbar = ax.figure.colorbar(sm, ax=ax, shrink=shrink, pad=0.04, aspect=30)

    for i in range(rows):
        for j in range(cols):
            val = plot_df.iloc[i, j]
            color = custom_cmap((val - vmin) / (vmax - vmin))

            rect = patches.FancyBboxPatch(
                (j + gap / 2, i + gap / 2),
                box_width, box_height,
                boxstyle="round,pad=0,rounding_size=0.15",
                linewidth=0,
                edgecolor='none',
                facecolor=color,
                zorder=2
            )
            ax.add_patch(rect)

            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_col = 'black'
            # text_col = 'white' if norm_val > 0.8 or norm_val < 0.2 else 'black'

            ax.text(j + 0.5, i + 0.5, f'{val:.4f}',
                    ha='center', va='center',
                    weight='bold', fontsize=6, color=text_col, zorder=3)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('auto')

    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_xticklabels(plot_df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(plot_df.index, fontsize=8)

    ax.spines[:].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.03, aspect=20)
        cbar.set_label('Dice Score')
        cbar.outline.set_visible(False)


if __name__ == "__main__":
    file_path_new = "/home/liudenghui/baseline/baseline3/实验数据.xlsx"
    plot_df, vmin, vmax, custom_cmap = prepare_heatmap_dataframe(file_path_new)

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_heatmap_on_ax(ax, plot_df, vmin, vmax, custom_cmap, show_colorbar=True)

    plt.tight_layout()
    plt.savefig("segmentation_heatmap_panel_style.png", dpi=600, bbox_inches='tight')
    plt.show()