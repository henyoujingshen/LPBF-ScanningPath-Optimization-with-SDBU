import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. 条带性打印（50x50矩阵）
# =============================

def scan_square(strategy, ax):
    """
    可视化扫描顺序，按策略索引为每一行赋色。
    :param strategy: 行索引顺序（长度为50的列表或数组，取值1-50）
    :param ax: matplotlib 子图对象
    :return: imshow对象
    """
    square = np.zeros((50, 50))
    for i, row in enumerate(strategy):
        square[row - 1, :] = i + 1
    im = ax.imshow(square, cmap='summer', interpolation='nearest')
    ax.axis('off')
    return im

# 各种扫描策略索引
index_strategies = [
    np.arange(1, 51, 1),  # 顺序
    [1, 50, 2, 49, 3, 48, 4, 47, 5, 46, 6, 45, 7, 44, 8, 43, 9, 42, 10, 41, 11, 40, 12, 39, 13, 38, 14, 37, 15, 36, 16, 35, 17, 34, 18, 33, 19, 32, 20, 31, 21, 30, 22, 29, 23, 28, 24, 27, 25, 26],
    [7, 14, 35, 28, 36, 17, 37, 21, 1, 15, 19, 40, 41, 29, 3, 5, 31, 10, 42, 44, 22, 46, 26, 8, 47, 12, 4, 32, 9, 39, 25, 23, 16, 48, 27, 2, 45, 13, 33, 50, 24, 49, 30, 6, 38, 18, 43, 34],
    [45, 16, 29, 36, 4, 18, 42, 12, 1, 47, 24, 33, 13, 35, 44, 7, 23, 9, 43, 2, 38, 25, 17, 26, 49, 40, 8, 37, 28, 6, 19, 46, 41, 50, 10, 3, 11, 27, 5, 20, 32, 48, 31, 39, 15, 22, 21, 30, 34, 14],
    [37, 50, 26, 42, 11, 3, 4, 21, 12, 31, 47, 30, 19, 46, 45, 9, 34, 24, 8, 16, 38, 2, 13, 48, 41, 18, 7, 28, 44, 39, 15, 35, 22, 1, 36, 5, 20, 40, 6, 32, 10, 49, 43, 33, 23, 17, 29, 14, 25, 27]
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={'wspace': -0.3})

for i, strategy in enumerate(index_strategies):
    im = scan_square(strategy, axes[i])

# 添加 colorbar 并标记 start/end
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08, aspect=100)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Scan Order', fontsize=14)
cbar.ax.text(-0.05, -1.5, 'start', ha='center', va='center', fontsize=14, transform=cbar.ax.transAxes)
cbar.ax.text(1.05, -1.5, 'end', ha='center', va='center', fontsize=14, transform=cbar.ax.transAxes)

plt.show()

# =============================
# 2. 岛屿打印（5x5矩阵，注释掉，如需用可取消注释）
# =============================

# def scan_square(strategy, ax):
#     """
#     可视化5x5岛屿打印策略。
#     :param strategy: 5x5矩阵
#     :param ax: matplotlib 子图对象
#     :return: imshow对象
#     """
#     im = ax.imshow(strategy, cmap='summer', interpolation='nearest')
#     ax.axis('off')
#     return im
#
# # 五种岛屿策略
# island_strategies = [
#     np.arange(1, 26).reshape(5, 5),
#     np.array([[1, 3, 17, 10, 8], [5, 15, 22, 19, 12], [7, 20, 25, 24, 14], [13, 18, 23, 21, 6], [9, 11, 16, 4, 2]]),
#     np.array([[24, 17, 3, 7, 16], [14, 20, 19, 21, 15], [22, 13, 2, 1, 23], [5, 6, 12, 25, 11], [8, 4, 9, 18, 10]]),
#     np.array([[21, 10, 3, 23, 8], [6, 22, 25, 11, 16], [18, 4, 20, 5, 1], [15, 12, 24, 19, 17], [9, 14, 7, 13, 2]]),
#     np.array([[9, 11, 15, 6, 19], [21, 2, 23, 17, 4], [10, 22, 8, 20, 25], [1, 18, 16, 5, 12], [7, 13, 24, 14, 3]])
# ]
#
# fig, axes = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={'wspace': -0.3})
# for i, strategy in enumerate(island_strategies):
#     im = scan_square(strategy, axes[i])
#
# cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08, aspect=100)
# cbar.ax.tick_params(labelsize=14)
# cbar.set_label('Scan Order', fontsize=14)
# cbar.ax.text(-0.05, 1.5, 'start', ha='center', va='center', fontsize=14, transform=cbar.ax.transAxes)
# cbar.ax.text(1.05, 1.5, 'end', ha='center', va='center', fontsize=14, transform=cbar.ax.transAxes)
# plt.show()
