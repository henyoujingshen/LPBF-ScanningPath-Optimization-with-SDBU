# coding:utf8
import os
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import ipdb
from builtins import float
class feature_engineering(object):
    def __init__(self,depo):
        self.depo=depo

    def distance_matrix(self, value):
        #使用向量化操作计算每一个位置与 (row_index, col_index) 之间的欧几里得距离，并将计算结果存储在矩阵中。
        matrix=self.depo
        matrix=np.floor(matrix)
        row_indices, col_indices = np.where(matrix == value)
        if len(row_indices) > 0:
            row_index, col_index = row_indices[0], col_indices[0]
            distances = np.zeros((matrix.shape[1], matrix.shape[1]))
            rows, cols = np.indices((matrix.shape[1], matrix.shape[1]))
            distances = np.sqrt((rows - row_index) ** 2 + (cols - col_index) ** 2)
        else:
            distances=np.full((matrix.shape[1], matrix.shape[1]), np.inf)

        return distances

    def t_hiz(self):
        a = self.depo
        a = np.floor(a)
        a[a == 0] = 1000000
        #a=a-1;
        # 初始化b为a的副本
        b = np.copy(a)

        # 获取a的行数和列数
        rows, cols = a.shape

        # 遍历矩阵的每一个元素
        for i in range(rows):
            for j in range(cols):
                # 获取周围元素的索引范围，考虑边界情况
                min_row = max(i - 1, 0)
                max_row = min(i + 1, rows - 1)
                min_col = max(j - 1, 0)
                max_col = min(j + 1, cols - 1)

                # 提取周围元素并找到最小值
                surrounding_elements = a[min_row:max_row + 1, min_col:max_col + 1]
                b[i, j] = np.min(surrounding_elements)
        b[b == 1000000] = 0
        return b

    def t_n(self):
        ##可以填充你想加入的其他特征工程，要改之后的维度，！！！
        b=0
        return b