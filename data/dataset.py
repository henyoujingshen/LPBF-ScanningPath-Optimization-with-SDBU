# coding:utf-8
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from data.feature_engineering import feature_engineering

class DepoTempDataset(Dataset):
    """
    用于处理Depo和Temp数据的Dataset
    """
    def __init__(self, data_dir, train=True, test=False):
        """
        :param data_dir: 数据集所在目录
        :param train: 是否为训练集
        :param test: 是否为测试集
        """
        self.test = test
        self.data_info = self._get_data_info(data_dir, test)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        path_depo, path_temp = self.data_info[idx]
        temp = self._load_csv(path_temp)
        depo = self._load_csv(path_depo)

        depo = np.floor(depo)
        temp = temp - 473

        # 特征工程区块（如有需要可启用）
        # features = feature_engineering(depo)
        # d = features.distance_matrix(1)
        # t_hiz = features.t_hiz()
        # input = np.concatenate((depo[np.newaxis, :, :], d[np.newaxis, :, :], t_hiz[np.newaxis, :, :]), axis=0)
        # input = input.astype(np.float32)
        # return input, temp

        return depo, temp

    def _get_data_info(self, data_dir, test):
        """
        读取数据索引文件，返回文件路径对列表
        """
        txt_name = 'root_data30.txt' if test else 'root_full_data_server1.txt'
        path_dir = os.path.join(data_dir, txt_name)
        data_info = []
        with open(path_dir) as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i % 100 == 99:
                    line = line.strip()
                    data_info.append([line[:92].strip(), line[92:]])
                    # 调试用 print(line[93:])，这里注意如果你进行了100次以上的扫描，请改成93看一些名称是否正确
        return data_info

    def _load_csv(self, path):
        """
        加载csv文件为np.float32数组
        """
        data_read = pd.read_csv(path, header=None)
        data = data_read.values.astype(np.float32)
        data = np.expand_dims(data, axis=0)  # 保持和原逻辑一致
        return data

class ValidDataset(Dataset):
    """
    用于处理验证集数据的Dataset
    """
    def __init__(self, data_dir, train=True, test=False):
        self.test = test
        self.data_info = self._get_data_info(data_dir, test)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        path_depo, path_temp = self.data_info[idx]
        temp = self._load_csv(path_temp)
        depo = self._load_csv(path_depo)

        depo = np.floor(depo)
        temp = temp - 473

        # 特征工程区块（如有需要可启用）
        # features = feature_engineering(depo)
        # d = features.distance_matrix(1)
        # t_hiz = features.t_hiz()
        # input = np.concatenate((depo[np.newaxis, :, :], d[np.newaxis, :, :], t_hiz[np.newaxis, :, :]), axis=0)
        # input = input.astype(np.float32)
        # return input, temp

        return depo, temp

    def _get_data_info(self, data_dir, test):
        txt_name = 'root_data30.txt' if test else 'root_full_data_server.txt'
        path_dir = os.path.join(data_dir, txt_name)
        data_info = []
        with open(path_dir) as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i % 49 == 1:
                    line = line.strip()
                    data_info.append([line[:93].strip(), line[93:]])
                    # 调试用 print(line[92:])
        return data_info

    def _load_csv(self, path):
        data_read = pd.read_csv(path, header=None)
        data = data_read.values.astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data
