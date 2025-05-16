#coding:utf8
import torch as t
import time
import os
class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        save_path = os.path.abspath("C:/Users/lenovo/Desktop/graduate_paper/DNN_RNN/checkpoints/")
        if name is None:
            prefix ='modelat_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pt')
        #import ipdb
        #ipdb.set_trace()
        t.save(self.state_dict(), os.path.join(save_path, name))
        #t.save(self.state_dict(), name)
        return name


    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
