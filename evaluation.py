import numpy as np
import torch # 导入torch模块
import models
from data.feature_engineering import feature_engineering
class evaluator: # 定义ML_modeling类
    def __init__(self,array,model_name): # 定义初始化方法
        self.array = array
        self.model = getattr(models, model_name)()
        #cuda_device = torch.device('cuda:0')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(r"C:\Users\lenovo\Desktop\graduate_paper\DNN_RNN\checkpoints\unet_model_140_epoch_pure5050", map_location=device))
        #model = torch.load(model_name)

    def to_depomatrix(self):
        ##island模式
        index=self.array
        nsteps_depo=np.zeros((len(index)*len(index),50, 50))
        depo = np.zeros((50, 50))
        n_tracks = 10
        for i in range(len(index)*5):
            row, col = np.where(index == i + 1)
            x_offset = (col[0]) * 10
            y_offset = (row[0] + 1) * 10
            if (row[0] + col[0]) % 2 == 0:
                Y_y = y_offset
                for k in range(n_tracks):
                    Y_y -= 1
                    if k % 2 == 0:
                        x_track = x_offset
                        for kk in range(n_tracks):
                            depo[Y_y, x_track] = depo[Y_y, x_track] + 0.000001
                            x_track += 1
                            depo[depo != 0] += 1
                    else:
                        x_track = x_offset + n_tracks - 1
                        for kk in range(n_tracks):
                            depo[Y_y, x_track] = depo[Y_y, x_track] + 0.000001
                            depo[depo != 0] += 1
                            x_track -= 1
            else:
                X_x = x_offset
                for k in range(n_tracks):
                    if k % 2 == 0:
                        y_track = y_offset
                        for kk in range(n_tracks):
                            y_track -= 1
                            depo[y_track, X_x] = depo[y_track, X_x] + 0.000001
                            depo[depo != 0] += 1

                    else:
                        y_track = y_offset - n_tracks
                        for kk in range(n_tracks):
                            depo[y_track, X_x] = depo[y_track, X_x] + 0.000001
                            depo[depo != 0] += 1
                            y_track += 1
                    X_x += 1
            #特征工程部分
            # features = feature_engineering(depo)
            # d = features.distance_matrix(1)
            # t_hiz = features.t_hiz()
            # nsteps_depo[i, 0, :] = depo
            # nsteps_depo[i, 1, :] = d
            # nsteps_depo[i, 2, :] = t_hiz
            nsteps_depo[i]=depo
        # stripe模式
        # depo = np.zeros((50, 50))
        # t = 0
        # n_tracks = 50
        # n = 50
        # x_left = 0
        # x_right = 49
        # nsteps_depo = np.zeros((len(index) ,3, 50, 50))
        #
        # for i in range(n):
        #     row = index[i]
        #     y_offset = (50 - row)
        #     if row % 2 == 1:
        #         x_track = x_left
        #         for k in range(n_tracks):
        #             t += 1
        #             depo[y_offset, x_track] = 0.000001
        #             depo[depo != 0] += 1
        #             x_track += 1
        #     else:
        #         x_track = x_right
        #         for k in range(n_tracks):
        #             t += 1
        #             depo[y_offset, x_track] = 0.000001
        #             depo[depo != 0] += 1
        #             x_track -= 1
        #
        #     features = feature_engineering(depo)
        #     d = features.distance_matrix(1)
        #     t_hiz = features.t_hiz()
        #     nsteps_depo[i, 0, :] = depo
        #     nsteps_depo[i, 1, :] = d
        #     nsteps_depo[i, 2, :] = t_hiz

        return nsteps_depo

    def calculate_output(self): # 定义calculate_output函数
        nsteps_depo1 = self.to_depomatrix() # 调用to_depomatrix函数得到二维矩阵

        #
        nsteps_depo = torch.from_numpy(nsteps_depo1).to(torch.float32)
        nsteps_depo=nsteps_depo.view((25, 1, 50, 50))
        b=self.model
        temperature_matrix = self.model(nsteps_depo)
        nsteps_temp = torch.split(temperature_matrix, 1, dim=0)
        output_total = np.zeros(25)
        for n in range(25):
            temperature_matrix=nsteps_temp[n]+473
            temperature_matrix=temperature_matrix.detach().numpy()
            # 计算平均温度
            T_avg = np.mean(temperature_matrix)
            # 计算(T(i,j) - T_avg)的平方和
            diff_squared_sum = np.sum(np.square(temperature_matrix - T_avg))
            # 计算标准差
            output_n = np.sqrt(diff_squared_sum / (np.size(temperature_matrix)*T_avg ** 2))
            output_total[n]=output_n
        output=np.sum(output_total)
        return output

