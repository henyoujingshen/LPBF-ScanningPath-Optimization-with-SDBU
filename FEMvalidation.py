from config import opt
import os
import numpy as np
import torch as t
import models
from data.dataset import Valid_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn

# =========================
# #直观可视化观察模型训练结果，在测试样例上的效果
# =========================

# =========================
# 1. 数据准备与模型加载
# =========================

# 数据集与dataloader
test_data = Valid_data('./SLM', train=False, test=True)
test_dataloader = DataLoader(test_data, batch_size=50, num_workers=0, shuffle=False)

# 加载模型
model = getattr(models, 'unet')()
model_path = r"C:\Users\lenovo\Desktop\graduate_paper\DNN_RNN\checkpoints\unet_model_5050_stripe_mse_half"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model.load_state_dict(t.load(model_path, map_location=device))
model.eval()

# =========================
# 2. 自定义相关性损失
# =========================

class CorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(CorrelationCoefficientLoss, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.reshape(input1.size(0), -1)
        input2 = input2.reshape(input2.size(0), -1)
        input1_mean = t.mean(input1, dim=1, keepdim=True)
        input2_mean = t.mean(input2, dim=1, keepdim=True)
        input1_centered = input1 - input1_mean
        input2_centered = input2 - input2_mean

        covariance = t.mean(input1_centered * input2_centered, dim=1, keepdim=True)
        input1_std = t.std(input1, dim=1, keepdim=True)
        input2_std = t.std(input2, dim=1, keepdim=True)

        denominator = (input1_std ** 2) * (input2_mean ** 2) + (input2_std ** 2) * (input1_mean ** 2)
        similarity_index = (2 * covariance * input1_mean * input2_mean + 1e-7) / (denominator + 1e-7) * input1.size(1) / (input1.size(1) - 1)
        similarity_index = abs(similarity_index)

        std_diff = abs((2 * input1_std * input2_std + 1e-7) / ((input1_std ** 2) + (input2_std ** 2) + 1e-7))
        R_diff = abs((2 * input1_std * input2_std * input1_mean * input2_mean + 1e-7) / (denominator + 1e-7))
        cov = abs((covariance + 1e-7) / (input1_std * input2_std + 1e-7))

        return std_diff.mean(), R_diff.mean(), similarity_index.mean()

# =========================
# 3. 精度评估函数
# =========================

def acc(predict, label):
    sum_loss = t.sum(t.abs(predict - label) / (t.abs(label) + 1e-7))
    mean_loss = sum_loss / (label.shape[0] * label.shape[1])
    accuracy = 1 - mean_loss
    return accuracy.item()

# =========================
# 4. 模型推理与可视化
# =========================

with t.no_grad():
    all_prediction = t.empty((0, 1, 50, 50))
    all_val_label = t.empty((0, 1, 50, 50))

    for val_input, val_label in test_dataloader:
        prediction = model(val_input)
        all_prediction = t.cat((all_prediction, prediction), dim=0)
        all_val_label = t.cat((all_val_label, val_label), dim=0)

# =========================
# 5. 逐步可视化与指标计算
# =========================

all_prediction = t.squeeze(all_prediction)
nsteps_prediction = t.split(all_prediction, 1, dim=0)
all_label = t.squeeze(all_val_label)
nsteps_label = t.split(all_label, 1, dim=0)

acc_arr = np.zeros(50, dtype=float)
loss_arr = np.zeros(50, dtype=float)
ssim_arr = np.zeros((50, 3), dtype=float)

for n in range(2):  # 如需全量可改为range(50)
    pred_matrix = nsteps_prediction[n].squeeze()
    gt_matrix = nsteps_label[n].squeeze()

    acc_arr[n] = acc(pred_matrix, gt_matrix)
    pred_np = pred_matrix.detach().numpy()
    gt_np = gt_matrix.detach().numpy()

    vmin_pred, vmax_pred = np.min(pred_np), np.max(pred_np)
    vmin_gt, vmax_gt = np.min(gt_np), np.max(gt_np)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(pred_np, cmap='viridis', interpolation='nearest', vmin=vmin_pred, vmax=vmax_pred)
    axs[0].set_title(f"Control Period:{n} Prediction")
    fig.colorbar(axs[0].images[0], ax=axs[0], orientation='vertical')

    axs[1].imshow(gt_np, cmap='viridis', interpolation='nearest', vmin=vmin_gt, vmax=vmax_gt)
    axs[1].set_title(f"Control Period:{n} Ground Truth")
    fig.colorbar(axs[1].images[0], ax=axs[1], orientation='vertical')

    pred_tensor = Variable(pred_matrix.float().unsqueeze(0), requires_grad=False)
    gt_tensor = Variable(gt_matrix.float().unsqueeze(0), requires_grad=False)
    ssim_loss = CorrelationCoefficientLoss()
    ssim_arr[n, :] = ssim_loss(pred_tensor, gt_tensor)

    mse_loss = nn.MSELoss()
    loss_arr[n] = mse_loss(pred_tensor, gt_tensor).item()

    main_title = f"Std_diff={ssim_arr[n, 0]:.5f}, R_diff={ssim_arr[n, 1]:.5f}, Cov={ssim_arr[n, 2]:.5f}, MSE={loss_arr[n]:.5f}"
    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    plt.show()


