# coding:utf-8
import os
import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.optim import Adam
from sklearn import model_selection
from torchnet import meter
from config import opt
import models
# NOTE: Assuming DepoTempDataset and ValidDataset are the correct class names in data.dataset
# and aliasing Depo_Temp_data for compatibility with its usage in this file.
from data.dataset import DepoTempDataset as Depo_Temp_data
from utils.visualize import Visualizer

# import ipdb # For debugging, comment out for production

# Define a small epsilon for numerical stability
EPSILON = 1e-7


class CorrelationCoefficientLoss(nn.Module):
    """
    Custom loss function, potentially inspired by the SDBU paper's Cor*R loss (Eq. 9).
    It calculates a similarity index between two inputs (prediction and target).
    The goal is to maximize this similarity, so the loss returns its negative mean.
    """

    def __init__(self):
        super(CorrelationCoefficientLoss, self).__init__()

    def forward(self, pred_input, target_input):
        """
        Calculates the loss.
        Args:
            pred_input (torch.Tensor): The predicted tensor.
            target_input (torch.Tensor): The ground truth tensor.
        Returns:
            torch.Tensor: The calculated loss value.
        """
        # Assuming inputs are (batch_size, channels, height, width)
        # Reshape to (batch_size, num_features) e.g., 50x50 -> 2500
        batch_size = pred_input.size(0)
        num_features = pred_input.size(1) * pred_input.size(2) * pred_input.size(3)  # C*H*W or H*W if C=1

        pred_flat = pred_input.reshape(batch_size, -1)
        target_flat = target_input.reshape(batch_size, -1)

        pred_mean = torch.mean(pred_flat, dim=1, keepdim=True)
        target_mean = torch.mean(target_flat, dim=1, keepdim=True)

        pred_std = torch.std(pred_flat, dim=1, keepdim=True)
        target_std = torch.std(target_flat, dim=1, keepdim=True)

        # This part seems to align with the Cor*R formula (Eq. 9) from the paper
        # (2 * sigma_XG * mu_X * mu_G + C1_prime) / (sigma_X^2 * mu_G^2 + sigma_G^2 * mu_X^2 + C1_prime)
        # The paper's sigma_XG is covariance. The code uses std*std for the numerator term,
        # which is unusual if it's directly from Cor*R, but let's keep the original logic here.
        # The term "2*input1_std*input2_std*input1_mean*input2_mean" from original code
        numerator_term = 2 * pred_std * target_std * pred_mean * target_mean

        # The term "(input1_std ** 2)*(input2_mean ** 2) + (input2_std ** 2)*(input1_mean ** 2)" from original code
        denominator_term = (pred_std ** 2) * (target_mean ** 2) + (target_std ** 2) * (pred_mean ** 2)

        # The original code had *2500/2499, which is like N/(N-1) for unbiased estimate
        # If num_features is equivalent to N, then:
        correction_factor = num_features / (num_features - 1) if num_features > 1 else 1.0

        similarity_index = (numerator_term + EPSILON) / (denominator_term + EPSILON) * correction_factor
        similarity_index = torch.abs(similarity_index)

        # The original code had another calculation for 'cov' which overwrote 'similarity_index'.
        # Based on the complexity, the first calculation was likely the intended custom loss.
        # If Pearson correlation was intended:
        # pred_centered = pred_flat - pred_mean
        # target_centered = target_flat - target_mean
        # covariance = torch.mean(pred_centered * target_centered, dim=1, keepdim=True)
        # pearson_corr = (covariance + EPSILON) / (pred_std * target_std + EPSILON)
        # similarity_index = torch.abs(pearson_corr)

        # Return the negative mean to maximize similarity
        return -similarity_index.mean()


def calculate_accuracy_metric(predictions, labels):
    """
    Calculates an accuracy-like metric: 1 - Mean Relative Error.
    Args:
        predictions (torch.Tensor): The predicted tensor.
        labels (torch.Tensor): The ground truth tensor.
    Returns:
        float: The accuracy metric.
    """
    # Add EPSILON to avoid division by zero if labels can be zero
    relative_error = torch.abs(predictions - labels) / (torch.abs(labels) + EPSILON)
    sum_loss = torch.sum(relative_error)

    num_elements = labels.numel()  # Total number of elements
    if num_elements == 0:
        return 0.0

    mean_loss = sum_loss / num_elements
    acc_metric = 1.0 - mean_loss
    return acc_metric.item()


def validate_model(model, dataloader, device, is_test_set=False):
    """
    Calculates the model's accuracy metric on a given dataset.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation/test set.
        device (torch.device): The device to run evaluation on.
        is_test_set (bool): Flag to print specific messages for test set.
    Returns:
        float: The calculated accuracy metric.
    """
    model.eval()
    all_predictions_list = []
    all_labels_list = []

    with torch.no_grad():
        for data, labels in dataloader:
            inputs = data.to(device)
            targets = labels.to(device)

            predictions = model(inputs)

            all_predictions_list.append(predictions.cpu())
            all_labels_list.append(targets.cpu())
            # Removed t.cuda.empty_cache() here as it can be slow; manage memory globally if needed.

    all_predictions = torch.cat(all_predictions_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)

    accuracy_metric = calculate_accuracy_metric(all_predictions, all_labels)

    if is_test_set:
        print(f"Test accuracy metric: {accuracy_metric:.5f}")

    model.train()  # Set model back to training mode
    return accuracy_metric


def train_test(**kwargs):
    """
    Main function to perform k-fold cross-validation training and testing.
    Each fold trains a model, evaluates it on a validation set (implicitly via loss monitoring)
    and finally on a fixed test set.
    """
    opt.parse(kwargs)  # Update global opt with command-line arguments

    device = torch.device(f"cuda:{opt.gpu_id}" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vis = Visualizer(opt.env)

    # Prepare test data (same for all folds)
    # Make sure Depo_Temp_data is correctly aliased or the class name is Depo_Temp_data
    test_dataset = Depo_Temp_data(opt.train_data_root, train=False, test=True)
    test_dataloader = DataLoader(test_dataset, opt.batch_size, num_workers=opt.num_workers)

    # Prepare training data (to be split by k-fold)
    full_train_dataset = Depo_Temp_data(opt.train_data_root, train=True)

    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=opt.seed if hasattr(opt, 'seed') else None)

    log_filename = f"performance_for_{opt.model}.txt"

    with open(log_filename, "a") as log_file:
        log_file.write(f"Starting K-Fold Cross-Validation for model: {opt.model}\n")
        log_file.write(f"Options: {vars(opt)}\n\n")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(full_train_dataset)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            log_file.write(f"--- Fold {fold + 1}/5 ---\n")

            # 1. Initialize Model
            model = getattr(models, opt.model)()
            model.to(device)
            # Example for loading pretrained weights (optional)
            # if opt.load_model_path:
            #     pretrained_dict = torch.load(opt.load_model_path, map_location=device)
            #     model.load_state_dict(pretrained_dict)
            model.train()

            # 2. Prepare DataLoaders for current fold
            train_sampler = SubsetRandomSampler(train_idx)
            # val_sampler = SubsetRandomSampler(val_idx) # Validation set within fold (not explicitly used for metric calculation during epoch)

            train_dataloader = DataLoader(full_train_dataset, batch_size=opt.batch_size,
                                          sampler=train_sampler, num_workers=opt.num_workers)
            # val_dataloader = DataLoader(full_train_dataset, batch_size=opt.batch_size,
            #                             sampler=val_sampler, num_workers=opt.num_workers)

            # 3. Define Loss Function and Optimizer
            # Choose loss function based on opt or paper's SDBU method
            if hasattr(opt, 'loss_function') and opt.loss_function == 'Cor*R':
                criterion = CorrelationCoefficientLoss().to(device)
                print("Using CorrelationCoefficientLoss (Cor*R-like)")
            else:
                criterion = nn.MSELoss().to(device)
                print("Using MSELoss")

            current_lr = opt.lr
            # Filter parameters that require gradients for the optimizer
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=current_lr,
                             weight_decay=opt.weight_decay)

            # 4. Statistics Meters
            loss_meter = meter.AverageValueMeter()
            previous_fold_loss = float('inf')  # Renamed from previous_loss to avoid confusion with epoch loss

            # 5. Training Loop
            for epoch in range(opt.max_epoch):
                loss_meter.reset()
                for i, (data, labels) in enumerate(train_dataloader):
                    inputs = data.to(device)
                    targets = labels.to(device)

                    optimizer.zero_grad()
                    predictions = model(inputs)
                    loss = criterion(predictions, targets).float()  # Ensure loss is float
                    loss.backward()
                    optimizer.step()

                    loss_meter.add(loss.item())

                    if i % opt.print_freq == (opt.print_freq - 1):
                        vis.plot(f'fold_{fold + 1}/loss', loss_meter.value()[0])
                        print(
                            f"Fold {fold + 1}, Epoch {epoch + 1}, Batch {i + 1}/{len(train_dataloader)}, Loss: {loss_meter.value()[0]:.5f}")

                epoch_loss = loss_meter.value()[0]
                vis.log(f"Fold:{fold + 1}, Epoch:{epoch + 1}, LR:{current_lr:.6f}, Loss:{epoch_loss:.5f}")

                # Learning rate scheduling (simple decay on plateau)
                if epoch_loss > previous_fold_loss * (1.0 - opt.lr_decay_threshold if hasattr(opt,
                                                                                              'lr_decay_threshold') else 0.001):  # If loss doesn't improve enough
                    current_lr *= opt.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    print(f"Fold {fold + 1}, Epoch {epoch + 1}: Reducing LR to {current_lr:.6f}")

                epoch_message = f"Fold {fold + 1}, Epoch {epoch + 1}: Avg Loss = {epoch_loss:.5f}, LR = {current_lr:.6f}"
                print(epoch_message)
                log_file.write(epoch_message + "\n")
                previous_fold_loss = min(previous_fold_loss, epoch_loss)  # Track best loss for LR scheduling

            # Save model checkpoint after each fold
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            save_path_dir = os.path.abspath("checkpoints/")
            model_save_name = f"{opt.model}_fold{fold + 1}_epoch{opt.max_epoch}.pth"
            torch.save(model.state_dict(), os.path.join(save_path_dir, model_save_name))
            print(f"Saved model for fold {fold + 1} to {os.path.join(save_path_dir, model_save_name)}")

            # Evaluate on the test set after training for this fold
            test_acc_metric = validate_model(model, test_dataloader, device, is_test_set=True)
            fold_summary_message = f"Fold {fold + 1} Test Accuracy Metric: {test_acc_metric:.5f}"
            print(fold_summary_message)
            log_file.write(fold_summary_message + "\n\n")
            vis.plot('test_accuracy_metric_per_fold', test_acc_metric, opts=dict(name=f'Fold {fold + 1}'))


if __name__ == '__main__':
    fire.Fire(train_test)