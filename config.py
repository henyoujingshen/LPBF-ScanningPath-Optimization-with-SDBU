# coding:utf-8
import warnings
import torch


class DefaultConfig:
    """
    Default configuration settings for the training and testing pipeline.

    Attributes can be overridden by command-line arguments passed to the main script
    if the main script uses `opt.parse(kwargs)`.
    """
    # --- Visdom Environment ---
    env: str = 'default'  # Visdom environment name for plotting
    vis_port: int = 8097  # Visdom server port

    # --- Model Configuration ---
    model: str = 'unet'  # Name of the model to use (must match a class in models/__init__.py or models.<model_name>.py)
    load_model_path: str = None  # Path to a pre-trained model to load (if None, model is initialized from scratch)
    # Example: load_model_path = './checkpoints/unet_fold1_epoch100.pth'

    # --- Data Configuration ---
    train_data_root: str = './data/processed_slm_data/'  # Root directory for training and testing data
    # test_data_root: str = './data/processed_slm_data/' # Typically, test data might be in a subfolder or identified by a different root file
    num_workers: int = 4  # Number of worker processes for DataLoader (adjust based on CPU cores)

    # --- Training Hyperparameters ---
    batch_size: int = 16  # Batch size for training
    test_batch_size: int = 32  # Batch size for testing/validation
    max_epoch: int = 200  # Maximum number of training epochs
    lr: float = 0.001  # Initial learning rate
    lr_decay: float = 0.5  # Factor to decay learning rate by (e.g., when validation loss plateaus)
    lr_decay_threshold: float = 0.005  # Minimum relative improvement required to not decay LR
    weight_decay: float = 1e-5  # Weight decay (L2 penalty) for the optimizer

    # --- GPU Configuration ---
    use_gpu: bool = True  # Whether to use GPU if available
    gpu_id: int = 0  # ID of the GPU to use if use_gpu is True

    # --- Logging and Debugging ---
    print_freq: int = 50  # Frequency (in batches) to print training information
    # debug_file: str = '/tmp/debug_slm' # If this file exists, ipdb debugger might be triggered (if implemented in main script)
    seed: int = 42  # Random seed for reproducibility

    # --- Loss Function ---
    loss_function: str = 'MSE'  # Options: 'MSE', 'Cor*R' (for CorrelationCoefficientLoss)

    # --- Device (will be set by parse method) ---
    device: torch.device = None

    def __init__(self):
        """
        Initializes the configuration.
        The `device` attribute will be set after parsing command-line arguments.
        """
        pass

    def parse(self, kwargs: dict) -> None:
        """
        Updates configuration attributes based on a dictionary of keyword arguments.
        This is typically used to override default settings with command-line arguments.

        Args:
            kwargs (dict): Dictionary of attribute_name: value pairs.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn(f"Warning: '{self.__class__.__name__}' has no attribute '{key}'. "
                              f"It will be added, but this might indicate a typo or unexpected parameter.")
            setattr(self, key, value)

        # Set the device based on GPU availability and configuration
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            if self.use_gpu and not torch.cuda.is_available():
                warnings.warn("Warning: 'use_gpu' is True, but CUDA is not available. Using CPU instead.")
            self.device = torch.device('cpu')

        self._print_config()

    def _print_config(self) -> None:
        """Prints the current configuration settings."""
        print('User Config:')
        # Iterate over attributes of the instance, not the class, to get current values
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Exclude private attributes like _print_config
                print(f"  {key}: {value}")
        print("-" * 30)


# Create a global instance of the configuration
opt = DefaultConfig()

if __name__ == '__main__':
    # Example of how to use and parse arguments
    print("--- Initial Default Config ---")
    opt._print_config()

    # Simulate command-line arguments
    cli_args = {
        'lr': 0.005,
        'batch_size': 32,
        'model': 'ResNetFCN',
        'env': 'experiment_001',
        'use_gpu': False,  # Test overriding to CPU
        'new_param': 'test_value'  # Test adding a new parameter
    }

    print("\n--- Config After Parsing Simulated CLI Args ---")
    opt.parse(cli_args)

    # Accessing a config value
    print(f"\nLearning rate from opt: {opt.lr}")
    print(f"Device from opt: {opt.device}")
    print(f"New param from opt: {opt.new_param if hasattr(opt, 'new_param') else 'Not set'}")