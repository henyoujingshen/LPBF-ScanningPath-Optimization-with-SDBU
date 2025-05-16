# Laser Powder Bed Fusion (L-PBF) Path Optimization with Deep Learning

This project implements a framework for optimizing scanning paths in Laser Powder Bed Fusion (L-PBF) using a deep learning model (SmallData-Boosted U-Net - SDBU) trained on small datasets. The deep learning model predicts temperature fields based on deposition matrices (scanning paths). A Genetic Algorithm (GA) is then used to find optimal scanning paths that improve thermal uniformity.

The workflow involves:
1.  **Data Generation (MATLAB & Abaqus):** `main.m` orchestrates Abaqus simulations (using `SLM.py` for setup and `Output.py` for result extraction) to generate raw temperature and deposition data for various scanning paths.
2.  **Data Preparation (MATLAB):** `data_prepare.m` processes the raw Abaqus output into structured CSV files (deposition matrices and corresponding temperature fields) and creates index files (`root_*.txt`) for the deep learning pipeline.
3.  **Deep Learning Model Training (Python):** `main1.py` trains a U-Net based model to learn the mapping from deposition matrices to temperature fields. It uses k-fold cross-validation and custom loss functions.
4.  **Scanning Path Optimization (Python):** `GA_test.py` utilizes a trained deep learning model as an evaluation function within a Genetic Algorithm (from `geatpy`) to search for optimal 5x5 scanning path sequences.

![图片1](https://github.com/user-attachments/assets/9defcc9b-dd9c-44a4-9695-73f24addf2f4)
Figure: Scanning Path Optimization Chain with SmallDataBoosted U-Net (SDBU): (a) The methodology encompasses three essential steps: dataset generation, machine learning modeling for a small dataset, and evolutionary algorithm (EA)-based optimization. (b) Comparative performance analysis of SDBU and other feature combinations trained on a small dataset. (c) Comparative analysis of the optimal solutions and predictive accuracy of the optimization chain using SDBU and the original U-Net. (d-e) Scanning orders in strip (left) and island (right) modes for Sequential Island Scanning, LHI Algorithm, and optimal paths identified by the Original U-Net with 10-set and 30-set data, and SDBU with 10-set data (above), along with their corresponding Temperature Uniformity Curves (below) for the five different scanning orders.

## File Descriptions

*   **Paper OCR:** `Optimization of Scanning Path...pdf` - The research paper describing the methodology.
*   **MATLAB Scripts:**
    *   `main.m`: Main script to run Abaqus simulations for generating datasets. It calls `SLM.py` and `Output.py`.
    *   `data_prepare.m`: Script to process Abaqus output data into CSVs and generate `root_*.txt` files for Python training. (Note: Relies on internal functions like `to_depomatrix` and `to_tempmatrix` which are not provided but are essential for its operation).
    *   `SLM.py` (called by `main.m`): Python script for Abaqus to set up the SLM simulation model (geometry, laser path).
    *   `Output.py` (called by `main.m`): Python script for Abaqus to extract simulation results (e.g., temperature data).
*   **Python Scripts (Main ML/GA Pipeline):**
    *   `main1.py`: Core script for training and testing the deep learning model. Uses `fire` for command-line arguments.
    *   `GA_test.py`: Script to run the Genetic Algorithm for path optimization, using a trained model.
    *   `config.py` (imported by `main1.py`): Contains default configuration parameters for training (e.g., learning rate, batch size, model name).
    *   `data/dataset.py`: Defines PyTorch `Dataset` classes (`Depo_Temp_data`, `Valid_data`) for loading and preprocessing the deposition and temperature data.
    *   `data/feature_engineering.py`: Contains classes for feature engineering on the deposition matrix (e.g., distance matrix, t_hiz). (Currently, feature engineering seems commented out in `dataset.py` but is part of the SDBU concept in the paper).
    *   `models/` (directory, contents not provided): This directory should contain Python files defining the neural network architectures (e.g., `unet.py`, as implied by `opt.model`). The paper refers to SDBU (SmallData-Boosted U-Net).
    *   `utils/visualize.py` (imported by `main1.py`): Utility for visualization, likely using Visdom.
    *   `utils/evaluation.py` (imported by `GA_test.py`): Contains the `evaluator` class, which is responsible for loading a trained model and calculating output for the GA.

## Prerequisites

1.  **Software:**
    *   MATLAB
    *   Abaqus (with Python scripting enabled)
    *   Python 3.7+
    *   conda or venv recommended for managing Python environments.
2.  **Python Libraries:**
    *   `torch` (PyTorch)
    *   `torchnet`
    *   `numpy`
    *   `pandas`
    *   `scikit-learn`
    *   `fire`
    *   `visdom` (and a running Visdom server: `python -m visdom.server`)
    *   `ipdb` (optional, for debugging)
    *   `geatpy` (for `GA_test.py`)
    Install them using pip:
    ```bash
    pip install torch torchnet numpy pandas scikit-learn fire visdom ipdb geatpy
    ```
3.  **Abaqus Environment:** Ensure Abaqus can be called from the command line and that its Python interpreter can access necessary libraries if `SLM.py` or `Output.py` require them beyond standard Abaqus Python.
4.  **MATLAB `to_depomatrix` and `to_tempmatrix`:** The `data_prepare.m` script relies on these functions. You need to ensure they are correctly implemented and available in the MATLAB path. They are responsible for:
    *   Reading raw Abaqus outputs.
    *   Generating individual `depo_xxxxx.csv` and `temp_xxxxx.csv` files.
    *   Returning the paths to these generated CSVs, which `data_prepare.m` then writes into the `root_*.txt` files.

## Workflow and Running Instructions

### Step 1: Data Generation and Preparation (MATLAB & Abaqus)

1.  **Configure `main.m`:**
    *   Set `n` (e.g., 5 for a 5x5 grid).
    *   Set `m` (number of different random scanning sequences to generate).
    *   Ensure paths for Abaqus commands (`system('abaqus ...')`) are correct for your system.
    *   Make sure `SLM.py` and `Output.py` are in the same directory as `main.m` or accessible by Abaqus.
2.  **Run `main.m` in MATLAB:**
    ```matlab
    % In MATLAB
    run('main.m');
    ```
    This will:
    *   Generate random scanning paths.
    *   Call Abaqus to run simulations for each path.
    *   Call `Output.py` to extract results (e.g., `TEMP1.csv`, `TEMP2.csv`, ... and `RP5.txt`, `PP5.txt`).
3.  **Configure `data_prepare.m`:**
    *   Set `n_try` to match the `m` value used in `main.m`.
    *   Define `xn_elements`, `yn_elements` (e.g., 50x50 for the temperature field resolution).
    *   Ensure the `to_depomatrix` and `to_tempmatrix` functions correctly process the outputs from Step 1.2 and save them as individual CSV files in a structured way. These functions should create a directory structure that `dataset.py` can later access using the paths stored in `root_*.txt`.
4.  **Run `data_prepare.m` in MATLAB:**
    ```matlab
    % In MATLAB
    run('data_prepare.m');
    ```
    This will:
    *   Call `to_depomatrix` and `to_tempmatrix` to process raw data into `depo_*.csv` and `temp_*.csv` files.
    *   Create `root_data<n_try>.txt` (e.g., `root_data3.txt`). This file will contain pairs of paths to the processed `depo_*.csv` and `temp_*.csv` files, one pair per line. Example line: `/path/to/your/data_root/depo_001.csv, /path/to/your/data_root/temp_001.csv`. The paths should be structured such that `dataset.py` can find them based on `opt.train_data_root`.
    *   It's crucial that `root_data30.txt` (for testing in `main1.py`) and `root_full_data_server1.txt` (for training in `main1.py`) are generated or renamed from the output of `data_prepare.m` and placed in the `opt.train_data_root` directory.

### Step 2: Deep Learning Model Training (`main1.py`)

1.  **Organize Data:**
    *   Create a main data directory (e.g., `./pytorch_data/`). This will be your `opt.train_data_root`.
    *   Place the `root_full_data_server1.txt` (for training/validation) and `root_data30.txt` (for testing) files generated in Step 1.4 into this `opt.train_data_root` directory.
    *   Ensure all the individual `depo_*.csv` and `temp_*.csv` files referenced in these `root_*.txt` files are accessible (e.g., also within `opt.train_data_root` or via absolute paths in the `root_*.txt` files).
2.  **Prepare Model Files:**
    *   Create a `models/` directory.
    *   Place your model definition Python files (e.g., `unet.py`, or a file defining `unet_SDBU` as suggested by the paper and checkpoint naming) inside this `models/` directory. The model class name in the file should match what you pass via `--model` argument.
3.  **Configure `config.py`:**
    *   Review default parameters in `config.py`. Many of these can be overridden via command-line arguments.
    *   Key defaults:
        *   `opt.train_data_root = '/path/to/your/data_root/'` (Update this or provide via CLI)
        *   `opt.model = 'YourModelName'` (e.g., `unet_SDBU`)
        *   `opt.env = 'lpbf_optimization'` (Visdom environment)
        *   `opt.use_gpu = True`
4.  **Start Visdom Server (in a separate terminal):**
    ```bash
    python -m visdom.server
    ```
5.  **Run `main1.py`:**
    The `train_test` function is the main entry point, callable via `fire`.
    ```bash
    python main1.py train_test --train_data_root ./pytorch_data/ --model unet_SDBU --max_epoch 100 --lr 0.001 --batch_size 32 --env my_experiment_sdbu
    ```
    **Key Command-Line Arguments for `train_test`:**
    *   `--train_data_root`: (Required if not set correctly in `config.py`) Path to the directory containing `root_*.txt` files and the CSV data.
    *   `--model`: (Required) Name of the model class to load from the `models` package (e.g., `unet_SDBU`).
    *   `--env`: Visdom environment name for plotting. Default is `opt.env`.
    *   `--use_gpu`: `True` or `False`. Default is `opt.use_gpu`.
    *   `--max_epoch`: Number of training epochs. Default is `opt.max_epoch`.
    *   `--lr`: Initial learning rate. Default is `opt.lr`.
    *   `--batch_size`: Training batch size. Default is `opt.batch_size`.
    *   `--num_workers`: Number of CPU workers for data loading. Default is `opt.num_workers`.
    *   `--print_freq`: How often to print loss and update Visdom plots. Default is `opt.print_freq`.
    *   `--lr_decay`: Learning rate decay factor. Default is `opt.lr_decay`.
    *   `--weight_decay`: Weight decay for Adam optimizer. Default is `opt.weight_decay`.

    **Output:**
    *   Training progress and loss values printed to the console.
    *   Loss plots updated in Visdom (accessible at `http://localhost:8097`).
    *   A performance log file: `performance_for_<model_name>.txt` (e.g., `performance_for_unet_SDBU.txt`) containing epoch losses and final test accuracies for each fold.
    *   Saved model checkpoints in the `checkpoints/` directory (e.g., `checkpoints/unet_SDBU_model_island_SDBU`).

### Step 3: Scanning Path Optimization (`GA_test.py`)

1.  **Trained Model:** Ensure you have a trained model checkpoint from Step 2 (e.g., `checkpoints/rnnDNN_model_island_SDBU` if you trained a model named `rnnDNN`, or adapt `GA_test.py` to load the SDBU model).
2.  **`utils/evaluation.py`:** The `evaluator` class in this file needs to be able to:
    *   Load the specific pre-trained model (e.g., by taking the model name `rnnDNN` and checkpoint path as arguments).
    *   Preprocess the GA candidate solution `x` (a 5x5 matrix) into the format expected by the loaded model.
    *   Perform inference using the model.
    *   Return a scalar fitness value.
    *   *This file is not provided, so its implementation is assumed.*
3.  **Run `GA_test.py`:**
    ```bash
    python GA_test.py
    ```
    **Configuration within `GA_test.py`:**
    *   `NIND`: Population size.
    *   `myAlgorithm.MAXGEN`: Maximum number of generations.
    *   The model name `'rnnDNN'` is hardcoded in `ml_modeling`. You might need to change this or make it configurable if you trained a different model (e.g., `unet_SDBU`). The `evaluator` class will need to handle loading this model.

    **Output:**
    *   GA progress.
    *   Best objective function value and corresponding control variable values (the optimized 5x5 scanning path).

## Notes and Customization

*   **Dataset Pathing:** The most critical setup step is ensuring the data generated by MATLAB is correctly structured and that the paths in `root_*.txt` files accurately point to the `depo_*.csv` and `temp_*.csv` files, relative to `opt.train_data_root` or as absolute paths.
*   **Model Architecture:** The specific model architecture (e.g., `unet_SDBU`) needs to be defined in a Python file within the `models/` directory.
*   **Loss Function:** `main1.py` uses `MSELoss` by default. It also defines `CorrelationCoefficientLoss`. The paper highlights a tailored loss function (Cor \* R, likely `CorrelationCoefficientLoss` multiplied by some R-value representation) for SDBU. You might need to modify `main1.py` to use this combined loss if it's not already doing so when SDBU is selected (the current code seems to only use `criterion = t.nn.MSELoss()`). The SDBU method mentioned in the paper (Section 3.4, Page 7, and Section 4.2, Page 21) involves using "Cor \* R as the training loss and applying feature expansion to both D and H features."
*   **Feature Engineering:** The paper emphasizes physics-based feature engineering (D and H matrices from `feature_engineering.py`). This is currently commented out in `dataset.py`. To fully implement SDBU as per the paper, you would need to uncomment and integrate this feature engineering step into the `Depo_Temp_data.__getitem__` method and adjust the model's input channels accordingly.
*   **`evaluator` in `GA_test.py`:** This is a key component for the GA to work. It acts as the bridge between the GA and your trained PyTorch model.
*   **Error `Depo_Temp_data` vs `DepoTempDataset`:**
    In `main1.py`:
    ```python
    from data.dataset import Depo_Temp_data # This is used
    from data.dataset import Temp_Temp_data # This is unused
    from data.dataset import Valid_data     # This is unused
    ```
    In `dataset.py`, the class is named `DepoTempDataset`.
    You'll need to make these consistent. Either change the import in `main1.py` to `from data.dataset import DepoTempDataset as Depo_Temp_data` or rename the class in `dataset.py` to `Depo_Temp_data`. The former is generally preferred to avoid modifying library code if possible.
    The same applies to `Valid_data` if you intend to use it (class name `ValidDataset` in `dataset.py`).

This README should provide a good starting point for understanding and running the project. The data generation and preparation steps are complex and require careful attention to detail.
