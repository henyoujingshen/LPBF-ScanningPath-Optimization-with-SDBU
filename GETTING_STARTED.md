# Repository guide and reproducibility notes

This repository contains research code for predicting temperature fields and optimizing laser powder bed fusion (L-PBF) scanning paths with deep learning and a genetic algorithm. For the research context, figures, and requested paper citation, start with the [README](README.md).

This guide describes the files currently included and the preparation needed to adapt the research workflow to another machine. It is based on source inspection; it does not certify an end-to-end reproduction of the published results.

## Workflow at a glance

1. **Generate simulation data:** define scanning sequences in MATLAB and run/extract thermal simulations in Abaqus.
2. **Prepare learning inputs:** convert simulation outputs into deposition and temperature matrices.
3. **Train a surrogate model:** use the PyTorch training entry point and the selected model configuration.
4. **Optimize scanning order:** evaluate candidate 5 x 5 island sequences using a trained surrogate inside the genetic algorithm.

## Where to start

| Area | Files | Purpose |
| --- | --- | --- |
| Simulation orchestration | `SLM/main.m`, `SLM/SLM.py`, `SLM/Output.py` | Scanning sequence generation, Abaqus model setup, and result extraction |
| MATLAB conversion | `SLM/data_prepare.m`, `SLM/to_depomatrix.m`, `SLM/to_tempmatrix.m` | Prepare paired deposition/temperature data |
| Training | `main1.py`, `config.py` | Training, fold splits, configuration, logs, and checkpoint output |
| Data loading | `data/dataset.py` | Read data-index files and load CSV matrices |
| Input features | `data/feature_engineering.py` | Feature engineering utilities |
| Neural networks | `models/` | Included CNN, RNN, and U-Net implementations |
| Optimization | `GA_test.py`, `evaluation.py` | Permutation-based genetic algorithm and model-based evaluation |
| Visualization | `visualize.py`, `draw/` | Plotting utilities and supporting material |
| Saved weights | `checkpoints/` | Included model checkpoints; verify architecture compatibility before loading |

## Environment preparation

- The simulation stage requires MATLAB and Abaqus, including access to the Abaqus command-line tools. These are separate from the Python learning environment.
- Use an isolated Python environment and inspect [requirements.txt](requirements.txt) before installation. It records several original dependencies, but is not a complete, frozen environment specification.
- The source also imports `torch`, `scikit-learn`, and `geatpy`; these are not listed in `requirements.txt`. Choose compatible versions for the target Python version, operating system, and CPU/CUDA setup.
- `requirements.txt` references the moving `master` branch of `pytorch/tnt`. Record the resolved version or commit when reproducing an experiment.
- The training pipeline uses Visdom. Configure its server and connection settings before a training run.

## Preparation required in the current snapshot

### Python imports and model selection

`main1.py` imports `utils.visualize`, and `GA_test.py` imports `utils.evaluation`. The corresponding files are currently at the repository root (`visualize.py` and `evaluation.py`), with no `utils/` directory. Align the module layout and imports in your working copy before running these entry points.

`models/__init__.py` imports several modules that are absent from the current tree, including `sdim_cnn1`, `cnn2`, `rnnDNN_nd`, `rnnDNN_att`, and `I3D`. It also imports `unet` while `models/unet.py` defines `UNet`. Reconcile the package exports with the selected implementation: `main1.py` constructs the model through `getattr(models, opt.model)()`.

### Data layout and paths

`config.py` defaults to `./data/processed_slm_data/`; this directory is not included. `DepoTempDataset` expects `root_full_data_server1.txt` for training and `root_data30.txt` for testing. The loader currently selects every 100th line and splits each line at character 92, so these index files are tied to the original path layout. Adapt the parser and index generation together, and verify that every referenced deposition/temperature pair exists.

The MATLAB conversion functions are included under `SLM/`. Several simulation and evaluation scripts contain machine-specific absolute paths; update them for your local working directory. In `SLM/main.m`, the Abaqus model-generation call is commented out, while the script submits `Job_TEMP`. Ensure the required simulation inputs are available before launching jobs.

### Model, features, and checkpoints

The feature-expansion block in `data/dataset.py` is currently commented out. Match input channels, preprocessing, loss selection, and weights to the experiment being reproduced. `config.py` defaults to MSE; `main1.py` also has a selectable `CorrelationCoefficientLoss`. Verify the exact mathematical formulation against the paper before interpreting results as the published SDBU configuration.

`GA_test.py` currently requests `rnnDNN`, while `evaluation.py` loads a hard-coded checkpoint path whose filename is not present in `checkpoints/`. Select matching model definitions and weights explicitly before optimization.

## Recording a reproducible experiment

Keep a record of:

- Repository commit and any local changes.
- Python/package versions, MATLAB/Abaqus versions, and CPU/GPU details.
- Data source, index files, preprocessing, train/test separation, and scanning mode.
- Model architecture, feature channels, loss, hyperparameters, and checkpoint used.
- Random seeds, fold definitions, and evaluation metrics.

`main1.py` seeds the shuffled fold split, but this alone does not make all training operations deterministic. It currently creates five training splits and evaluates each trained model on the fixed test set; the within-fold validation loader is commented out. Account for this when comparing results or extending the evaluation protocol.

Training writes `performance_for_<model>.txt` and checkpoints named `<model>_fold<fold>_epoch<max_epoch>.pth`. Use a separate experiment directory or preserve prior outputs before repeating a run.

## Citation and license

Follow the citation request in the [README](README.md). The repository's existing [Apache License 2.0](LICENSE) remains unchanged.
