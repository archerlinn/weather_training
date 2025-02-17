# Weather Forecasting Using U-Net

## Overview
This project leverages a U-Net model to predict weather patterns from CFD (Computational Fluid Dynamics) simulation data. It uses PyTorch for deep learning, Weights & Biases for experiment tracking, and NRRD files as the primary data format.

## Features
- Implements a **U-Net** model for weather data prediction
- Supports **training and validation** with **custom dataset preprocessing**
- Uses **adaptive pooling** to handle variable input sizes
- **Tracks experiments** via Weights & Biases (WandB)
- Includes **automatic model checkpointing**
- Provides **visualization tools** for comparing predictions with ground truth

## Installation
To set up the project, install the required dependencies:

```bash
pip install torch torchvision tqdm pynrrd wandb matplotlib
```

## Data Preparation
The dataset consists of **NRRD files** containing CFD simulation data. These files are preprocessed and normalized using the `WeatherDataset` class.

### Dataset Structure
```
/weather_dataset
   ├── CFD_Subset
   │   ├── file1.nrrd
   │   ├── file2.nrrd
   │   ├── ...
```

### Normalization Strategy
- Mean and standard deviation are computed across all samples.
- Data is normalized using `(data - mean) / std`.

## Model Architecture
The model is a **U-Net** with an encoder-decoder structure:
- **Encoder:** Convolutional blocks with batch normalization and pooling.
- **Bottleneck:** Deeper feature extraction layer.
- **Decoder:** Transpose convolutions with skip connections.
- **Output:** A final 1x1 convolution layer to predict weather parameters.

## Training Process
To train the model, run:

```bash
python train.py
```

### Hyperparameters:
- **Batch Size**: `1`
- **Optimizer**: `Adam (lr=0.001)`
- **Loss Function**: `MSELoss`
- **Epochs**: `10000`
- **Device**: `CUDA` (if available)

### Model Checkpoints:
- Saves the model weights after each epoch.
- Resumes training from the last saved checkpoint if available.

## Visualization
After each validation step, the predictions are visualized using Matplotlib:

```python
plot_prediction_vs_ground_truth(prediction, ground_truth)
```

## Logging with WandB
This project integrates Weights & Biases for logging loss and training metrics. To use it:

1. Log in using `wandb.login()`
2. View experiment results at:
   - [Project Dashboard](https://wandb.ai/your-username/weather_forecast)

