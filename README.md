<p align="center">
    <h3 align="center">CNN Model for Digit Prediction</h3>
  </a>
</p>

<p align="center">Python project to train dataset with CNN model for digit predcition. It exports tunned model to the backend of <a href="https://github.com/arnon3339/mnist-project.git">Mnits project</a>.</p>

<br/>

## Introduction

This project utilizes a **Convolutional Neural Network (CNN)** for digit classification. The model undergoes **hyperparameter tuning** using the [Optuna](https://optuna.org/) library to find the best-performing parameters for digit recognition.

## How it works

### Configuration
The configuration settings are stored in `config.toml`:

```toml
[default]
learning_rate = 1e-4
number_filters = 32
dropout_rate = 0.2
epochs = 10
batch_size = 32
export = "keras"
number_trails = 10

[tuned]  # Tuned hyperparameters
learning_rate = 0.0013
number_filters = 32
dropout_rate = 0.29
epochs = 10
batch_size = 32
export = "keras"
number_trials = 10

[path]  # Dataset paths
image_zip = "./assets/images/compress/mnist.zip"
dataset = "./assets/data/dataset.csv"

[path.model]  # Path of trained model
h5 = "output/models/mnist_cnn.h5"
keras = "output/models/mnist_cnn.keras"
onnx = "output/models/mnist_cnn.onnx"
tflite = "output/models/mnist_cnn.tflite"

[tuning]  # Hyperparameters used with Optuna tuning
learning_rate = [1e-4, 1e-2]  # Range
dropout_rate = [0.1, 0.5]  # Range
number_filters = [32, 64, 128]  # Possible values
epochs = 10
batch_size = 32
number_trials = 10
```

### Dataset format

| label | 1x1 | 1x2 | ...  | 28x27 | 28x28 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 0 | 0 | ...  | 0 | 255 |
| 1 | 0 | 255 | ...  | 0 | 0 |
| ... | 0 | 0 | ...  | 0 | 255 |
| 8 | 0 | 255 | ...  | 0 | 255 |
| 9 | 0 | 0 | ...  | 0 | 0 |


The dataset is structured as a **Pandas DataFrame**, where each row represents a **flattened 28×28 grayscale image** with pixel values between 0 and 255. Images must be resized to 28x28 pixels and stored in the dataset path specified in `config.toml`.

The MNIST dataset used for training is open-source and available on [Kraggle](https://www.kaggle.com).

## Getting Started

### &#x31;&#xFE0F;&#x20E3; Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # restrict with versions
```

### &#x32;&#xFE0F;&#x20E3; Usage
Check available commands using:
Ouput of `python main.py -h`:

```bash
usage: Drawing digit classifier [-h] [-op] [-f] [-t] [-e] [-cf] [-mf MODEL_FORMAT]

The application use CNN to classify drawing digit.

optional arguments:
  -h, --help            show this help message and exit
  -tn, --tuned          use optimized values
  -f, --fit             fit the model
  -t, --tune            fine-tunning the model
  -e, --export          export model
  -cf, --convert-format
                        model format h5 -> onnx
  -mf MODEL_FORMAT, --model-format MODEL_FORMAT
                        model format (h5 or onnx)
```

### &#x33;&#xFE0F;&#x20E3; Example Commands

The hyperparameters and other values can be adjusted through `config.toml`.

### Fine-Tune the Model
```bash
python main.py -t
```

### Train the Model with fine-tuned Hyperparameters
```bash
python main.py -tn -f
```

### Convert Model Format to ONNX
```bash
python main.py -cf -mf onnx
```

### &#x34;&#xFE0F;&#x20E3; Adjust Hyperparameters
Modify the hyperparameters and dataset paths in `config.toml` as needed.

## Demo

Live deployment available on Vercel:
👉 [MNIST Project](https://mnist-project.vercel.app/)

## Repositories

- Deployment: [mnist-project](https://github.com/arnon3339/mnist-project.git)
- CNN model: [mnits-model](https://github.com/arnon3339/mnist-model.git)
- Docker deployment: *(Coming soon)*

---

🚀 **Now you're ready to train and deploy your CNN model for digit recognition!**

---