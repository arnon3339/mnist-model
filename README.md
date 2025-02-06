<p align="center">
    <h3 align="center">CNN Model for Digit Prediction</h3>
  </a>
</p>

<p align="center">Python project to train dataset with CNN model for digit predcition. It exports tunned model to the backend of <a href="https://github.com/arnon3339/mnist-project.git">Mnits project</a>.</p>

<br/>

## Introduction

The CNN model has to be fine-tunned with hyperparameters. The [Optuna](https://optuna.org/) library is used to fine the best model for digit prediction.

## How it works

### Configurations
The configuration can be found in `config.toml`

```toml
[optimize] # tunned hyperparameters
learning_rate = 0.0013
number_filters = 32
dropout_rate = 0.29
epochs = 10
batch_size = 32

[path] # dataset paths
image_zip = "./assets/images/compress/mnist.zip"
dataset = "./assets/data/dataset.csv"

[path.model] # path of trained model
h5 = "output/models/mnist_cnn.h5"
keras = "output/models/mnist_cnn.keras"
onnx = "output/models/mnist_cnn.onnx" 
tflite = "output/models/mnist_cnn.tflite"

[tunning] # hyperparameters use with Optuna tunning
learning_rate = [1e-4, 1e-2] # range
dropout_rate = [0.1, 0.5] # range
number_filters = [32, 64, 128] # array values
epochs = 10
batch_size = 32
```

### Dataset format

| label | 1x1 | 1x2 | ...  | 28x27 | 28x28 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 0 | 0 | ...  | 0 | 255 |
| 1 | 0 | 255 | ...  | 0 | 0 |
| ... | 0 | 0 | ...  | 0 | 255 |
| 8 | 0 | 255 | ...  | 0 | 255 |
| 9 | 0 | 0 | ...  | 0 | 0 |

The dataset has to be constructed as table above for being read by [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html). The image data needs to be resized as 28x28 pixels of grayscale image. The grayscale image can be flatten by [Numpy](https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.flatten.html) as 1D array. Finally, the data can be represted as suggested table.

The dataset path has to be the same path that show in `config.toml`.

The CNN model use mnist dataset that is the open source. It can be found in [Kraggle](https://www.kaggle.com).

## Getting Started

First, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # restrict with versions
```

Ouput of `python main.py -h`:

```bash
usage: Drawing digit classifier [-h] [-op] [-f] [-t] [-d] [-e] [-cf] [-mf MODEL_FORMAT] [-ed] [-nt NUMBER_TRIALS] [-lr LEARNING_RATE] [-nf NUMBER_FILTERS] [-dr DROPOUT_RATE]

The application use CNN to classify drawing digit.

optional arguments:
  -h, --help            show this help message and exit
  -op, --optimized      use optimized values
  -f, --fit             fit the model
  -t, --tune            fine-tunning the model
  -d, --deploy          deploy application
  -e, --export          export model
  -cf, --convert-format
                        model format h5 -> onnx
  -mf MODEL_FORMAT, --model-format MODEL_FORMAT
                        model format (h5 or onnx)
  -ed, --export-data    export image data to csv dataset
  -nt NUMBER_TRIALS, --number-trials NUMBER_TRIALS
                        Number of trials for tunning
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -nf NUMBER_FILTERS, --number-filters NUMBER_FILTERS
                        Number of CNN filters
  -dr DROPOUT_RATE, --dropout-rate DROPOUT_RATE
                        Drop rate
```

Then, run the development server(python dependencies will be installed automatically here):

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The FastApi server will be running on [http://127.0.0.1:8000](http://127.0.0.1:8000) – feel free to change the port in `package.json` (you'll also need to update it in `next.config.js`).

## Demo

Vercel: https://mnist-project.vercel.app/

## Repositories

Deployment: https://github.com/arnon3339/mnist-project.git  
CNN model: https://github.com/arnon3339/mnist-model.git  
Docker deployment: https://github.com/arnon3339/mnist-model.git
