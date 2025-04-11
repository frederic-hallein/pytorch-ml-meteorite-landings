# Pytorch ML Meteorite Landings

## Description

A machine learning project utilizing PyTorch to analyze and predict meteorite classes based on [The Meteoritical Society ](https://www.kaggle.com/datasets/nasa/meteorite-landings?resource=download) dataset of over 45,000 meteorite impacts worldwide. The neural network contains a total of 7 layers, mainly consisting of linear, dropout, and ReLU activation function layers, in a Sequential container.

## Installation (Linux)

1. Clone the repository:

```bash
 git clone git@github.com:frederic-hallein/pytorch-ml-meteorite-landings.git
```

2. Inside the root directory, create a Python virtual environment:

```bash
python3 -m venv .venv
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Using ```pip3```, install all dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

To train, test and evaluate the model:

```bash
python3 main.py
```

## Extra: How to Run Tests

To run all unit tests:

```bash
python3 -m pytest tests
```

To run specific tests:

```bash
python3 -m pytest tests/test_<package_name>.py
```

where ```package_name``` currently only support the following options:
- ```data_loader```
- ```data_transform```

