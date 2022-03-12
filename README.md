# Neural-network-from-scratch
 An implementation example for neural network from scratch by Python 3.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)

## General Information
- Neural network, is one of concepts in in Machine Learning (ML), and widely uses in nowadays. The example is written to comprehend in the   ML theoretical concept, Feed forwarding, and Backpropagation.    

## Technologies Used
- Python 3
- Tensorflow 2
- Numpy
- Other libraries.

## Setup

- Install [Python version 3.9.0](https://www.python.org/downloads/release/python-390/) and set it to your `PATH` environment.
- Clone this repository.
- Install required dependencies by using this command `pip install -r requirements.txt`.


## Usage
The example code has been separated in two categories, for predict the XOR logical table, and the MNIST dataset classification.
- For the XOR logical table prediction:
    - please run this command `python3 nn_xor`.
- For the MNIST dataset classification:
    - please run this command `python3 nn_mnist.py -e <EPOCH> -lr <LEARNING RATE>`.
        - `EPOCH` is a number of epoch.
        - `LEARNING RATE` is a learning rate parameter.


## Project Status
Project is: _complete_