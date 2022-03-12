import tensorflow as tf
from losses import mse, mse_prime
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import *
import utils
import numpy as np
import argparse