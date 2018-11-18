"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
from tasks.addition.addition import AdditionCore
from tasks.addition.env.config import CONFIG, get_args, ScratchPad
import pickle


def model_train(epochs, verbose=0):
    pass