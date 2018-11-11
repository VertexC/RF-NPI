"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
from tasks.addition.addition import AdditionCore
from tasks.addition.env.config import CONFIG, get_args, ScratchPad
import pickle

MOVE_PID, WRITE_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1
DATA_PATH = "tasks/addition/data/train.pik"
LOG_PATH = "tasks/addition/log/"


def model_train(epochs, verbose=0):
    pass