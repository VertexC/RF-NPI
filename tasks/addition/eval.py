"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from tasks.addition.addition import AdditionCore
from tasks.addition.env.config import CONFIG, get_args, PROGRAM_SET, ScratchPad
import numpy as np
import pickle

LOG_PATH = "tasks/addition/log/"
CKPT_PATH = "tasks/addition/log/model.ckpt"
TEST_PATH = "tasks/addition/data/test.pik"
MOVE_PID, WRITE_PID = 0, 1
W_PTRS = {0: "OUT", 1: "CARRY"}
PTRS = {0: "IN1_PTR", 1: "IN2_PTR", 2: "CARRY_PTR", 3: "OUT_PTR"}
R_L = {0: "LEFT", 1: "RIGHT"}


def model_eval():
    pass