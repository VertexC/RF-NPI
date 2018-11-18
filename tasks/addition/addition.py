"""
addition.py

Core task-specific model definition file. Sets up encoder model, program embeddings, argument
handling.
"""
from tasks.addition.env.config import CONFIG
from tasks.task_base import TaskBase
import torch
from torch import nn
import torch.nn.functional as F
from tasks.addition.env.config import CONFIG, ScratchPad
import numpy as np

MOVE_PID, WRITE_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1

class AdditionCore(TaskBase):
    def __init__(self, hidden_dim, state_dim, batch_size=1):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.env_dim = CONFIG["ENVIRONMENT_ROW"] * CONFIG["ENVIRONMENT_DEPTH"]  # 4 * 10 = 40
        self.arg_dim = CONFIG["ARGUMENT_NUM"] * CONFIG["ARGUMENT_DEPTH"]        # 3 * 10 = 30
        self.in_dim = self.env_dim + self.arg_dim
        self.prog_embedding_dim = CONFIG["PROGRAM_EMBEDDING_SIZE"]
        self.prog_dim = CONFIG["PROGRAM_SIZE"]
        self.scratchPad = None
        # for f_enc
        self.fenc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fenc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fenc3 = nn.Linear(self.hidden_dim, self.state_dim)
        # for f_env
        self.fenv1 = nn.Embedding(self.prog_dim, self.prog_embedding_dim)
        
        
    def init_scratchPad(self, in1, in2):
        self.scratchPad = ScratchPad(in1, in2)
        print('Scratch Initialized')
    
    def get_args(self, args, arg_in=True):
        if arg_in:
            arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
        else:
            arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                    range(CONFIG["ARGUMENT_NUM"])]
        if len(args) > 0:
            for i in range(CONFIG["ARGUMENT_NUM"]):
                if i >= len(args):
                    arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
                else:
                    arg_vec[i][args[i]] = 1
        else:
            for i in range(CONFIG["ARGUMENT_NUM"]):
                arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
        return [arg_vec.flatten() if arg_in else arg_vec]

    def f_enc(self, env, args):
        merge = torch.cat([env, args], -1)
        elu = F.elu(self.fenc1(merge))
        elu = F.elu(self.fenc2(elu))
        out = self.fenc3(elu)
        return out 

    def get_program_embedding(self, prog_id, args):
        embedding = self.fenv1(prog_id)
        return embedding
    
    def f_env(self, prog_id, args):
        if prog_id == MOVE_PID or prog_id == WRITE_PID:
            self.scratchPad.execute(prog_id, args)
        return [self.scratchPad.get_env()]